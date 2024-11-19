from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Generator
import gc
import logging
import os
import numpy as np
import pandas as pd
import requests
import requests.exceptions

from backoff import expo, on_exception
from ratelimit import limits, sleep_and_retry
from config import TrackingConfig

os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '1200'
import mlflow

# Slightly over the limit to account for timing variations
CALLS = 16
RATE_LIMIT = 1  # 1 second

@lru_cache(maxsize=8096)
@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
@on_exception(expo, 
             requests.exceptions.RequestException, 
             max_tries=10,
             max_time=30)
def query_ensembl(chrom: str, pos: int) -> Optional[List[Dict[str, Any]]]:
    """
    Query the Ensembl REST API for genes overlapping a specific genomic position.
    Rate limited to 16 requests per second with aggressive retry strategy.

    Args:
        chrom (str): Chromosome (e.g., '1', 'X').
        pos (int): Genomic position.

    Returns:
        Optional[List[Dict[str, Any]]]: List of gene data or None if the query fails.
    """
    server = "https://rest.ensembl.org"
    ext = f"/overlap/region/human/{chrom}:{pos}-{pos+1}?feature=gene"
    
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers, timeout=10)
    
    if response.ok:
        return response.json()
    elif response.status_code == 429:
        raise requests.exceptions.RequestException(f"Rate limit exceeded: {response.status_code}")
    else:
        logging.getLogger(__name__).warning(f"Ensembl query failed for {chrom}:{pos} with status code {response.status_code}")
        return None

@lru_cache(maxsize=8096)
def map_feature_to_gene_symbol(feature_name: str) -> str:
    """
    Map a feature name to its corresponding gene symbol(s) using Ensembl.

    Args:
        feature_name (str): Feature name in the format 'chr<chrom>:<pos>' (e.g., 'chr1:123456').

    Returns:
        str: Comma-separated gene symbols or an error message.
    """
    try:
        chromosome, position = feature_name.replace('chr', '').split(':')
        pos = int(position)
        gene_data = query_ensembl(chromosome, pos)
        if gene_data:
            gene_names = [gene.get('external_name', 'Symbol not found') for gene in gene_data if 'external_name' in gene]
            return ', '.join(gene_names) if gene_names else 'Symbol not found'
        else:
            return 'N/A'
    except Exception as e:
        logging.getLogger(__name__).error(f"Error mapping feature '{feature_name}' to gene symbol: {e}")
        return 'Error parsing'

def fetch_gene_symbols_threaded(feature_names: List[str]) -> List[str]:
    """
    Fetch gene symbols for a list of feature names using multithreading.

    Args:
        feature_names (List[str]): List of feature names in the format 'chr<chrom>:<pos>'.

    Returns:
        List[str]: List of gene symbols corresponding to the feature names.
    """
    gene_symbols = []
    max_workers = 20  # Adjust based on system capabilities and API rate limits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feature = {executor.submit(map_feature_to_gene_symbol, feature): feature for feature in feature_names}
        for future in as_completed(future_to_feature):
            feature = future_to_feature[future]
            try:
                gene_symbol = future.result()
            except Exception as e:
                logging.getLogger(__name__).error(f"Error fetching gene symbol for {feature}: {e}")
                gene_symbol = 'Error fetching'
            gene_symbols.append(gene_symbol)
    
    return gene_symbols

def save_summary(summaries: Dict[str, Any], tracking_config: TrackingConfig, outer_iteration: int = None, run_name: str = "summary"):
    """
    Save the summarized DataFrames to MLflow as a new run.
    Handles multiple summary DataFrames, including dictionaries of DataFrames.
    Adds a 'gene_symbol' column if 'feature_name' is present.

    Args:
        summaries: Dictionary containing summarized DataFrames.
        tracking_config: Configuration for MLflow tracking.
        outer_iteration: The outer bootstrap iteration number.
    """
    mlflow.set_tracking_uri(tracking_config.tracking_uri)
    mlflow.set_experiment(tracking_config.experiment_name)

    with mlflow.start_run(run_name=run_name, nested=True):
        if outer_iteration is not None:
            mlflow.set_tag("outer_iteration", outer_iteration)
        for name, df in summaries.items():
            if isinstance(df, dict):
                # Handle dictionaries of DataFrames (e.g., feature_importances_summary)
                for sub_name, sub_df in df.items():
                    if sub_df.empty:
                        mlflow.log_text(f"{name}_{sub_name} summary is empty.", 
                                      artifact_file=f"summaries/{name}_{sub_name}_empty.txt")
                        continue

                    # Add 'gene_symbol' column if 'feature_name' exists using threading
                    if 'feature_name' in sub_df.columns:
                        logging.getLogger(__name__).info(f"Adding 'gene_symbol' to {name}_{sub_name}_summary using threading")
                        feature_names = sub_df['feature_name'].tolist()
                        gene_symbols = fetch_gene_symbols_threaded(feature_names)
                        feature_idx = sub_df.columns.get_loc('feature_name') + 1
                        sub_df.insert(feature_idx, 'gene_symbol', gene_symbols)

                    # Save DataFrame as a CSV artifact
                    temp_path = f"{name}_{sub_name}_summary.csv" if 'summary' not in sub_name and 'summary' not in name else f"{name}_{sub_name}.csv"
                    sub_df.to_csv(temp_path, index=False)
                    mlflow.log_artifact(temp_path)
                    os.remove(temp_path)
            else:
                if df.empty:
                    mlflow.log_text(f"{name} summary is empty.", 
                                  artifact_file=f"summaries/{name}_empty.txt")
                    continue

                # Add 'gene_symbol' column if 'feature_name' exists using threading
                if 'feature_name' in df.columns:
                    logging.getLogger(__name__).info(f"Adding 'gene_symbol' to {name}_summary using threading")
                    feature_names = df['feature_name'].tolist()
                    gene_symbols = fetch_gene_symbols_threaded(feature_names)
                    feature_idx = df.columns.get_loc('feature_name') + 1
                    df.insert(feature_idx, 'gene_symbol', gene_symbols)

                # Save DataFrame as a CSV artifact
                if name.startswith('outer_bootstrap_'):
                    temp_path = f"{name}.csv"  # Only change: don't append _summary for outer bootstrap files
                else:
                    temp_path = f"{name}_summary.csv" if 'summary' not in name else f"{name}.csv"
                
                df.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path)
                os.remove(temp_path)

                # Log some basic metrics about the summary
                mlflow.log_metric(f"{name}_row_count", len(df))
                mlflow.log_metric(f"{name}_column_count", len(df.columns))

        print(f"Saved summaries to MLflow experiment: {tracking_config.experiment_name}")

class BootstrapSummarizer:
    """
    Class to summarize fetched bootstrap run data.
    """

    def __init__(self, run_data: List[Dict[str, Any]]):
        self.run_data = run_data
        self.metrics_df = self._construct_metrics_df()
        self.feature_importances_df = self._construct_feature_importances_df()
        self.crossval_sample_metrics_df = self._construct_sample_metrics_df(sample_type='crossval')
        self.holdout_sample_metrics_df = self._construct_sample_metrics_df(sample_type='holdout')
        self.long_metrics_df = self._reshape_metrics_df()

    def _construct_metrics_df(self) -> pd.DataFrame:
        metrics_records = []
        for run in self.run_data:
            record = {
                'run_id': run['run_id'],
                'model_name': run['run_name'],
                'bootstrap_iteration': int(run['params'].get('iteration', -1))
            }
            for key, value in run['metrics'].items():
                record[key] = value
            metrics_records.append(record)
        metrics_df = pd.DataFrame(metrics_records)
        logging.getLogger(__name__).debug(f"Metrics DataFrame Columns: {metrics_df.columns.tolist()}")
        logging.getLogger(__name__).debug(f"Metrics DataFrame Head:\n{metrics_df.head()}")
        return metrics_df

    def _construct_feature_importances_df(self) -> pd.DataFrame:
        """
        Constructs a DataFrame containing feature importances from all runs.
        Preserves class-specific probabilities if they exist in the original data.

        Returns:
            A DataFrame containing feature importances and optionally class probabilities.
        """
        logger = logging.getLogger(__name__)
        fi_records = []
        
        for run in self.run_data:
            fi_df = run['artifacts'].get('feature_importances.json')
            if fi_df is not None and not fi_df.empty:
                fi_df = fi_df.copy()
                fi_df['run_id'] = run['run_id']
                fi_records.append(fi_df)
        
        if fi_records:
            feature_importances_df = pd.concat(fi_records, ignore_index=True)
            logger.debug(f"Feature Importances DataFrame Columns: {feature_importances_df.columns.tolist()}")
            logger.debug(f"Feature Importances DataFrame Head:\n{feature_importances_df.head()}")
            return feature_importances_df
        else:
            return pd.DataFrame()

    def _construct_sample_metrics_df(self, sample_type: str) -> pd.DataFrame:
        """
        Constructs a DataFrame containing sample metrics from all runs.

        Returns:
            A concatenated DataFrame of all sample metrics.
        """
        logger = logging.getLogger(__name__)
        sample_metrics_records = []
        for run in self.run_data:
            dataset_types = [
                artifact_path for artifact_path in run['artifacts']
                if 'sample_metrics' in artifact_path and (
                    (sample_type == 'crossval' and 'test' in artifact_path) or
                    (sample_type == 'holdout' and 'holdout' in artifact_path)
                )
            ]
            for dataset_type in dataset_types:
                sample_metrics = run['artifacts'].get(dataset_type)
                table = dataset_type.split('/')[-1].replace('_sample_metrics.json', '')
                if sample_metrics is not None and not sample_metrics.empty:
                    sample_metrics = sample_metrics.copy()
                    sample_metrics['run_id'] = run['run_id']
                    # Determine dataset based on the artifact name
                    dataset = 'train' if 'train' in dataset_type else 'test' if 'test' in dataset_type else 'holdout'
                    sample_metrics['dataset'] = dataset
                    sample_metrics['table'] = table
                    sample_metrics_records.append(sample_metrics)
        if sample_metrics_records:
            sample_metrics_df = pd.concat(sample_metrics_records, ignore_index=True)
            # Log the structure of sample_metrics_df
            logger.debug(f"Sample Metrics DataFrame Columns: {sample_metrics_df.columns.tolist()}")
            logger.debug(f"Sample Metrics DataFrame Head:\n{sample_metrics_df.head()}")
            return sample_metrics_df
        else:
            logger.warning(f"No {sample_type} sample metrics found in any run.")
            return pd.DataFrame()

    def _reshape_metrics_df(self) -> pd.DataFrame:
        """
        Reshape the metrics DataFrame from wide to long format.
        Split the metric keys into dataset, table (if any), and metric name.

        Returns:
            A DataFrame in long format with columns: run_id, model_name, bootstrap_iteration, dataset, table, metric, value
        """
        reshaped_records = []
        for _, row in self.metrics_df.iterrows():
            run_id = row['run_id']
            model_name = row['model_name']
            bootstrap_iteration = row['bootstrap_iteration']
            for key, value in row.items():
                if key in ['run_id', 'model_name', 'bootstrap_iteration']:
                    continue
                parts = key.split('/')
                if len(parts) == 2:
                    dataset, metric = parts
                    table = None
                elif len(parts) == 3 and parts[0] == 'holdout':
                    dataset, table, metric = parts
                elif len(parts) == 3 and parts[0] != 'holdout':
                    dataset, metric_group, metric = parts
                elif len(parts) == 4 and parts[0] == 'holdout':
                    dataset, table, metric_group, metric = parts
                else:
                    # Unexpected format
                    continue
                reshaped_records.append({
                    'run_id': run_id,
                    'model_name': model_name,
                    'bootstrap_iteration': bootstrap_iteration,
                    'dataset': dataset,
                    'table': table,
                    'metric': metric,
                    'value': value
                })
        long_metrics_df = pd.DataFrame(reshaped_records)
        logging.getLogger(__name__).debug(f"Long Metrics DataFrame Columns: {long_metrics_df.columns.tolist()}")
        logging.getLogger(__name__).debug(f"Long Metrics DataFrame Head:\n{long_metrics_df.head()}")
        return long_metrics_df

    def summarize_model_metrics(self) -> pd.DataFrame:
        """
        Summarize model metrics aggregated by model, dataset, and metric.

        Returns:
            A DataFrame containing mean and std of each metric per model and dataset.
        """
        logger = logging.getLogger(__name__)
        # Define the metrics to aggregate across datasets
        metrics_to_aggregate = ['f1', 'case_accuracy', 'control_accuracy', 'mcc', 'roc_auc', 'pr_auc']

        # Filter the long_metrics_df for relevant metrics
        filtered_metrics = self.long_metrics_df[self.long_metrics_df['metric'].isin(metrics_to_aggregate)].copy()

        # Group by model_name, dataset, metric
        grouped = filtered_metrics.groupby(['model_name', 'dataset', 'metric'])

        # Aggregate mean and std
        aggregation = grouped['value'].agg(['mean', 'std']).reset_index()

        # Pivot the table to have metrics as columns with mean and std
        summary = aggregation.pivot_table(
            index=['model_name', 'dataset'],
            columns='metric',
            values=['mean', 'std']
        )

        # Flatten the MultiIndex columns
        try:
            summary.columns = [f"{metric}_{stat}" for stat, metric in summary.columns]
        except Exception as e:
            logger.error(f"Error flattening columns in summarize_model_metrics: {e}")
            raise

        summary.reset_index(inplace=True)

        logger.debug(f"Model Metrics Summary Columns: {summary.columns.tolist()}")
        logger.debug(f"Model Metrics Summary Head:\n{summary.head()}")

        return summary

    def summarize_holdout_metrics_per_table(self) -> pd.DataFrame:
        """
        Summarize holdout metrics aggregated by model, table, and metric.

        Returns:
            A DataFrame containing mean and std of each metric per model and table.
        """
        logger = logging.getLogger(__name__)
        # Define the metrics to aggregate
        metrics_to_aggregate = ['case_accuracy', 'control_accuracy']

        # Filter for holdout dataset and relevant metrics
        holdout_metrics = self.long_metrics_df[
            (self.long_metrics_df['dataset'] == 'holdout') &
            (self.long_metrics_df['metric'].isin(metrics_to_aggregate))
        ].copy()

        # Drop runs without table information
        before_drop = len(holdout_metrics)
        holdout_metrics.dropna(subset=['table'], inplace=True)
        after_drop = len(holdout_metrics)
        logger.debug(f"Dropped {before_drop - after_drop} holdout metric rows without table information.")

        if holdout_metrics.empty:
            logger.warning("No holdout metrics with table information found. Skipping summarization.")
            return pd.DataFrame()

        # Group by model_name, table, metric
        grouped = holdout_metrics.groupby(['model_name', 'table', 'metric'])

        # Aggregate mean and std
        aggregation = grouped['value'].agg(['mean', 'std']).reset_index()

        # Pivot the table to have metrics as columns with mean and std
        summary = aggregation.pivot_table(
            index=['model_name', 'table'],
            columns='metric',
            values=['mean', 'std']
        )

        # Flatten the MultiIndex columns
        try:
            summary.columns = [f"{metric}_{stat}" for stat, metric in summary.columns]
        except Exception as e:
            logger.error(f"Error flattening columns in summarize_holdout_metrics_per_table: {e}")
            raise

        summary.reset_index(inplace=True)

        logger.debug(f"Holdout Metrics per Table Summary Columns: {summary.columns.tolist()}")
        logger.debug(f"Holdout Metrics per Table Summary Head:\n{summary.head()}")

        return summary

    def summarize_sample_metrics(self, sample_type: str) -> pd.DataFrame:
        """
        Summarize sample metrics separated by model and indexed by sample_id.
        Maintains consistent column ordering by keeping metrics paired with their standard deviations.

        Args:
            sample_type (str): Either 'crossval' or 'holdout'

        Returns:
            pd.DataFrame: DataFrame containing summarized metrics with consistent ordering:
                - ID columns (sample_id, label, [table])
                - Model-specific metric columns in pairs (metric_model, metric_model_std)
        """
        logger = logging.getLogger(__name__)
        
        # Get appropriate metrics DataFrame based on sample type
        sample_metrics_df = (self.crossval_sample_metrics_df if sample_type == 'crossval' 
                            else self.holdout_sample_metrics_df)

        if sample_metrics_df.empty:
            logger.warning(f"{sample_type.capitalize()} sample metrics DataFrame is empty. Skipping summarization.")
            return pd.DataFrame()

        try:
            # Reset index if it exists to ensure sample_id is a column
            if sample_metrics_df.index.name:
                sample_metrics_df = sample_metrics_df.reset_index()
            
            # Identify the sample ID column
            sample_id_col = next((col for col in ['sample_id', 'index'] if col in sample_metrics_df.columns), None)
            if sample_id_col is None:
                logger.error("Could not find sample ID column")
                return pd.DataFrame()
            
            if sample_id_col != 'sample_id':
                sample_metrics_df = sample_metrics_df.rename(columns={sample_id_col: 'sample_id'})

            # Define ID columns including table for holdout metrics
            id_vars = ['run_id', 'sample_id', 'label']
            if sample_type == 'holdout' and 'table' in sample_metrics_df.columns:
                id_vars.append('table')
            
            # Get metric columns by excluding non-metric columns
            exclude_cols = ['run_id', 'dataset', 'sample_id', 'label', 'y_pred', 'table']
            metric_cols = [col for col in sample_metrics_df.columns if col not in exclude_cols]

            if not metric_cols:
                logger.warning(f"No metric columns found in {sample_type} sample_metrics_df")
                return pd.DataFrame()

            # Melt metrics into long format
            melted_df = sample_metrics_df.melt(
                id_vars=id_vars,
                value_vars=metric_cols,
                var_name='metric_name',
                value_name='value'
            )

            # Merge with metrics_df to get model_name
            merged_df = pd.merge(
                melted_df,
                self.metrics_df[['run_id', 'model_name']],
                on='run_id',
                how='left'
            )

            # Define grouping columns including table if present
            group_cols = ['sample_id', 'label', 'model_name', 'metric_name']
            if 'table' in merged_df.columns:
                group_cols.append('table')

            # Group and aggregate
            agg_df = merged_df.groupby(group_cols)['value'].agg([
                'mean',
                'std'
            ]).reset_index()

            # Create model-specific metric names
            agg_df['metric_model'] = agg_df['metric_name'] + '_' + agg_df['model_name']

            # Get the sample_id to label mapping from the original data
            sample_label_map = merged_df[['sample_id', 'label']].drop_duplicates()
            
            # Create result DataFrame starting with sample_id and label
            result_df = sample_label_map.copy()

            # Add table column if present
            if 'table' in group_cols:
                table_info = merged_df.groupby('sample_id')['table'].first()
                result_df = result_df.merge(table_info.to_frame(), left_on='sample_id', right_index=True)

            # Add model-specific metrics and their standard deviations as pairs
            for metric_model in sorted(agg_df['metric_model'].unique()):
                mask = agg_df['metric_model'] == metric_model
                metric_data = agg_df[mask].set_index('sample_id')
                result_df[metric_model] = result_df['sample_id'].map(metric_data['mean'])
                result_df[f"{metric_model}_std"] = result_df['sample_id'].map(metric_data['std'])

            return result_df

        except Exception as e:
            logger.error(f"Error in summarize_sample_metrics: {str(e)}")
            logger.debug("DataFrame info:", exc_info=True)
            raise

    def summarize_feature_importances(self) -> Dict[str, pd.DataFrame]:
        """
        Summarize feature importances aggregated by model and feature.
        Handles both single-feature_importances and multi-class feature importances.
        For models like MultinomialNB with logprob feature importances, 
        it aggregates across classes to produce a single feature_importances column.
        Aggregates standard deviations appropriately when combining across models.
        Includes an 'all_models' summary with normalized feature importances.
    
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing:
                - One DataFrame per model with feature importance summaries
                - An 'all_models' entry with normalized overall feature importances
        """
        logger = logging.getLogger(__name__)
        feature_importance_summaries = {}

        if self.feature_importances_df.empty:
            logger.warning("Feature importances DataFrame is empty. Skipping summarization.")
            return feature_importance_summaries

        # Merge with metrics_df to get model_name
        merged_fi_df = pd.merge(
            self.feature_importances_df,
            self.metrics_df[['run_id', 'model_name']],
            on='run_id',
            how='left'
        )

        # Process each model separately
        for model in merged_fi_df['model_name'].unique():
            model_fi_df = merged_fi_df[merged_fi_df['model_name'] == model].copy()
            
            if model_fi_df.empty:
                continue

            # Initialize aggregation dictionary with feature_importances
            agg_dict = {
                'feature_importances': ['mean', 'std']
            }
            
            # Add class columns to aggregation only if they contain non-null values
            class_cols = [col for col in model_fi_df.columns if col.startswith('class_')]
            for col in class_cols:
                if not model_fi_df[col].isna().all():  # Only include if column has non-null values
                    agg_dict[col] = ['mean', 'std']
            
            # Perform aggregation
            fi_agg = model_fi_df.groupby('feature_name').agg(agg_dict)
            
            # Flatten column names
            if isinstance(fi_agg.columns, pd.MultiIndex):
                fi_agg.columns = [
                    f"{col[0]}_mean" if col[1] == 'mean' else f"{col[0]}_std"
                    for col in fi_agg.columns
                ]
            
            feature_importance_summaries[model] = fi_agg.reset_index()

        # Create all_models summary using only feature_importances
        if feature_importance_summaries:
            all_models_dfs = []
            for model, df in feature_importance_summaries.items():
                df_subset = df[['feature_name', 'feature_importances_mean', 'feature_importances_std']].copy()
                df_subset['model'] = model
                all_models_dfs.append(df_subset)
            
            all_models_df = pd.concat(all_models_dfs, ignore_index=True)
            
            # Aggregate across models
            all_models_summary = all_models_df.groupby('feature_name').agg({
                'feature_importances_mean': 'mean',
                'feature_importances_std': lambda x: np.sqrt(np.mean(np.square(x)))  # RMS of stds
            }).reset_index()
            
            # Normalize feature importances
            total_importance = all_models_summary['feature_importances_mean'].sum()
            if total_importance > 0:
                all_models_summary['feature_importances_mean'] /= total_importance
                all_models_summary['feature_importances_std'] /= total_importance
                feature_importance_summaries['all_models'] = all_models_summary
            else:
                logger.warning("Total feature importances sum to zero. Skipping all_models summary.")

        return feature_importance_summaries

    def summarize_all(self) -> Dict[str, Any]:
        """
        Summarize all relevant data.

        Returns:
            Dictionary containing summarized DataFrames.
        """
        crossval_metrics_summary = self.summarize_model_metrics()
        holdout_metrics_summary = self.summarize_holdout_metrics_per_table()
        crossval_sample_metrics_summary = self.summarize_sample_metrics('crossval')
        holdout_sample_metrics_summary = self.summarize_sample_metrics('holdout')
        feature_importances_summary = self.summarize_feature_importances()

        summaries = {
            'crossval_model_metrics_summary': crossval_metrics_summary,
            'holdout_model_metrics_summary': holdout_metrics_summary,
            'crossval_sample_metrics_summary': crossval_sample_metrics_summary,
            'holdout_sample_metrics_summary': holdout_sample_metrics_summary,
            'feature_importances_summary': feature_importances_summary
        }

        return summaries

class FeatureSelectionSummarizer:
    """
    Class to summarize validation metrics from feature selection runs.
    """

    def __init__(self, run_data: List[Dict[str, Any]]):
        self.run_data = run_data
        self.validation_metrics_df = self._construct_validation_metrics_df()
        self.long_validation_metrics_df = self._reshape_validation_metrics_df()

    def _construct_validation_metrics_df(self) -> pd.DataFrame:
        """
        Constructs a DataFrame containing validation metrics from feature selection runs.

        Returns:
            A DataFrame of validation metrics.
        """
        logger = logging.getLogger(__name__)
        validation_records = []
        for run in self.run_data:
            # Assuming 'validation_aggregated_results.json' is the artifact file name
            validation_df = run['artifacts'].get('validation_aggregated_results.json')
            if validation_df is not None and not validation_df.empty:
                validation_df = validation_df.copy()
                validation_df['run_id'] = run['run_id']
                
                # Convert accuracy_std to variance for aggregation
                validation_df['variance_accuracy'] = validation_df['accuracy_std'] ** 2
                # Convert prediction_std to variance for aggregation
                if 'prediction_std' in validation_df.columns:
                    validation_df['variance_prediction'] = validation_df['prediction_std'] ** 2
                
                validation_records.append(validation_df)

        if validation_records:
            validation_metrics_df = pd.concat(validation_records, ignore_index=True)
            logger.debug(f"Validation Metrics DataFrame Columns: {validation_metrics_df.columns.tolist()}")
            logger.debug(f"Validation Metrics DataFrame Head:\n{validation_metrics_df.head()}")
            return validation_metrics_df
        else:
            logger.warning("No validation aggregated results found in any run.")
            return pd.DataFrame()

    def _reshape_validation_metrics_df(self) -> pd.DataFrame:
        """
        Reshape the validation metrics DataFrame to a long format for easier aggregation.

        Returns:
            A long-form DataFrame with separate entries for each metric.
        """
        logger = logging.getLogger(__name__)
        if self.validation_metrics_df.empty:
            logger.warning("Validation metrics DataFrame is empty. Skipping reshaping.")
            return pd.DataFrame()

        # Melt the DataFrame to long format
        melted_df = self.validation_metrics_df.melt(
            id_vars=['run_id', 'sample_id', 'label', 'draw_count'],
            value_vars=[
                'accuracy', 'accuracy_std',
                'avg_prediction', 'prediction_std'
            ],
            var_name='metric_name',
            value_name='value'
        )

        return melted_df

    def summarize_validation_metrics(self) -> pd.DataFrame:
        """
        Summarize validation metrics by computing weighted mean and overall standard deviation.

        Returns:
            A DataFrame containing aggregated validation metrics.
        """
        logger = logging.getLogger(__name__)
        if self.validation_metrics_df.empty:
            logger.warning("Validation metrics DataFrame is empty. Skipping validation metrics summarization.")
            return pd.DataFrame()

        # Calculate total number of draws
        total_draws = self.validation_metrics_df['draw_count'].sum()
        
        # Calculate weighted mean_accuracy using draw_count as weights
        weighted_mean_accuracy = np.average(
            self.validation_metrics_df['accuracy'],
            weights=self.validation_metrics_df['draw_count']
        )

        # Calculate overall variance using the formula:
        # Var_total = (sum(n_i * (Var_i + (mu_i - mu_total)^2))) / N_total
        mean_diff_sq = (self.validation_metrics_df['accuracy'] - weighted_mean_accuracy) ** 2  # Changed from mean_accuracy
        total_variance = (
            (self.validation_metrics_df['draw_count'] * (self.validation_metrics_df['variance_accuracy'] + mean_diff_sq))
            .sum()
        ) / total_draws

        # Calculate overall standard deviation
        overall_std_accuracy = np.sqrt(total_variance)

        # Prepare the summary DataFrame
        summary = pd.DataFrame({
            'weighted_mean_accuracy': [weighted_mean_accuracy],
            'accuracy_std': [overall_std_accuracy],
            'total_draw_count': [total_draws]
        })

        logger.debug(f"Validation Metrics Summary:\n{summary}")

        return summary

    def _construct_long_format_metrics(self) -> pd.DataFrame:
        """
        Constructs a long-format DataFrame for validation metrics.

        Returns:
            A DataFrame in long format with run_id, sample_id, label, metric, and value.
        """
        logger = logging.getLogger(__name__)
        if self.validation_metrics_df.empty:
            logger.warning("Validation metrics DataFrame is empty. Skipping long format construction.")
            return pd.DataFrame()

        melted_df = self.validation_metrics_df.melt(
            id_vars=['run_id', 'sample_id', 'label', 'draw_count'],
            value_vars=['accuracy', 'accuracy_std'],
            var_name='metric',
            value_name='value'
        )

        logger.debug(f"Long Format Validation Metrics Columns: {melted_df.columns.tolist()}")
        logger.debug(f"Long Format Validation Metrics Head:\n{melted_df.head()}")

        return melted_df

    def summarize_sample_validation_metrics(self) -> pd.DataFrame:
        """
        Summarize sample validation metrics by aggregating mean and standard deviation.
        Also includes the total draw count for each sample.

        Returns:
            A DataFrame where each sample_id has aggregated metrics with their respective 
            means, standard deviations, total draw count, and label.
        """
        logger = logging.getLogger(__name__)
        if self.validation_metrics_df.empty:
            logger.warning("Validation metrics DataFrame is empty. Skipping sample validation metrics summarization.")
            return pd.DataFrame()

        # Melt the DataFrame to long format if not already done
        melted_df = self.long_validation_metrics_df

        # Pivot to have metrics as columns
        pivot_df = melted_df.pivot_table(
            index=['sample_id', 'label'],
            columns='metric_name',
            values='value',
            aggfunc='mean'  # Initial aggregation
        ).reset_index()

        # Calculate aggregated variance and total draw count for each sample_id
        agg_df = self.validation_metrics_df.groupby('sample_id').agg({
            'variance_accuracy': 'sum',
            'variance_prediction': 'sum',
            'draw_count': 'sum'
        }).reset_index()
        agg_df.rename(columns={
            'draw_count': 'total_draw_count'
        }, inplace=True)

        # Merge pivoted metrics with aggregated variance and draw count
        summary_df = pd.merge(pivot_df, agg_df, on='sample_id', how='left')

        # Calculate standard deviations from aggregated variances
        summary_df['accuracy_std'] = np.sqrt(summary_df['variance_accuracy'])
        summary_df['prediction_std'] = np.sqrt(summary_df['variance_prediction'])

        # Select and rename relevant columns
        summary_df = summary_df[[
            'sample_id', 
            'label', 
            'accuracy', 
            'accuracy_std',
            'avg_prediction',
            'prediction_std', 
            'total_draw_count'
        ]]

        logger.debug(f"Sample Validation Metrics Summary:\n{summary_df.head()}")
        return summary_df

    def summarize_all(self) -> Dict[str, Any]:
        """
        Perform full summarization of all validation metrics.

        Returns:
            Dictionary containing summarized DataFrames.
        """
        validation_summary = self.summarize_validation_metrics()
        sample_validation_summary = self.summarize_sample_validation_metrics()

        summaries = {
            'feature_selection_validation_summary': validation_summary,
            'feature_selection_sample_validation_metrics': sample_validation_summary,
        }

        return summaries

class OuterBootstrapSummarizer:
    """
    Summarizes results across multiple outer bootstrap iterations.
    """
    def __init__(self, outer_bootstrap_summaries: List[Dict[str, Any]]):
        """
        Initialize summarizer with dynamic artifact mapping.
        
        Args:
            outer_bootstrap_summaries: List of summary runs containing artifacts with metrics:
                - crossval/holdout: f1, accuracy, mcc, roc_auc, pr_auc (with _mean/_std)
                - feature selection: brier, log_loss (with _mean/_std)
        """
        self.bootstrap_summaries = outer_bootstrap_summaries
        self.logger = logging.getLogger(__name__)
        
        # First scan available artifacts across all summaries
        available_artifacts = set()
        for summary in self.bootstrap_summaries:
            # Only add artifacts that aren't None
            available_artifacts.update(
                key for key, value in summary['artifacts'].items() 
                if value is not None
            )
        
        # Map artifact paths to DataFrame collections
        self.model_metrics_dfs = []
        self.holdout_metrics_dfs = []
        self.sample_metrics_dfs = []
        self.holdout_sample_metrics_dfs = []
        self.feature_selection_validation_dfs = []
        self.feature_selection_sample_validation_dfs = []
        
        # Feature importance DataFrames by model
        self.feature_importances_by_model = {}
        
        # Process each summary
        for summary in self.bootstrap_summaries:
            artifacts = summary['artifacts']
            
            # Core metrics - only append if not None
            if 'crossval_model_metrics_summary.csv' in artifacts and artifacts['crossval_model_metrics_summary.csv'] is not None:
                self.model_metrics_dfs.append(artifacts['crossval_model_metrics_summary.csv'])
                
            if 'holdout_model_metrics_summary.csv' in artifacts and artifacts['holdout_model_metrics_summary.csv'] is not None:
                self.holdout_metrics_dfs.append(artifacts['holdout_model_metrics_summary.csv'])
                
            if 'crossval_sample_metrics_summary.csv' in artifacts and artifacts['crossval_sample_metrics_summary.csv'] is not None:
                self.sample_metrics_dfs.append(artifacts['crossval_sample_metrics_summary.csv'])
                
            if 'holdout_sample_metrics_summary.csv' in artifacts and artifacts['holdout_sample_metrics_summary.csv'] is not None:
                self.holdout_sample_metrics_dfs.append(artifacts['holdout_sample_metrics_summary.csv'])
                
            # Feature selection metrics - only append if not None
            if 'feature_selection_validation_summary.csv' in artifacts and artifacts['feature_selection_validation_summary.csv'] is not None:
                self.feature_selection_validation_dfs.append(artifacts['feature_selection_validation_summary.csv'])
                
            if 'feature_selection_sample_validation_metrics_summary.csv' in artifacts and artifacts['feature_selection_sample_validation_metrics_summary.csv'] is not None:
                self.feature_selection_sample_validation_dfs.append(
                    artifacts['feature_selection_sample_validation_metrics_summary.csv']
                )
            
            # Feature importances by model - only process if not None
            for artifact_name in artifacts:
                if (artifact_name.startswith('feature_importances_summary_') and 
                    artifact_name.endswith('.csv') and 
                    artifacts[artifact_name] is not None):
                    model_name = artifact_name.replace('feature_importances_summary_', '').replace('.csv', '')
                    if model_name not in self.feature_importances_by_model:
                        self.feature_importances_by_model[model_name] = []
                    self.feature_importances_by_model[model_name].append(artifacts[artifact_name])
        
        # Remove empty model entries
        self.feature_importances_by_model = {
            model: dfs for model, dfs in self.feature_importances_by_model.items() 
            if dfs  # Only keep models with non-empty DataFrame lists
        }
        
        # Log available artifacts
        self.logger.info(f"Found {len(available_artifacts)} unique artifact types")
        self.logger.info(f"Loaded feature importances for models: {list(self.feature_importances_by_model.keys())}")

    def summarize_model_metrics(self) -> pd.DataFrame:
        """Memory-efficient model metrics aggregation"""
        metrics = ['f1', 'case_accuracy', 'control_accuracy', 'mcc', 'roc_auc', 'pr_auc']
        aggregated_metrics = []
        
        # Process one DataFrame at a time
        for df in self.model_metrics_dfs:
            for model in df['model_name'].unique():
                for dataset in df['dataset'].unique():
                    model_data = df[
                        (df['model_name'] == model) & 
                        (df['dataset'] == dataset)
                    ]
                    
                    record = {'model_name': model, 'dataset': dataset}
                    
                    for metric in metrics:
                        if f'{metric}_mean' in model_data.columns:
                            record[f'{metric}_mean'] = model_data[f'{metric}_mean'].mean()
                            if f'{metric}_std' in model_data.columns:
                                variances = model_data[f'{metric}_std'].pow(2)
                                record[f'{metric}_std'] = np.sqrt(variances.mean())
                    
                    aggregated_metrics.append(record)
            
            # Clear DataFrame reference
            del df
            gc.collect()
        
        return pd.DataFrame(aggregated_metrics)

    def summarize_feature_importances(self) -> Dict[str, pd.DataFrame]:
        """
        Aggregate feature importances across outer bootstraps, separated by model.
        
        Returns:
            Dict mapping model names to their aggregated feature importance DataFrames,
            plus an 'all_models' key with normalized combined importances.
        """
        if not self.feature_importances_by_model:
            self.logger.warning("No feature importance data available")
            return {}
        
        aggregated_importances = {}
        all_models_data = []
        
        # Process each model's feature importances
        for model_name, importance_dfs in self.feature_importances_by_model.items():
            if not importance_dfs:
                continue
                
            # Combine all iterations for this model
            combined_fi = pd.concat(importance_dfs, ignore_index=True)
            
            # Build aggregation dictionary
            agg_dict = {
                'feature_importances_mean': 'mean',
                'feature_importances_std': lambda x: np.sqrt(np.mean(x.pow(2)))  # RMS of stds
            }
            
            # Add class log probability columns if they exist
            class_mean_cols = [col for col in combined_fi.columns if col.endswith('_log_prob_mean')]
            for mean_col in class_mean_cols:
                base_name = mean_col[:-5]  # Remove '_mean' suffix
                std_col = f"{base_name}_std"
                if std_col in combined_fi.columns:
                    agg_dict[mean_col] = 'mean'
                    agg_dict[std_col] = lambda x: np.sqrt(np.mean(x.pow(2)))  # RMS of stds
            
            # Group by feature name and aggregate
            model_fi = combined_fi.groupby('feature_name').agg(agg_dict).reset_index()
            
            # Store model-specific results
            aggregated_importances[model_name] = model_fi
            
            # Add to all-models collection (only using feature importances, not class probs)
            model_fi_copy = model_fi[['feature_name', 'feature_importances_mean', 'feature_importances_std']].copy()
            model_fi_copy['model'] = model_name
            all_models_data.append(model_fi_copy)
        
        # Create all-models summary if we have data
        if all_models_data:
            all_models_df = pd.concat(all_models_data, ignore_index=True)
            
            # Aggregate across models
            all_models_summary = all_models_df.groupby('feature_name').agg({
                'feature_importances_mean': 'mean',
                'feature_importances_std': lambda x: np.sqrt(np.mean(np.square(x)))
            }).reset_index()
            
            # Normalize feature importances
            total_importance = all_models_summary['feature_importances_mean'].sum()
            if total_importance > 0:
                all_models_summary['feature_importances_mean'] /= total_importance
                all_models_summary['feature_importances_std'] /= total_importance
                aggregated_importances['all_models'] = all_models_summary
        
        return aggregated_importances

    def summarize_sample_metrics(self, metrics_type: str = 'sample') -> pd.DataFrame:
        """
        Aggregate sample metrics across outer bootstraps.
        """
        dfs = {
            'sample': self.sample_metrics_dfs,
            'holdout': self.holdout_sample_metrics_dfs,
            'feature_selection': self.feature_selection_sample_validation_dfs
        }[metrics_type]

        if not dfs:
            return pd.DataFrame()

        # Combine all sample metrics DataFrames with their run_ids
        dfs_with_run_ids = []
        for summary, df in zip(self.bootstrap_summaries, dfs):
            df_copy = df.copy()
            df_copy['run_id'] = summary['run_id']
            dfs_with_run_ids.append(df_copy)

        combined_metrics = pd.concat(dfs_with_run_ids, ignore_index=True)
        
        # Identify column types
        count_metrics = ['total_draw_count']
        std_suffix = '_std'
        preserved_cols = ['sample_id', 'label']
        if metrics_type == 'holdout' and 'table' in combined_metrics.columns:
            preserved_cols.append('table')
        
        # Get base metric names (without _std or model suffixes)
        base_metrics = {
            col.split('_')[0] for col in combined_metrics.columns 
            if col not in preserved_cols + ['run_id'] and col not in count_metrics
        }
        
        # Group by sample_id and aggregate
        aggregated_metrics = []
        ordered_columns = preserved_cols.copy()
        first_sample = combined_metrics.iloc[0]
        
        # Add total_draw_count if it exists
        if 'total_draw_count' in first_sample:
            ordered_columns.append('total_draw_count')
        
        # Process each base metric to build ordered column list
        for base_metric in base_metrics:
            metric_cols = [col for col in combined_metrics.columns if col.startswith(f"{base_metric}_") or col == base_metric]
            
            # Case 1: Direct mean/std pair
            if base_metric in combined_metrics.columns:
                ordered_columns.append(base_metric)
                std_col = f"{base_metric}_std"
                if std_col in combined_metrics.columns:
                    ordered_columns.append(std_col)
                continue
            
            # Case 2: Model-specific metrics
            model_cols = [col for col in metric_cols if not col.endswith(std_suffix)]
            if model_cols:
                # Add each model-specific column and its std
                for col in model_cols:
                    ordered_columns.append(col)
                    std_col = f"{col}_std"
                    if std_col in combined_metrics.columns or len(combined_metrics['run_id'].unique()) > 1:
                        ordered_columns.append(std_col)
                
                # Add overall metric if multiple models
                if len(model_cols) > 1:
                    ordered_columns.append(base_metric)
                    ordered_columns.append(f"{base_metric}_std")
        
        # Now process each sample using the ordered columns
        for sample_id in combined_metrics['sample_id'].unique():
            sample_data = combined_metrics[combined_metrics['sample_id'] == sample_id]
            record = {}
            
            # Process columns in order
            for col in ordered_columns:
                if col in preserved_cols:
                    record[col] = sample_data[col].iloc[0]
                elif col == 'total_draw_count':
                    record[col] = sample_data[col].astype(int).sum()
                elif col.endswith(std_suffix):
                    base_col = col[:-4]  # Remove _std suffix
                    if col in sample_data.columns:
                        # Use RMS for existing std columns
                        record[col] = np.sqrt(sample_data[col].pow(2).mean())
                    else:
                        # For model-specific metrics, find the relevant columns
                        base_cols = [c for c in sample_data.columns if c.startswith(f"{base_col}_") and not c.endswith(std_suffix)]
                        if base_cols:
                            values = sample_data[base_cols].mean(axis=1)
                            if len(values) > 1:
                                record[col] = values.std(ddof=1)
                            else:
                                record[col] = np.nan
                        elif base_col in sample_data.columns and len(sample_data[base_col]) > 1:
                            # Direct metric case
                            record[col] = sample_data[base_col].std(ddof=1)
                        else:
                            record[col] = np.nan
                else:
                    # Handle means for both direct and model-specific metrics
                    if col in sample_data.columns:
                        record[col] = sample_data[col].mean()
                    else:
                        # For overall metrics with multiple models
                        model_cols = [c for c in sample_data.columns if c.startswith(f"{col}_") and not c.endswith(std_suffix)]
                        if model_cols:
                            record[col] = sample_data[model_cols].mean(axis=1).mean()
            
            aggregated_metrics.append(record)
        
        return pd.DataFrame(aggregated_metrics, columns=ordered_columns)

    def summarize_all(self) -> Dict[str, Any]:
        """
        Generate all summaries across outer bootstrap iterations.
        
        Returns:
            Dictionary containing all available summary DataFrames.
        """
        summaries = {
            'outer_bootstrap_model_metrics': self.summarize_model_metrics(),
            'outer_bootstrap_feature_importances': self.summarize_feature_importances(),
            'outer_bootstrap_sample_metrics': self.summarize_sample_metrics('sample'),
            'outer_bootstrap_holdout_sample_metrics': self.summarize_sample_metrics('holdout'),
            'outer_bootstrap_feature_selection_sample_metrics': self.summarize_sample_metrics('feature_selection')
        }
        
        # Filter out empty DataFrames and handle feature importances specially
        filtered_summaries = {}
        for key, value in summaries.items():
            if isinstance(value, dict):  # Feature importances case
                if value:  # Only include if there are any model results
                    filtered_summaries[key] = value
            elif not value.empty:  # Regular DataFrame case
                filtered_summaries[key] = value
        
        return filtered_summaries
