import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import os

os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '1200'

import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import logging
from config import TrackingConfig

@lru_cache(maxsize=8096)
def query_ensembl(chrom: str, pos: int) -> Optional[List[Dict[str, Any]]]:
    """
    Query the Ensembl REST API for genes overlapping a specific genomic position.

    Args:
        chrom (str): Chromosome (e.g., '1', 'X').
        pos (int): Genomic position.

    Returns:
        Optional[List[Dict[str, Any]]]: List of gene data or None if the query fails.
    """
    server = "https://rest.ensembl.org"
    ext = f"/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene"
    
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get(server + ext, headers=headers, timeout=10)
        if response.ok:
            return response.json()
        else:
            logging.getLogger(__name__).warning(f"Ensembl query failed for {chrom}:{pos} with status code {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.getLogger(__name__).error(f"Ensembl query exception for {chrom}:{pos}: {e}")
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

def save_summary(summaries: Dict[str, Any], tracking_config: TrackingConfig):
    """
    Save the summarized DataFrames to MLflow as a new run.
    Handles multiple summary DataFrames, including dictionaries of DataFrames.
    Adds a 'gene_symbol' column if 'feature_name' is present.

    Args:
        summaries: Dictionary containing summarized DataFrames.
        tracking_config: Configuration for MLflow tracking.
    """
    mlflow.set_tracking_uri(tracking_config.tracking_uri)
    mlflow.set_experiment(tracking_config.experiment_name)

    with mlflow.start_run(run_name='summary'):
        for name, df in summaries.items():
            if isinstance(df, dict):
                # Handle dictionaries of DataFrames (e.g., feature_importances_summary)
                for sub_name, sub_df in df.items():
                    if sub_df.empty:
                        mlflow.log_text(f"{name}_{sub_name} summary is empty.", artifact_file=f"summaries/{name}_{sub_name}_empty.txt")
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
                    mlflow.log_artifact(temp_path, f"summaries/{name}/{sub_name}")
                    os.remove(temp_path)  # Clean up temporary file

                    # Log some basic metrics about the summary
                    mlflow.log_metric(f"{name}_{sub_name}_row_count", len(sub_df))
                    mlflow.log_metric(f"{name}_{sub_name}_column_count", len(sub_df.columns))
            else:
                if df.empty:
                    mlflow.log_text(f"{name} summary is empty.", artifact_file=f"summaries/{name}_empty.txt")
                    continue

                # Add 'gene_symbol' column if 'feature_name' exists using threading
                if 'feature_name' in df.columns:
                    logging.getLogger(__name__).info(f"Adding 'gene_symbol' to {name}_summary using threading")
                    feature_names = df['feature_name'].tolist()
                    gene_symbols = fetch_gene_symbols_threaded(feature_names)
                    df['gene_symbol'] = gene_symbols

                # Save DataFrame as a CSV artifact
                temp_path = f"{name}_summary.csv" if 'summary' not in name else f"{name}.csv"
                df.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path, f"summaries/{name}")
                os.remove(temp_path)  # Clean up temporary file

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
        fi_records = []
        for run in self.run_data:
            fi_df = run['artifacts'].get('feature_importances.json')
            if fi_df is not None and not fi_df.empty:
                fi_df = fi_df.copy()
                fi_df['run_id'] = run['run_id']
                
                # Check if this is a multinomial naive bayes model with logprobs
                if 'class_0' in fi_df.columns and 'class_1' in fi_df.columns:
                    # Convert logprobs to probabilities
                    fi_df['class_0'] = np.exp(fi_df['class_0'])
                    fi_df['class_1'] = np.exp(fi_df['class_1'])
                    
                    # Compute mean probability across classes
                    fi_df['feature_importances'] = fi_df[['class_0', 'class_1']].mean(axis=1)
                    
                    # Drop the original class columns
                    fi_df = fi_df.drop(columns=['class_0', 'class_1'])
                
                fi_records.append(fi_df)
        
        if fi_records:
            feature_importances_df = pd.concat(fi_records, ignore_index=True)
            logging.getLogger(__name__).debug(f"Feature Importances DataFrame Columns: {feature_importances_df.columns.tolist()}")
            logging.getLogger(__name__).debug(f"Feature Importances DataFrame Head:\n{feature_importances_df.head()}")
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

        Args:
            sample_type: Either 'crossval' or 'holdout'

        Returns:
            A DataFrame containing sample metrics per model, indexed by sample_id.
        """
        logger = logging.getLogger(__name__)
        sample_metrics_df = self.crossval_sample_metrics_df if sample_type == 'crossval' else self.holdout_sample_metrics_df

        if sample_metrics_df.empty:
            logger.warning(f"{sample_type.capitalize()} sample metrics DataFrame is empty. Skipping summarization.")
            return pd.DataFrame()

        # Identify metric columns (excluding run_id, dataset, sample_id, label, y_pred)
        exclude_cols = ['run_id', 'dataset', 'sample_id', 'label', 'y_pred']
        metric_cols = [col for col in sample_metrics_df.columns if col not in exclude_cols]

        if not metric_cols:
            logger.warning(f"No metric columns found in {sample_type} sample_metrics_df. Skipping summarization.")
            return pd.DataFrame()

        id_cols = ['run_id', 'dataset', 'sample_id', 'label']
        if sample_type == 'holdout':
            id_cols.append('table')

        # Melt the DataFrame to long format
        melted_df = sample_metrics_df.melt(
            id_vars=id_cols,
            value_vars=metric_cols,
            var_name='metric_name',
            value_name='value'
        )

        # Define the sample metrics to retain
        sample_metrics_to_retain = ['brier', 'log_loss', 'accuracy']  # Adjust based on actual metrics

        # Filter for relevant sample metrics
        filtered_sample_metrics = melted_df[
            melted_df['metric_name'].isin(sample_metrics_to_retain)
        ].copy()

        if filtered_sample_metrics.empty:
            logger.warning(f"No relevant {sample_type} sample metrics found after filtering. Skipping summarization.")
            return pd.DataFrame()

        # Merge with metrics_df to get model_name
        merged_df = pd.merge(
            filtered_sample_metrics,
            self.metrics_df[['run_id', 'model_name']],
            on='run_id',
            how='left'
        )

        # Define preserved index columns
        preserved_index_cols = ['sample_id', 'label']
        if sample_type == 'holdout':
            preserved_index_cols.append('table')

        # Pivot the DataFrame to have sample_id as index and metrics per model as separate columns
        pivot_df = merged_df.pivot_table(
            index=preserved_index_cols,
            columns=['model_name', 'metric_name'],
            values='value',
            aggfunc='mean'  # Use appropriate aggregation function if needed
        )

        # Flatten the MultiIndex columns for pivoted columns only
        pivoted_columns = [f"{metric}_{model}" for model, metric in pivot_df.columns]
        pivot_df.columns = pivoted_columns

        # Reset index to convert preserved index columns back to DataFrame columns
        pivot_df = pivot_df.reset_index()

        return pivot_df

    def summarize_feature_importances(self) -> Dict[str, pd.DataFrame]:
        """
        Summarize feature importances aggregated by model and feature.
        Handles both single-feature_importances and multi-class feature importances.
        For models like MultinomialNB with logprob feature importances, 
        it aggregates across classes to produce a single feature_importances column.
        Aggregates standard deviations appropriately when combining across models.
        Includes an 'all_models' summary with normalized feature importances.
    
        Returns:
            A dictionary where keys are model names and values are corresponding
            feature importance DataFrames with 'feature_importances_mean' and 
            'feature_importances_std' columns. Additionally, includes an 'all_models'
            key containing a normalized combined summary of feature importances across all models.
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
    
        # Identify unique models
        models = merged_fi_df['model_name'].unique()

        for model in models:
            model_fi_df = merged_fi_df[merged_fi_df['model_name'] == model]

            if model_fi_df.empty:
                logger.warning(f"No feature importances found for model {model}. Skipping.")
                continue

            aggregation = pd.DataFrame({'feature_name': model_fi_df['feature_name'].unique()})

            if 'feature_importances' in model_fi_df.columns:
                # Single-class feature importances
                fi_agg = model_fi_df.groupby('feature_name')['feature_importances'].agg(['mean', 'std']).reset_index()
                fi_agg.rename(columns={
                    'mean': 'feature_importances_mean',
                    'std': 'feature_importances_std'
                }, inplace=True)
                aggregation = aggregation.merge(fi_agg, on='feature_name', how='left')

            class_columns = [col for col in model_fi_df.columns if col.startswith('class_')]
            if class_columns:
                # Multi-class feature importances (e.g., MultinomialNB with logprob)
                # Assuming feature_log_prob_ columns are named 'class_0', 'class_1', etc.
                if 'feature_importances' in model_fi_df.columns:
                    logger.info(f"Both 'feature_importances' and class-specific columns found for model {model}. Aggregating both.")
                else:
                    # Compute log odds ratio
                    log_odds_df = model_fi_df[class_columns[0]] - model_fi_df[class_columns[1]]

                    aggregation['feature_importances_mean'] = log_odds_df.mean(axis=1)
                    aggregation['feature_importances_std'] = log_odds_df.std(axis=1)

                # Aggregate class-specific importances
                for col in class_columns:
                    class_agg = model_fi_df.groupby('feature_name')[col].agg(['mean', 'std']).reset_index()
                    class_agg.rename(columns={
                        'mean': f'{col}_mean',
                        'std': f'{col}_std'
                    }, inplace=True)
                    aggregation = aggregation.merge(class_agg, on='feature_name', how='left')

            if aggregation[['feature_importances_mean', 'feature_importances_std']].isnull().all().all():
                logger.warning(f"Unexpected feature importances structure for model {model}. Skipping summarization.")
                continue

            feature_importance_summaries[model] = aggregation

        # Combine all model summaries into a single DataFrame for overall analysis
        combined_feature_importances = []
        for model, df in feature_importance_summaries.items():
            df_copy = df.copy()
            df_copy['model_name'] = model
            combined_feature_importances.append(df_copy)
    
        if combined_feature_importances:
            combined_df = pd.concat(combined_feature_importances, ignore_index=True)
    
            # Aggregate feature importances across all models
            # Compute the mean of feature_importances_mean and aggregate variance
            all_models_aggregation = combined_df.groupby('feature_name').agg({
                'feature_importances_mean': 'mean',  # Mean of mean importances
                'feature_importances_std': lambda x: np.sqrt(np.mean(np.square(x)))  # Aggregated std using RMS
            }).reset_index()
    
            # Rename columns appropriately
            all_models_aggregation.rename(columns={
                'feature_importances_mean': 'feature_importances_mean',
                'feature_importances_std': 'feature_importances_std'
            }, inplace=True)
    
            # Normalize feature_importances_mean so that they sum to 1
            total_importance = all_models_aggregation['feature_importances_mean'].sum()
            if total_importance > 0:
                all_models_aggregation['feature_importances_mean'] /= total_importance
                # Normalize the standard deviation proportionally
                all_models_aggregation['feature_importances_std'] /= total_importance
                # Optional: Validate normalization
                assert np.isclose(all_models_aggregation['feature_importances_mean'].sum(), 1.0), "Normalization error: Sum of feature importances is not 1."
            else:
                logger.warning("Total feature importances across all models sum to zero. Skipping normalization for 'all_models' summary.")
    
            feature_importance_summaries['all_models'] = all_models_aggregation[['feature_name', 'feature_importances_mean', 'feature_importances_std']]
    
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
                validation_records.append(validation_df)

        if validation_records:
            validation_metrics_df = pd.concat(validation_records, ignore_index=True)
            # Convert std_accuracy to variance for aggregation purposes
            validation_metrics_df['variance_accuracy'] = validation_metrics_df['std_accuracy'] ** 2
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
            value_vars=['mean_accuracy', 'std_accuracy'],
            var_name='metric_name',
            value_name='value'
        )

        # Log the structure of the melted DataFrame
        logger.debug(f"Long Validation Metrics DataFrame Columns: {melted_df.columns.tolist()}")
        logger.debug(f"Long Validation Metrics DataFrame Head:\n{melted_df.head()}")

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
            self.validation_metrics_df['mean_accuracy'],
            weights=self.validation_metrics_df['draw_count']
        )

        # Calculate overall variance using the formula:
        # Var_total = (sum(n_i * (Var_i + (mu_i - mu_total)^2))) / N_total
        mean_diff_sq = (self.validation_metrics_df['mean_accuracy'] - weighted_mean_accuracy) ** 2
        total_variance = (
            (self.validation_metrics_df['draw_count'] * (self.validation_metrics_df['variance_accuracy'] + mean_diff_sq))
            .sum()
        ) / total_draws

        # Calculate overall standard deviation
        overall_std_accuracy = np.sqrt(total_variance)

        # Prepare the summary DataFrame
        summary = pd.DataFrame({
            'weighted_mean_accuracy': [weighted_mean_accuracy],
            'overall_std_accuracy': [overall_std_accuracy],
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
            value_vars=['mean_accuracy', 'std_accuracy'],
            var_name='metric',
            value_name='value'
        )

        logger.debug(f"Long Format Validation Metrics Columns: {melted_df.columns.tolist()}")
        logger.debug(f"Long Format Validation Metrics Head:\n{melted_df.head()}")

        return melted_df

    def summarize_sample_validation_metrics(self) -> pd.DataFrame:
        """
        Summarize sample validation metrics by aggregating mean and standard deviation.

        Returns:
            A DataFrame where each sample_id has aggregated metrics with their respective means and standard deviations.
        """
        logger = logging.getLogger(__name__)
        if self.validation_metrics_df.empty:
            logger.warning("Validation metrics DataFrame is empty. Skipping sample validation metrics summarization.")
            return pd.DataFrame()

        # Define the metrics to aggregate
        metrics_to_aggregate = ['mean_accuracy', 'std_accuracy']

        # Melt the DataFrame to long format if not already done
        melted_df = self.long_validation_metrics_df

        # Pivot to have metrics as columns
        pivot_df = melted_df.pivot_table(
            index=['sample_id'],
            columns='metric_name',
            values='value',
            aggfunc='mean'  # Initial aggregation; adjust if necessary
        ).reset_index()

        # Calculate aggregated variance for each metric
        variance_df = self.validation_metrics_df.groupby('sample_id')['variance_accuracy'].sum().reset_index()
        variance_df.rename(columns={'variance_accuracy': 'aggregated_variance_accuracy'}, inplace=True)

        # Merge pivoted metrics with aggregated variance
        summary_df = pd.merge(pivot_df, variance_df, on='sample_id', how='left')

        # Calculate standard deviation from aggregated variance
        summary_df['aggregated_std_accuracy'] = np.sqrt(summary_df['aggregated_variance_accuracy'])

        # Select and rename relevant columns
        summary_df = summary_df.rename(columns={
            'mean_accuracy': 'accuracy_mean',
            'aggregated_std_accuracy': 'accuracy_std'
        })

        # Retain only sample_id, accuracy_mean, and accuracy_std
        summary_df = summary_df[['sample_id', 'accuracy_mean', 'accuracy_std']]

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

