import logging
from typing import Dict, Optional, List
from functools import partial
import re

import hail as hl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import pyro
from joblib import Parallel, delayed
from mlflow.models import infer_signature
from skopt import BayesSearchCV
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from skopt.plots import plot_convergence

from config import SamplingConfig, TrackingConfig
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from eval.feature_selection import BayesianFeatureSelector, FeatureSelectionResult

def get_feature_importances(model, feature_labels):
    """Get feature importances from fitted model."""
    if hasattr(model, 'beta_loc_'):
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df['feature_importances'] = model.beta_loc_
        model_coefficient_df['feature_name'] = feature_labels
    elif hasattr(model, 'coef_'):
        model_coefficient_df = pd.DataFrame()
        if len(model.coef_.shape) > 1:
            model_coefficient_df['feature_importances'] = model.coef_[0]
        else:
            model_coefficient_df['feature_importances'] = model.coef_.flatten()
        model_coefficient_df['feature_name'] = feature_labels
    elif hasattr(model, 'feature_importances_'):
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df['feature_name'] = feature_labels
        model_coefficient_df['feature_importances'] = model.feature_importances_
    else:
        model_coefficient_df = None
    return model_coefficient_df

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate sample-wise metrics
    sample_metrics = {
        "brier": (y_true - y_pred) ** 2,
        "log_loss": -(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)),
        "accuracy": y_true == y_pred_binary
    }
    
    aggregate_metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "case_accuracy": accuracy_score(y_true[y_true != 0], y_pred_binary[y_true != 0]),
        "control_accuracy": accuracy_score(y_true[y_true == 0], y_pred_binary[y_true == 0])
    }
    
    # Combine sample-wise and aggregate metrics
    metrics = {
        "sample_wise": sample_metrics,
        "aggregate": aggregate_metrics
    }
    
    return metrics

def sanitize_mlflow_name(name: str) -> str:
    """
    Sanitize the name to comply with MLflow naming conventions.
    
    Args:
    name (str): The original name.
    
    Returns:
    str: The sanitized name.
    """
    # Replace any character that's not alphanumeric, underscore, dash, period, space, or slash
    # with an underscore
    name = re.sub(r'[^a-zA-Z0-9_\-. /]', '_', name)
    
    # Remove leading/trailing whitespace
    name = name.strip()
    
    return name

def log_mlflow_metrics(metrics, best_params, model, X_test, y_test, y_pred_test, y_train, y_pred_train, df_y_pred_train, df_y_pred_test, model_coefficient_df, trackingConfig):
    """Log metrics, artifacts, and visualizations to MLflow."""
    signature = infer_signature(X_test, y_pred_test)
    
    mlflow.log_params(model.get_params())
    
    # Log train and test aggregate metrics
    for dataset in ['train', 'test']:
        for metric_name, metric_value in metrics['aggregate'][dataset].items():
            mlflow.log_metric(f"{dataset}/{metric_name}", metric_value)
    
    # Combine predictions and sample-wise metrics for train and test
    train_sample_metrics = df_y_pred_train.copy()
    test_sample_metrics = df_y_pred_test.copy()
    
    for metric_name, metric_values in metrics['sample_wise']['train'].items():
        train_sample_metrics[f'{metric_name}'] = metric_values.values
    
    for metric_name, metric_values in metrics['sample_wise']['test'].items():
        test_sample_metrics[f'{metric_name}'] = metric_values.values
    
    # Log combined sample data as tables
    mlflow.log_table(data=train_sample_metrics, artifact_file=sanitize_mlflow_name("train_sample_metrics.json"))
    mlflow.log_table(data=test_sample_metrics, artifact_file=sanitize_mlflow_name("test_sample_metrics.json"))
    
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    if model_coefficient_df is not None:
        mlflow.log_table(data=model_coefficient_df, artifact_file=sanitize_mlflow_name("feature_importances.json"))

def evaluate_model(model, search_spaces, X_train, y_train, X_test, y_test, sample_processor: SampleProcessor, trackingConfig: TrackingConfig, n_iter=15, is_worker=True, num_variants: int = None, total_variants: int = None, confidence_level: float = None, iteration: int = 0):
    """Perform Bayesian hyperparameter optimization for a model and evaluate on holdout samples."""
    logger = logging.getLogger(__name__)
    
    # Retrieve model parameters
    params = model.get_params()
    
    # Check if 'unique_id' is a valid parameter
    if 'unique_id' in params:
        params['unique_id'] = f"{iteration}"
    
    # Initialize a fresh model instance with updated parameters
    process_safe_model = model.__class__(**params)
    
    try:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', process_safe_model)
        ])
        
        search_spaces = {f'classifier__{key}': value for key, value in search_spaces.items()}
        
        search = BayesSearchCV(
            pipeline,
            search_spaces,
            n_iter=n_iter,
            cv=5,
            n_jobs=-1,
            n_points=10,
            scoring='roc_auc'
        )
        
        search.fit(X_train, y_train)
                
        # Use the fitted pipeline for predictions
        y_pred_train = search.predict_proba(X_train)[:, 1]
        y_pred_test = search.predict_proba(X_test)[:, 1]

        df_y_pred_train = y_train.to_frame(name='label').assign(y_pred=y_pred_train).reset_index()
        df_y_pred_test = y_test.to_frame(name='label').assign(y_pred=y_pred_test).reset_index()
        
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        metrics = {
            'aggregate': {
                'train': train_metrics['aggregate'],
                'test': test_metrics['aggregate']
            },
            'sample_wise': {
                'train': train_metrics['sample_wise'],
                'test': test_metrics['sample_wise']
            }
        }
        
        best_params = {key.replace('classifier__', ''): value for key, value in search.best_params_.items()}

        # Get feature names from X_train
        feature_labels = X_train.columns.tolist()

        model_coefficient_df = get_feature_importances(search.best_estimator_['classifier'], feature_labels)

        # Set up MLflow tracking
        mlflow.set_tracking_uri(trackingConfig.tracking_uri)
        mlflow.set_experiment(trackingConfig.experiment_name)

        # Start a run for this model evaluation
        with mlflow.start_run(run_name=f"{model.__class__.__name__}", nested=True) as run:
            mlflow.set_tag("model", model.__class__.__name__)

            plot_and_log_convergence('ROC AUC', search, y_train, trackingConfig, num_variants, total_variants, confidence_level)
            
            log_mlflow_metrics(
                metrics,
                best_params,
                search.best_estimator_['classifier'],
                X_test,
                y_test,
                y_pred_test,
                y_train,
                y_pred_train,
                df_y_pred_train,
                df_y_pred_test,
                model_coefficient_df,
                trackingConfig
            )

            # Pass the feature selection metrics to the visualization function
            create_and_log_visualizations(
                model.__class__.__name__,
                y_test,
                y_pred_test,
                trackingConfig,
                set_type="test",
                table_name="crossval",  # Default table name for cross-validation
                num_variants=num_variants,
                total_variants=total_variants,
                confidence_level=confidence_level
            )
            create_and_log_visualizations(
                model.__class__.__name__,
                y_train,
                y_pred_train,
                trackingConfig,
                set_type="train",
                table_name="crossval",  # Default table name for cross-validation
                num_variants=num_variants,
                total_variants=total_variants,
                confidence_level=confidence_level
            )

            # --- Holdout Evaluation ---
            logger.info("Starting holdout evaluation.")

            holdout_tables = sample_processor.holdout_data

            for table_name, table_info in holdout_tables.items():
                logger.info(f"Evaluating holdout samples for table: {table_name}")

                # Get holdout sample IDs for the current table from the id_mapping
                holdout_sample_ids = [
                    genotype_id for sample_id, genotype_id in sample_processor.id_mapping.items()
                    if sample_id in table_info['data'].index
                ]

                if not holdout_sample_ids:
                    logger.warning(f"No holdout samples found for table: {table_name}. Skipping.")
                    continue

                # Fetch genotypes for these samples
                _, X_holdout, _, y_holdout = prepare_data(
                    parquet_path=f"/tmp/{trackingConfig.experiment_name}.parquet",
                    sample_processor=sample_processor,
                    train_samples=[],
                    test_samples=holdout_sample_ids,
                    selected_features=feature_labels,
                    dataset='holdout'
                )

                if X_holdout.empty:
                    logger.warning(f"No data available after processing holdout samples for table: {table_name}. Skipping.")
                    continue

                # Apply the same preprocessing pipeline to holdout data
                X_holdout_scaled = search.best_estimator_.named_steps['scaler'].transform(X_holdout)

                # Make predictions using the classifier directly (as scaling is already done)
                y_pred_holdout = search.best_estimator_.named_steps['classifier'].predict_proba(X_holdout_scaled)[:, 1]

                # Calculate metrics
                holdout_metrics = calculate_metrics(y_holdout, y_pred_holdout)

                # Log aggregate holdout metrics
                for metric_name, metric_value in holdout_metrics['aggregate'].items():
                    if metric_value is not None:
                        sanitized_name = sanitize_mlflow_name(f"holdout/{table_name}/{metric_name}")
                        mlflow.log_metric(sanitized_name, metric_value)

                # Combine predictions and sample-wise metrics for holdout
                holdout_sample_metrics = y_holdout.to_frame(name='label').assign(y_pred=y_pred_holdout).reset_index()
                
                for metric_name, metric_values in holdout_metrics['sample_wise'].items():
                    holdout_sample_metrics[f'{metric_name}'] = metric_values.values

                # Log holdout sample metrics as a table
                mlflow.log_table(
                    data=holdout_sample_metrics, 
                    artifact_file=sanitize_mlflow_name(f"holdout/{table_name}_sample_metrics.json")
                )

                # Log holdout visualizations with actual table_name and feature selection metrics
                create_and_log_visualizations(
                    model.__class__.__name__,
                    y_holdout,
                    y_pred_holdout,
                    trackingConfig,
                    set_type="holdout",
                    table_name=table_name,  # Actual holdout table name
                    num_variants=num_variants,
                    total_variants=total_variants,
                    confidence_level=confidence_level
                )

            logger.info("Completed holdout evaluation.")

        return metrics, y_pred_test, y_pred_train, best_params, run.info.run_id
    except Exception as e:
        logger.exception("An error occurred during model evaluation.")
        raise e

def prepare_data(parquet_path: str, sample_processor, train_samples, test_samples, selected_features: Optional[List[str]] = None, dataset: str = 'crossval'):
    """
    Prepare data for a single bootstrap iteration by reading from Parquet.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading data from Parquet at {parquet_path}")

    # Load only the necessary rows (samples) and columns (features) to reduce memory usage
    all_sample_ids = train_samples + test_samples
    if selected_features is not None:
        columns = ['sample_id'] + selected_features
    else:
        columns = None  # Read all columns
    sampled_data = pd.read_parquet(parquet_path, filters=[('sample_id', 'in', all_sample_ids)], columns=columns)
    
    # Set sample_id as index if it's not already
    if 'sample_id' in sampled_data.columns:
        sampled_data.set_index('sample_id', inplace=True)
    
    # Split into train and test sets
    X_train = sampled_data.loc[train_samples]
    X_test = sampled_data.loc[test_samples]

    # Get labels
    y_train = sample_processor.get_labels(train_samples, dataset)
    y_test = sample_processor.get_labels(test_samples, dataset)

    # Ensure that the indices align
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    return X_train, X_test, y_train, y_test

def process_iteration(
    iteration: int,
    parquet_path: str,
    sample_processor: SampleProcessor,
    genotype_processor: GenotypeProcessor,
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    selected_features: pd.Index,
    num_variants: int,
    total_variants: int,
    confidence_level: float,
    random_state: int,
) -> Dict:
    """
    Process a single bootstrap iteration using the selected features and feature selection metrics.
    """
    logger = logging.getLogger(f"bootstrap_iteration_{iteration}")
    logger.info(f"Bootstrapping iteration {iteration + 1}/{samplingConfig.bootstrap_iterations}")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    mlflow.set_experiment(trackingConfig.experiment_name)
    
    train_test_sample_ids = sample_processor.draw_train_test_split(
        test_size=samplingConfig.test_size,
        random_state=random_state + iteration
    )

    train_samples = list(train_test_sample_ids['train']['samples'].values())
    test_samples = list(train_test_sample_ids['test']['samples'].values())
    
    # Prepare data by reading from Parquet with selected features
    X_train, X_test, y_train, y_test = prepare_data(
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        train_samples=train_samples,
        test_samples=test_samples,
        selected_features=selected_features.tolist()
    )

    with mlflow.start_run(run_name=f"Bootstrap_{iteration}", nested=True) as parent_run:
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("random_state", random_state + iteration)
        mlflow.log_param("num_selected_features", len(selected_features))
        mlflow.log_param("total_features", total_variants)
        mlflow.log_param("confidence_level", confidence_level)

        iteration_results = {
            "iteration": iteration,
            "models": []
        }

        # Iterate over each model in the stack and evaluate
        for model, search_spaces in stack.items():
            model_name = model.__class__.__name__
            logger.info(f"Evaluating model: {model_name}")

            metrics, y_pred_test, y_pred_train, best_params, child_run_id = evaluate_model(
                model=model,
                search_spaces=search_spaces,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                trackingConfig=trackingConfig,
                sample_processor=sample_processor,
                n_iter=15,
                is_worker=True,
                num_variants=num_variants,
                total_variants=total_variants,
                confidence_level=confidence_level,
                iteration=iteration
            )

            mlflow.log_metric(f"{model_name}_test_f1", metrics['aggregate']['test']['f1'])
            mlflow.log_param(f"{model_name}_run_id", child_run_id)

            iteration_results["models"].append({
                "model_name": model_name,
                "test_f1": metrics['aggregate']['test']['f1'],
                "best_params": best_params,
                "run_id": child_run_id
            })

        pyro.clear_param_store()

    return iteration_results

def parallel_feature_selection(
    parquet_path: str,
    sample_processor: SampleProcessor,
    samplingConfig: SamplingConfig,
    num_iterations: int = 10,
    total_variants: int = None,
    random_state: int = 42,
    trackingConfig: TrackingConfig = None
) -> FeatureSelectionResult:
    """
    Run feature selection in parallel over multiple splits and collect overlapping features.

    Returns:
        FeatureSelectionResult: Encapsulates selected features and related metrics.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting parallel feature selection.")

    # Perform feature selection iterations
    selected_features_list = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(feature_selection_iteration)(
            i, 
            parquet_path, 
            sample_processor, 
            samplingConfig, 
            random_state + i,  # Unique seed
            trackingConfig
        )
        for i in range(num_iterations)
    )

    logger.info("Completed parallel feature selection.")

    # Combine selected features from all iterations
    feature_counts = pd.Series(np.concatenate(selected_features_list)).value_counts()

    # Determine overlapping features using a statistical threshold
    threshold = np.floor(num_iterations * 0.2)  # e.g., features selected in at least 20% of iterations
    overlapping_features = feature_counts[feature_counts >= threshold].index

    num_variants = len(overlapping_features)
    confidence_level = samplingConfig.feature_confidence_level  # Assuming this is defined in your config

    logger.info(f"{num_variants} of {total_variants} variants selected at {confidence_level*100:.2f}% credible interval.")

    return FeatureSelectionResult(
        selected_features=overlapping_features,
        num_variants=num_variants,
        total_variants=total_variants,
        confidence_level=samplingConfig.feature_confidence_level
    )

def feature_selection_iteration(
    iteration: int,
    parquet_path: str,
    sample_processor: SampleProcessor,
    samplingConfig: SamplingConfig,
    random_state: int,
    trackingConfig: TrackingConfig = None
) -> pd.Index:
    """
    Perform feature selection on a random split of the data.
    """
    logger = logging.getLogger(f"feature_selection_iteration_{iteration}")
    logger.info(f"Feature selection iteration {iteration + 1}")

    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    mlflow.set_experiment(trackingConfig.experiment_name)

    # Start an MLflow run for this feature selection iteration
    with mlflow.start_run(run_name=f"Feature_Selection_{iteration}", nested=True) as run:
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("random_state", random_state + iteration)
        mlflow.set_tag('feature_selection_status', 'in_progress')

        try:
            train_test_sample_ids = sample_processor.draw_train_test_split(
                test_size=samplingConfig.test_size,
                random_state=random_state + iteration
            )
    
            train_samples = list(train_test_sample_ids['train']['samples'].values())
            test_samples = list(train_test_sample_ids['test']['samples'].values())
    
            # Prepare data by reading from Parquet
            X_train, X_test, y_train, y_test = prepare_data(
                parquet_path=parquet_path,
                sample_processor=sample_processor,
                train_samples=train_samples,
                test_samples=test_samples
            )
    
            # Feature Selection Step
            logger.info("Performing feature selection using Bayesian Feature Selector.")
            mlflow.log_param("feature_selector", "BayesianFeatureSelector")
            max_attempts = 5
            attempt = 0
            selected_features = []

            while len(selected_features) == 0 and attempt < max_attempts:
                feature_selector = BayesianFeatureSelector(
                    num_iterations=10,
                    lr=1e-2,
                    confidence_level=samplingConfig.feature_confidence_level,
                    num_samples=2000,
                    patience=800,
                    validation_split=0.2,
                    batch_size=512,
                    verbose=True,
                    checkpoint_path=f"/tmp/feature_selection_checkpoint.params",
                    unique_id=f"{iteration}"
                )
                feature_selector.fit(X_train, y_train)
                selected_features = feature_selector.selected_features_

                if len(selected_features) == 0:
                    logger.warning(f"No features selected on attempt {attempt + 1}. Retrying...")
                    mlflow.log_metric(f"attempt_{attempt + 1}_selected_features", 0)
                    attempt += 1
                    mlflow.log_param(f"attempt_{attempt}_status", "no_features_selected")

            if len(selected_features) == 0:
                logger.error("Failed to select features after maximum attempts. Proceeding with all features.")
                selected_features = X_train.columns.tolist()
                mlflow.set_tag("feature_selection_status", "fallback_to_all_features")
            else:
                mlflow.set_tag("feature_selection_status", "successful")
            
            # If successful, log selected features
            selected_features_df = pd.DataFrame(selected_features, columns=['feature'])
            mlflow.log_table(selected_features_df, artifact_file=f"selected_features_{iteration}.json")

            # Update the feature selection status
            mlflow.set_tag('feature_selection_status', 'successful')

            mlflow.log_param("number_of_selected_features", len(selected_features))

            logger.info(f"Number of selected features in iteration {iteration}: {len(selected_features)}")
    
            return pd.Index(selected_features)
        except Exception as e:
            # Log the exception details
            logger.exception("An error occurred during feature selection.")
            mlflow.set_tag('feature_selection_status', 'failed')
            # Optionally, log the exception message
            mlflow.log_text(str(e), artifact_file="errors/feature_selection_error.txt")
            # Re-raise the exception to be handled by the calling function
            raise e

def bootstrap_models(
    sample_processor: SampleProcessor,
    genotype_processor: GenotypeProcessor,
    data: hl.MatrixTable,
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    random_state=42
) -> pd.DataFrame:
    """
    Perform bootstrapping of models based on the configuration using Joblib with Loky backend.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Starting bootstrapping of models.")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    client = MlflowClient()
    
    # Try to create the experiment, handle the case where it already exists
    try:
        experiment_id = client.create_experiment(trackingConfig.experiment_name)
    except MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            experiment = client.get_experiment_by_name(trackingConfig.experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
                logger.info(f"Experiment '{trackingConfig.experiment_name}' already exists. Using existing experiment.")
            else:
                raise e
        else:
            raise e

    mlflow.set_experiment(experiment_id=experiment_id)

    # Pivot the Hail MatrixTable
    spark_df = genotype_processor.to_spark_df(data)
    pivoted_df = genotype_processor.pivot_genotypes(spark_df)

    # Persist pivoted_df as Parquet
    parquet_path = f"/tmp/{trackingConfig.experiment_name}.parquet"
    logger.info(f"Saving pivoted DataFrame to Parquet at {parquet_path}")
    pivoted_df.write.parquet(parquet_path, mode="overwrite")

    total_variant_count = len(pivoted_df.columns) - 1  # Subtract the sample ID column

    # Force execution of Hail operations and close the Hail context
    hl.stop()

    # Feature Selection Step
    feature_selection_result = parallel_feature_selection(
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        samplingConfig=samplingConfig,
        num_iterations=10,  # Number of parallel feature selection iterations
        total_variants=total_variant_count,
        random_state=random_state,
        trackingConfig=trackingConfig
    )

    selected_features = feature_selection_result.selected_features
    num_variants = feature_selection_result.num_variants
    total_variants = feature_selection_result.total_variants
    confidence_level = feature_selection_result.confidence_level

    # Prepare arguments for parallel processing of bootstrap iterations
    iterations = range(samplingConfig.bootstrap_iterations)
    process_func = partial(
        process_iteration,
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        genotype_processor=genotype_processor,
        samplingConfig=samplingConfig,
        trackingConfig=trackingConfig,
        stack=stack,
        selected_features=selected_features,
        num_variants=num_variants,
        total_variants=total_variants,
        confidence_level=confidence_level,
        random_state=random_state
    )

    # Use Joblib with Loky backend for parallel processing
    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(process_func)(i) for i in tqdm(iterations, desc="Bootstrapping Progress")
    )

    logger.info("Completed bootstrapping of models.")

    # Convert results to DataFrame
    records = []
    for result in results:
        iteration = result["iteration"]
        for model in result["models"]:
            records.append({
                "iteration": iteration,
                "model_name": model["model_name"],
                "test_auc": model["test_auc"],
                "test_f1": model["test_f1"],
                "best_params": model["best_params"],
                "run_id": model["run_id"]
            })

    return pd.DataFrame.from_records(records)

def create_and_log_visualizations(model_name, y_true, y_pred, trackingConfig, set_type="test", table_name="crossval", num_variants=None, total_variants=None, confidence_level=None):
    """
    Create and log visualizations directly to MLflow.
    
    Args:
        model_name (str): Name of the model.
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
        trackingConfig (TrackingConfig): Configuration for tracking.
        set_type (str, optional): Type of dataset ('test', 'train', 'holdout'). Defaults to "test".
        table_name (str, optional): Name of the table. Defaults to "crossval".
        num_variants (int, optional): Number of selected features. Defaults to None.
        total_variants (int, optional): Total number of features before selection. Defaults to None.
        confidence_level (float, optional): Confidence level used in feature selection. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    
    # Sanitize set_type and table_name once
    sanitized_set_type = sanitize_mlflow_name(set_type)
    sanitized_table_name = sanitize_mlflow_name(table_name) if table_name != "crossval" else ""
    
    # Construct base paths
    base_metrics_path = f"{sanitized_set_type}"
    base_plots_path = f"plots/{sanitized_set_type}"
    
    if sanitized_table_name:
        base_metrics_path += f"/{sanitized_table_name}"
        base_plots_path += f"/{sanitized_table_name}"
    
    def log_plot(plot_func, artifact_name, square=False):
        """
        Helper function to create, log, and close a plot.
        
        Args:
            plot_func (callable): Function that takes a Figure and Axes object and plots on it.
            artifact_name (str): Name of the artifact file to log.
            square (bool, optional): Whether to enforce a square aspect ratio. Defaults to False.
        
        Returns:
            Any: Result returned by the plot_func.
        """
        if square:
            fig_size = (8, 8)  # Smaller figure size for square plots
        else:
            fig_size = (10, 6)  # Default size for non-square plots

        # Create figure and axes with constrained_layout to handle layout automatically
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        result = plot_func(fig, ax)

        if square:
            # Maintain square aspect ratio where meaningful
            ax.set_aspect('equal')

        artifact_path = f"{base_plots_path}/{artifact_name}"
        try:
            mlflow.log_figure(fig, artifact_path)
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to log figure {artifact_name} due to: {e}")
        plt.close(fig)
        return result

    # Construct the subtitle with feature selection info if available
    if num_variants is not None and total_variants is not None and confidence_level is not None:
        subtitle = (f"{num_variants} of {total_variants} variants selected at "
                    f"{confidence_level*100:.0f}% credible interval\n")
    else:
        subtitle = ""
    
    # Existing subtitle information
    num_cases = np.sum(y_true != 0)
    num_controls = np.sum(y_true == 0)
    metrics = {
        f"{base_metrics_path}/num_cases": num_cases,
        f"{base_metrics_path}/num_controls": num_controls
    }
    mlflow.log_metrics(metrics)

    # Append to the subtitle
    subtitle += f"Cases: {num_cases}, Controls: {num_controls}"
    if table_name != "crossval":
        subtitle = f"Evaluated on: {table_name}\n{subtitle}"
    subtitle = f"\n{subtitle}"

    # ROC Curve
    def plot_roc(fig, ax):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random Guess (AUC=0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()

        # Set the title at the figure level
        fig.suptitle(f'ROC Curve\n{trackingConfig.name}\n{subtitle}')
        return fpr, tpr, thresholds, auc_score
    
    # Precision-Recall Curve
    def plot_pr(fig, ax):
        display = PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=ax, name=f'Precision-Recall')
        avg_score = average_precision_score(y_true, y_pred)
        no_skill = len(y_true[y_true == 1]) / len(y_true)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (Precision={no_skill:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Set the title at the figure level
        fig.suptitle(f'Precision-Recall Curve\n{trackingConfig.name}\n{subtitle}')
        return display.precision, display.recall, avg_score

    if len(np.unique(y_true)) > 1:
        # Plot ROC Curve with square aspect ratio
        roc_result = log_plot(plot_roc, "roc_curve.svg", square=True)
        if roc_result:
            fpr, tpr, thresholds, auc_score = roc_result
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            mlflow.log_table(roc_df, f"{base_metrics_path}/roc_curve.json")
            mlflow.log_metrics({
                f"{base_metrics_path}/roc_auc": auc_score,
                f"{base_metrics_path}/roc_fpr_mean": float(np.mean(fpr)),
                f"{base_metrics_path}/roc_tpr_mean": float(np.mean(tpr)),
                f"{base_metrics_path}/roc_thresholds_mean": float(np.mean(thresholds))
            })

            # Plot Precision-Recall Curve with square aspect ratio
            pr_result = log_plot(plot_pr, "pr_curve.svg", square=True)
            if pr_result:
                precision, recall, avg_score = pr_result
                pr_df = pd.DataFrame({
                    'precision': precision,
                    'recall': recall,
                })
                mlflow.log_table(pr_df, f"{base_metrics_path}/pr_curve.json")
                mlflow.log_metrics({
                    f"{base_metrics_path}/pr_auc": avg_score,
                    f"{base_metrics_path}/pr_precision_mean": float(np.mean(precision)),
                    f"{base_metrics_path}/pr_recall_mean": float(np.mean(recall)),
                })
    else:
        logger.warning(f"Not enough unique values in y_true to compute ROC & precision-recall for {base_metrics_path}")

    # Distribution of Predictions
    def plot_dist(fig, ax):
        # Plot the histogram
        sns.histplot(y_pred, ax=ax, bins=100, kde=True,
                  line_kws={'color': 'crimson', 'lw': 5, 'ls': ':'})
    
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_aspect('auto')  # Use default aspect ratio for histograms

        # Set x-ticks every 0.05 and labels every 0.1
        ax.set_xticks(np.arange(0, 1.05, 0.05))
        ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)

        # Set the title at the figure level
        fig.suptitle(f'Distribution of Predictions\n{trackingConfig.name}\n{subtitle}')

    # Plot Distribution with default aspect ratio
    log_plot(plot_dist, "pred_distribution.svg", square=False)

    # Confusion Matrix
    def plot_cm(fig, ax):
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Set the title at the figure level
        fig.suptitle(f'Confusion Matrix\n{trackingConfig.name}\n{subtitle}')
        return cm

    # Plot Confusion Matrix with square aspect ratio
    cm_result = log_plot(plot_cm, "confusion_matrix.svg", square=True)
    if cm_result is not None:
        if cm_result.size == 4:
            tn, fp, fn, tp = cm_result.ravel()
            cm_df = pd.DataFrame({
                'true_negatives': [tn],
                'false_positives': [fp],
                'false_negatives': [fn],
                'true_positives': [tp]
            })
            mlflow.log_table(cm_df, f"{base_metrics_path}/confusion_matrix.json")
            mlflow.log_metrics({
                f"{base_metrics_path}/confusion_matrix/true_negatives": tn,
                f"{base_metrics_path}/confusion_matrix/false_positives": fp,
                f"{base_metrics_path}/confusion_matrix/false_negatives": fn,
                f"{base_metrics_path}/confusion_matrix/true_positives": tp
            })
        else:
            logger.warning(f"Confusion matrix has unexpected shape: {cm_result.shape}")
    plt.close('all')

def plot_and_log_convergence(metric_name,search, y_true, trackingConfig, num_variants=None, total_variants=None, confidence_level=None):
    """
    Plot hyperparameter convergence using skopt's plot_convergence and log it to MLflow.

    Args:
        search (BayesSearchCV): The fitted Bayesian search object.
        model_name (str): Name of the model.
        trackingConfig (TrackingConfig): Configuration for tracking.
        num_variants (int, optional): Number of selected features. Defaults to None.
        total_variants (int, optional): Total number of features before selection. Defaults to None.
        confidence_level (float, optional): Confidence level used in feature selection. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    model_name = search.estimator.__class__.__name__

    # Construct the subtitle with feature selection info if available
    if num_variants is not None and total_variants is not None and confidence_level is not None:
        subtitle = (f"{num_variants} of {total_variants} variants selected at "
                    f"{confidence_level*100:.0f}% credible interval\n")
    else:
        subtitle = ""

    # Existing subtitle information
    num_cases = np.sum(y_true != 0)
    num_controls = np.sum(y_true == 0)
    subtitle += f"Cases: {num_cases}, Controls: {num_controls}\n"

    # Set the title of the plot
    title = f'Hyperparameter Convergence for {model_name} ({metric_name})\n{trackingConfig.name}\n\n{subtitle}'

    try:
        # Create a new figure for the convergence plot
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        
        # Plot the convergence using skopt's plot_convergence
        plot_convergence(search.optimizer_results_, ax=ax)
        
        # Set the title of the plot
        fig.suptitle(title, fontsize=14)

        fig.tight_layout()
        
        # Log the figure to MLflow
        mlflow.log_figure(fig, "plots/train/hyperparam_convergence.svg")
        
        # Close the figure to free up memory
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Failed to plot hyperparameter convergence for {model_name}: {e}")
        raise e