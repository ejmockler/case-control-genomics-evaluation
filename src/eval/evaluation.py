import gc
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
import ray
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
import jax.numpy as jnp

from config import SamplingConfig, TrackingConfig
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from eval.feature_selection import BayesianFeatureSelector, FeatureSelectionResult

# Initialize Ray
ray.init(ignore_reinit_error=True)

def get_feature_importances(model, feature_labels):
    """Get feature importances from fitted model."""
    if hasattr(model, 'beta_loc_'):
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df['feature_name'] = feature_labels
        model_coefficient_df['feature_importances'] = model.beta_loc_
    elif hasattr(model, 'coef_'):
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df['feature_name'] = feature_labels
        if len(model.coef_.shape) > 1:
            model_coefficient_df['feature_importances'] = model.coef_[0]
        else:
            model_coefficient_df['feature_importances'] = model.coef_.flatten()
    elif hasattr(model, 'feature_importances_'):
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df['feature_name'] = feature_labels
        model_coefficient_df['feature_importances'] = model.feature_importances_
    elif hasattr(model, 'feature_log_prob_'):
        if model.feature_log_prob_.shape[0] != 2:
            raise ValueError("Log-odds ratio is only applicable for binary classification.")

        # Compute log-odds ratio
        log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]

        model_coefficient_df = pd.DataFrame({
            'feature_name': feature_labels,
            'feature_importances': log_odds,
        })

        # Add individual class log probabilities
        model_coefficient_df['class_0_log_prob'] = model.feature_log_prob_[0]
        model_coefficient_df['class_1_log_prob'] = model.feature_log_prob_[1]

        # Sort by importance
        model_coefficient_df = model_coefficient_df.sort_values(by='feature_importances', ascending=False)
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

def evaluate_model(
        model, 
        search_spaces, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        parquet_path: str,
        sample_processor: SampleProcessor, 
        trackingConfig: TrackingConfig, 
        samplingConfig: SamplingConfig, 
        feature_selection_result: FeatureSelectionResult,
        n_iter=10, 
        is_worker=True, 
        iteration: int = 0,
        outer_iteration: int = 1):
    """Perform Bayesian hyperparameter optimization for a model and evaluate on holdout samples."""
    logger = logging.getLogger(__name__)  # Use module-level logger
    
    # Retrieve model parameters
    params = model.get_params()
    
    # Check if 'unique_id' is a valid parameter
    if 'unique_id' in params:
        params['unique_id'] = f"{iteration}"
    
    # Initialize a fresh model instance with updated parameters
    process_safe_model = model.__class__(**params)
    
    try:
        # Set up MLflow tracking
        mlflow.set_tracking_uri(trackingConfig.tracking_uri)
        mlflow.set_experiment(trackingConfig.experiment_name)

        # Start a run for this model evaluation
        with mlflow.start_run(run_name=f"{model.__class__.__name__}", nested=True) as run:
            if outer_iteration is not None:
                mlflow.set_tag("outer_iteration", outer_iteration)
            
            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('classifier', process_safe_model)
            ])
            
            search_spaces = {f'classifier__{key}': value for key, value in search_spaces.items()}
            
            search = BayesSearchCV(
                pipeline,
                search_spaces,
                n_iter=n_iter,
                cv=samplingConfig.crossval_folds,
                n_jobs=-1 if not is_worker else 1,
                n_points=10,
                scoring='roc_auc'
            )
            
            search.fit(jnp.array(X_train.values), jnp.array(y_train.values))
                    
            # Use the fitted pipeline for predictions
            if hasattr(search.best_estimator_['classifier'], "predict_proba"):
                y_pred_train = search.predict_proba(jnp.array(X_train.values))[:, 1]
                y_pred_test = search.predict_proba(jnp.array(X_test.values))[:, 1]
            else:
                y_pred_train = search.predict(jnp.array(X_train.values))
                y_pred_test = search.predict(jnp.array(X_test.values))

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
            
            mlflow.set_tag("model", model.__class__.__name__)

            num_variants = feature_selection_result.num_variants
            total_variants = feature_selection_result.total_variants
            credible_interval = feature_selection_result.credible_interval

            plot_and_log_convergence('ROC AUC', search, y_train, trackingConfig, num_variants, total_variants, credible_interval)
            
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
                feature_selection_result=feature_selection_result
            )
            create_and_log_visualizations(
                model.__class__.__name__,
                y_train,
                y_pred_train,
                trackingConfig,
                set_type="train",
                table_name="crossval",  # Default table name for cross-validation
                feature_selection_result=feature_selection_result
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
                    parquet_path=parquet_path,
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
                classifier = search.best_estimator_.named_steps['classifier']
                if hasattr(classifier, 'predict_proba'):
                    y_pred_holdout = classifier.predict_proba(jnp.array(X_holdout_scaled))[:, 1]
                else:
                    y_pred_holdout = classifier.predict(jnp.array(X_holdout_scaled))

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
                    feature_selection_result=feature_selection_result
                )

            logger.info("Completed holdout evaluation.")

        return metrics, y_pred_test, y_pred_train, best_params, run.info.run_id
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        raise e

def prepare_data(parquet_path: str, sample_processor, train_samples, selected_features: Optional[List[str]] = None, dataset: str = 'crossval', test_samples: Optional[List[str]] = []):
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

@ray.remote(num_cpus=1)
def process_iteration(
    iteration: int,
    parquet_path: str,
    validation_samples: List[str],
    sample_processor: SampleProcessor,
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    feature_selection_result: FeatureSelectionResult,
    random_state: int,
    outer_iteration: int = 1
) -> Dict:
    """
    Process a single bootstrap iteration using the selected features and feature selection metrics.
    """
    logger = logging.getLogger(__name__)  # Use module-level logger
    logger.info(f"Bootstrapping iteration {iteration + 1}/{samplingConfig.bootstrap_iterations}")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    mlflow.set_experiment(trackingConfig.experiment_name)
    
    train_test_sample_ids = sample_processor.draw_train_test_split(
        test_size=samplingConfig.test_size,
        random_state=random_state + iteration * outer_iteration + 1,
        subset=validation_samples
    )

    train_samples = list(train_test_sample_ids['train']['samples'].values())
    test_samples = list(train_test_sample_ids['test']['samples'].values())

    selected_features = feature_selection_result.selected_features
    total_variants = feature_selection_result.total_variants
    credible_interval = feature_selection_result.credible_interval
    
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
        mlflow.log_param("credible_interval", credible_interval)
        mlflow.set_tag("outer_iteration", outer_iteration)

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
                parquet_path=parquet_path,
                trackingConfig=trackingConfig,
                samplingConfig=samplingConfig,
                sample_processor=sample_processor,
                feature_selection_result=feature_selection_result,
                n_iter=10,
                is_worker=True,
                iteration=iteration,
                outer_iteration=outer_iteration
            )

            mlflow.log_metric(f"{model_name}_test_f1", metrics['aggregate']['test']['f1'])
            mlflow.log_param(f"{model_name}_run_id", child_run_id)
    gc.collect()
    return

def bootstrap_models(
    sample_processor: SampleProcessor,
    parquet_path: str,  # Changed parameter
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    experiment_id: str,
    random_state=42,
    outer_iteration: int = 1
) -> pd.DataFrame:
    """
    Perform bootstrapping of models based on the configuration using Ray.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Starting bootstrapping of models.")

    mlflow.set_experiment(experiment_id=experiment_id)

    # Read total variant count from Parquet metadata
    total_variants = len(pd.read_parquet(parquet_path).columns) - 1  # Subtract sample_id column

    train_test_sample_ids = sample_processor.draw_train_test_split(
        test_size=samplingConfig.test_size,
        random_state=random_state * outer_iteration + 1,
    )

    train_samples = list(train_test_sample_ids['train']['samples'].values())
    validation_samples = list(train_test_sample_ids['test']['samples'].values())

    # Feature Selection Step
    feature_selection_result = parallel_feature_selection(
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        samplingConfig=samplingConfig,
        train_samples=train_samples,
        num_iterations=10,  # Number of parallel feature selection iterations
        total_variants=total_variants,
        random_state=random_state,
        trackingConfig=trackingConfig,
        outer_iteration=outer_iteration
    )

    # Prepare arguments for parallel processing of bootstrap iterations
    iterations = range(samplingConfig.bootstrap_iterations)
    tasks = [
        process_iteration.remote(
            i,
            parquet_path,
            validation_samples,
            sample_processor,
            samplingConfig,
            trackingConfig,
            stack,
            feature_selection_result,
            random_state,
            outer_iteration=outer_iteration
        )
        for i in iterations
    ]

    # Execute tasks in parallel using Ray
    ray.get(tasks)

    logger.info("Completed bootstrapping of models.")

    return

def parallel_feature_selection(
    parquet_path: str,
    sample_processor: SampleProcessor,
    train_samples: List[str],
    samplingConfig: SamplingConfig,
    num_iterations: int = 10,
    total_variants: int = None,
    random_state: int = 42,
    trackingConfig: TrackingConfig = None,
    outer_iteration: int = 1
) -> FeatureSelectionResult:
    """
    Run feature selection in parallel over multiple splits and collect overlapping features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting parallel feature selection.")

    # Launch feature selection tasks remotely
    tasks = [
        feature_selection_iteration.remote(
            i,
            parquet_path,
            train_samples,
            sample_processor,
            samplingConfig,
            random_state + i,
            trackingConfig,
            outer_iteration
        )
        for i in range(num_iterations)
    ]

    # Collect results
    results = ray.get(tasks)
    selected_features_list = [r[0] for r in results]
    fitted_feature_selectors = [r[1] for r in results]
    
    # Extract selected credible intervals from feature selectors
    selected_intervals = [
        fs.selected_credible_interval 
        for fs in fitted_feature_selectors 
        if hasattr(fs, 'selected_credible_interval') and fs.selected_credible_interval is not None
    ]

    # Calculate mean and std of selected intervals if available
    if selected_intervals:
        mean_interval = float(np.mean(selected_intervals))
        std_interval = float(np.std(selected_intervals))
        logger.info(f"Mean selected credible interval: {mean_interval:.4f} ± {std_interval:.4f}")
    else:
        mean_interval = samplingConfig.feature_credible_interval
        std_interval = None
        logger.warning("No valid selected credible intervals found. Using default interval.")

    # Count occurrences of each feature and keep only those appearing multiple times
    all_selected_features = np.concatenate(selected_features_list)
    feature_counts = pd.Series(all_selected_features).value_counts()
    features_to_keep = feature_counts[feature_counts > 1].index
    unique_selected_features = pd.Series(all_selected_features).drop_duplicates()
    unique_selected_features = unique_selected_features[unique_selected_features.isin(features_to_keep)]
    
    num_variants = len(unique_selected_features)
    logger.info(f"{num_variants} of {total_variants} variants selected across all iterations (after removing single occurrences).")
    
    # Log results to MLflow if tracking is enabled
    if trackingConfig:
        with mlflow.start_run(run_name=f"Feature_Selection_Summary") as run:
            mlflow.set_tag("outer_iteration", outer_iteration)
            
            if len(selected_intervals) > 0:
                mlflow.log_metric("mean_selected_credible_interval", mean_interval)
                mlflow.log_metric("std_selected_credible_interval", std_interval)
                mlflow.log_table(
                    pd.DataFrame({'Credible Interval': selected_intervals}), 
                    artifact_file="selected_credible_intervals.json"
                )
            
            # Log feature selection metrics
            feature_counts_df = feature_counts.reset_index()
            feature_counts_df.columns = ['Feature', 'Count']
            mlflow.log_table(feature_counts_df, artifact_file="all_feature_counts.json")
            
            kept_features_df = feature_counts[features_to_keep].reset_index()
            kept_features_df.columns = ['Feature', 'Count']
            mlflow.log_table(kept_features_df, artifact_file="selected_features.json")
            
            mlflow.log_metrics({
                "total_unique_features": len(feature_counts),
                "features_kept": len(features_to_keep),
                "features_filtered": len(feature_counts) - len(features_to_keep)
            })

    return FeatureSelectionResult(
        selected_features=unique_selected_features,
        num_variants=num_variants,
        total_variants=total_variants,
        credible_interval=samplingConfig.feature_credible_interval,
        selected_credible_interval=mean_interval,
        selected_credible_interval_deviation=std_interval
    )

@ray.remote(num_cpus=1)
def feature_selection_iteration(
    iteration: int,
    parquet_path: str,
    train_samples: List[str],
    sample_processor: SampleProcessor,
    samplingConfig: SamplingConfig,
    random_state: int,
    trackingConfig: TrackingConfig = None,
    outer_iteration: int = 1
) -> pd.Index:
    """
    Perform feature selection on a random split of the data.
    """
    logger = logging.getLogger(__name__)  # Use module-level logger
    logger.info(f"Feature selection iteration {iteration + 1}")

    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    mlflow.set_experiment(trackingConfig.experiment_name)

    # Start an MLflow run for this feature selection iteration
    with mlflow.start_run(run_name=f"Feature_Selection_{iteration}", nested=True) as run:
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("random_state", random_state + iteration)
        mlflow.set_tag('feature_selection_status', 'in_progress')
        mlflow.set_tag('outer_iteration', outer_iteration)

        try:
            # Prepare data by reading from Parquet
            X_train, _, y_train, _ = prepare_data(
                parquet_path=parquet_path,
                sample_processor=sample_processor,
                train_samples=train_samples,
            )
    
            # Feature Selection Step
            logger.info("Performing feature selection using Bayesian Feature Selector.")
            mlflow.log_param("feature_selector", "BayesianFeatureSelector")
            max_attempts = 5
            attempt = 0
            selected_features = []

            while len(selected_features) == 0 and attempt < max_attempts:
                feature_selector = BayesianFeatureSelector(
                    num_iterations=1000,
                    lr=1e-3,
                    credible_interval=samplingConfig.feature_credible_interval,
                    num_samples=3000,
                    patience=10,
                    validation_split=samplingConfig.test_size,
                    batch_size=int(np.floor(X_train.shape[0] * 0.1)),
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

                # Update final status and log results
                if len(selected_features) == 0:
                    mlflow.set_tag("feature_selection_status", "no_features_found")
                else:
                    mlflow.set_tag("feature_selection_status", "successful")
                    selected_features_df = pd.DataFrame(selected_features, columns=['feature'])
                    mlflow.log_table(selected_features_df, artifact_file=f"selected_features.json")
                    mlflow.log_param("number_of_selected_features", len(selected_features))

                logger.info(f"Number of selected features in iteration {iteration}: {len(selected_features)}")
                return pd.Index(selected_features), feature_selector

        except Exception as e:
            logger.error(f"An error occurred during feature selection: {str(e)}")
            mlflow.set_tag('feature_selection_status', 'failed')
            mlflow.log_text(str(e), artifact_file="errors/feature_selection_error.txt")
            raise e

def create_and_log_visualizations(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trackingConfig: TrackingConfig,
    set_type: str = "test",
    table_name: str = "crossval",
    feature_selection_result: Optional[FeatureSelectionResult] = None
) -> None:
    """
    Create and log visualizations directly to MLflow.
    
    Args:
        model_name (str): Name of the model.
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        trackingConfig (TrackingConfig): Configuration for tracking.
        set_type (str, optional): Type of dataset ('test', 'train', 'holdout'). Defaults to "test".
        table_name (str, optional): Name of the table. Defaults to "crossval".
        feature_selection_result (FeatureSelectionResult, optional): Results from feature selection,
            including selected features, credible intervals, and their uncertainties. Defaults to None.
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
            fig_size = (8, 8)
        else:
            fig_size = (10, 6)

        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            result = plot_func(fig, ax)

            if square:
                # Maintain square aspect ratio where meaningful
                ax.set_aspect('equal')

            artifact_path = f"{base_plots_path}/{artifact_name}"
            try:
                mlflow.log_figure(fig, artifact_path)
            except np.linalg.LinAlgError as e:
                logger.error(f"Failed to log figure {artifact_name} due to: {e}")
            return result
        finally:
            plt.close(fig)

    # Construct the subtitle with feature selection info
    if feature_selection_result is not None:
        num_variants = feature_selection_result.num_variants
        total_variants = feature_selection_result.total_variants
        
        # Format the credible interval with standard deviation if available
        if feature_selection_result.selected_credible_interval is not None:
            interval_str = f"{feature_selection_result.selected_credible_interval:.3f}"
            if feature_selection_result.selected_credible_interval_deviation is not None:
                interval_str += f" ± {feature_selection_result.selected_credible_interval_deviation:.3f}"
            feature_info = (f"{num_variants} of {total_variants} variants selected\n"
                          f"Selected credible interval: {interval_str}")
        else:
            feature_info = f"{num_variants} of {total_variants} variants selected"
        
        subtitle = f"\n{feature_info}\n"
    else:
        subtitle = ""

    # Add dataset information
    num_cases = np.sum(y_true != 0)
    num_controls = np.sum(y_true == 0)
    metrics = {
        f"{base_metrics_path}/num_cases": num_cases,
        f"{base_metrics_path}/num_controls": num_controls
    }
    mlflow.log_metrics(metrics)

    if table_name != "crossval":
        subtitle += f"Evaluated on: {table_name}\n"
    else:
        subtitle += f"Evaluated on: {set_type} split\n"
    subtitle += f"Cases: {num_cases}, Controls: {num_controls}"

    # ROC Curve
    def plot_roc(fig, ax):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random Guess (AUC=0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        fig.suptitle(f'ROC Curve for {model_name}\n{trackingConfig.name}\n{subtitle}')
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
        fig.suptitle(f'Precision-Recall Curve for {model_name}\n{trackingConfig.name}\n{subtitle}')
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
        fig.suptitle(f'Distribution of Predictions for {model_name}\n{trackingConfig.name}\n{subtitle}')

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
        fig.suptitle(f'Confusion Matrix for {model_name}\n{trackingConfig.name}\n{subtitle}')
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

def plot_and_log_convergence(metric_name,search, y_true, trackingConfig, num_variants=None, total_variants=None, credible_interval=None):
    """
    Plot hyperparameter convergence using skopt's plot_convergence and log it to MLflow.

    Args:
        search (BayesSearchCV): The fitted Bayesian search object.
        model_name (str): Name of the model.
        trackingConfig (TrackingConfig): Configuration for tracking.
        num_variants (int, optional): Number of selected features. Defaults to None.
        total_variants (int, optional): Total number of features before selection. Defaults to None.
        credible_interval (float, optional): Confidence level used in feature selection. Defaults to None.
    """
    logger = logging.getLogger(__name__)
    model_name = search.best_estimator_['classifier'].__class__.__name__

    # Construct the subtitle with feature selection info if available
    if num_variants is not None and total_variants is not None and credible_interval is not None:
        subtitle = (f"{num_variants} of {total_variants} variants "
                    f"(selected by {credible_interval*100:.0f}% credible interval)\n")
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

        plt.xticks(np.arange(0, search.n_iter+1, 5))

        # Set the title of the plot
        fig.suptitle(title, fontsize=14)

        fig.tight_layout()
        
        # Log the figure to MLflow
        mlflow.log_figure(fig, "plots/train/hyperparam_convergence.svg")
        
    except Exception as e:
        logger.error(f"Failed to plot hyperparameter convergence for {model_name}: {e}")
        raise e
    
    finally:
        plt.close(fig)