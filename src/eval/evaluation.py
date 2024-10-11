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
from joblib import Parallel, delayed
from mlflow.models import infer_signature
from skopt import BayesSearchCV
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config import SamplingConfig, TrackingConfig
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from eval.feature_selection import BayesianFeatureSelector

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
    
    # Calculate aggregate metrics
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = None
    
    aggregate_metrics = {
        "auc": auc,
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, average='binary'),
        "recall": recall_score(y_true, y_pred_binary, average='binary')
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
    
    # mlflow.log_params(best_params)
    
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

def evaluate_model(model, search_spaces, X_train, y_train, X_test, y_test, genotype_processor: GenotypeProcessor, sample_processor: SampleProcessor, trackingConfig: TrackingConfig, n_iter=10, is_worker=True):
    """Perform Bayesian hyperparameter optimization for a model and evaluate on holdout samples."""
    try:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', model)
        ])
        
        search_spaces = {f'classifier__{key}': value for key, value in search_spaces.items()}
        
        search = BayesSearchCV(
            pipeline,
            search_spaces,
            n_iter=n_iter,
            cv=3,
            n_jobs=1 if is_worker else -1,
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
            mlflow.log_params(model.get_params())
            mlflow.set_tag("model", model.__class__.__name__)
            
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

            # Updated calls with sanitized table_name
            create_and_log_visualizations(
                y_test,
                y_pred_test,
                trackingConfig,
                set_type="test",
                table_name="crossval"  # Default table name for cross-validation
            )
            create_and_log_visualizations(
                y_train,
                y_pred_train,
                trackingConfig,
                set_type="train",
                table_name="crossval"  # Default table name for cross-validation
            )

            # --- Holdout Evaluation ---
            logger = logging.getLogger(__name__)
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
                    parquet_path=f"data/{trackingConfig.experiment_name}.parquet",
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

                # Log holdout visualizations with actual table_name
                create_and_log_visualizations(
                    y_holdout,
                    y_pred_holdout,
                    trackingConfig,
                    set_type="holdout",
                    table_name=table_name  # Actual holdout table name
                )

            logger.info("Completed holdout evaluation.")

        return metrics, y_pred_test, y_pred_train, best_params, run.info.run_id
    except Exception as e:
        logger = logging.getLogger(__name__)
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
                    lr=1e-3,
                    confidence_level=0.0005,
                    num_samples=2000,
                    patience=800,
                    fallback_percentile=95,
                    validation_split=0.2,
                    batch_size=512,
                    verbose=True,
                    checkpoint_path=f"/tmp/feature_selection_checkpoint_{iteration}.params",
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

def parallel_feature_selection(
    parquet_path: str,
    sample_processor: SampleProcessor,
    samplingConfig: SamplingConfig,
    num_iterations: int = 1,
    random_state: int = 42,
    trackingConfig: TrackingConfig = None
) -> pd.Index:
    """
    Run feature selection in parallel over multiple splits and collect overlapping features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting parallel feature selection.")

    # Prepare arguments for parallel processing
    iterations = range(num_iterations)
    process_func = partial(
        feature_selection_iteration,
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        samplingConfig=samplingConfig,
        random_state=random_state,
        trackingConfig=trackingConfig
    )

    # Use Joblib for parallel processing
    selected_features_list = Parallel(n_jobs=1, backend="loky", verbose=10)(
        delayed(process_func)(i) for i in tqdm(iterations, desc="Feature Selection Progress")
    )

    logger.info("Completed parallel feature selection.")

    # Combine selected features from all iterations
    feature_counts = pd.Series(np.concatenate(selected_features_list)).value_counts()

    # Determine overlapping features using a statistical threshold
    # For soft overlap, select features that appear in at least a certain fraction of iterations
    threshold = num_iterations * 0.5  # e.g., features selected in at least 50% of iterations
    overlapping_features = feature_counts[feature_counts >= threshold].index

    logger.info(f"Number of overlapping features selected: {len(overlapping_features)}")

    return overlapping_features

def process_iteration(
    iteration: int,
    parquet_path: str,
    sample_processor: SampleProcessor,
    genotype_processor: GenotypeProcessor,
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    selected_features: pd.Index,
    random_state: int,
) -> Dict:
    """
    Process a single bootstrap iteration using the selected features.
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
                genotype_processor=genotype_processor,
                sample_processor=sample_processor,
                n_iter=10
            )

            mlflow.log_metric(f"{model_name}_test_auc", metrics['aggregate']['test']['auc'])
            mlflow.log_metric(f"{model_name}_test_f1", metrics['aggregate']['test']['f1'])
            mlflow.log_param(f"{model_name}_run_id", child_run_id)

            iteration_results["models"].append({
                "model_name": model_name,
                "test_auc": metrics['aggregate']['test']['auc'],
                "test_f1": metrics['aggregate']['test']['f1'],
                "best_params": best_params,
                "run_id": child_run_id
            })

    return iteration_results

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
    
    # Initialize MlflowClient with the tracking URI
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
    parquet_path = f"data/{trackingConfig.experiment_name}.parquet"
    logger.info(f"Saving pivoted DataFrame to Parquet at {parquet_path}")
    pivoted_df.write.parquet(parquet_path, mode="overwrite")

    # Force execution of Hail operations and close the Hail context
    hl.stop()

    # Run parallel feature selection
    selected_features = parallel_feature_selection(
        parquet_path=parquet_path,
        sample_processor=sample_processor,
        samplingConfig=samplingConfig,
        num_iterations=1,  # Number of parallel feature selection iterations
        random_state=random_state,
        trackingConfig=trackingConfig
    )

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

def create_and_log_visualizations(y_true, y_pred, trackingConfig, set_type="test", table_name="crossval"):
    """Create and log visualizations directly to MLflow."""
    logger = logging.getLogger(__name__)
    
    # Sanitize set_type and table_name once
    sanitized_set_type = sanitize_mlflow_name(set_type)
    sanitized_table_name = sanitize_mlflow_name(table_name) if table_name != "crossval" else ""
    
    # Construct base paths
    base_metrics_path = f"metrics/{sanitized_set_type}"
    base_plots_path = f"plots/{sanitized_set_type}"
    
    if sanitized_table_name:
        base_metrics_path += f"/{sanitized_table_name}"
        base_plots_path += f"/{sanitized_table_name}"
    
    def log_plot(plot_func, artifact_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        result = plot_func(ax)
        fig.tight_layout()
        artifact_path = f"{base_plots_path}/{artifact_name}"
        mlflow.log_figure(fig, artifact_path)
        plt.close(fig)
        return result

    # Log number of cases and controls
    num_cases = np.sum(y_true != 0)
    num_controls = np.sum(y_true == 0)
    metrics = {
        f"{base_metrics_path}/num_cases": num_cases,
        f"{base_metrics_path}/num_controls": num_controls
    }
    mlflow.log_metrics(metrics)

    # Create a subtitle with case/control counts and table name if applicable
    subtitle = f"Cases: {num_cases}, Controls: {num_controls}"
    if table_name != "crossval":
        subtitle = f"Evaluated on: {table_name}\n{subtitle}"

    # ROC Curve
    def plot_roc(ax):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr, label='ROC Curve')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve\n{trackingConfig.name}\n{subtitle}')
        ax.legend()
        return fpr, tpr, thresholds

    if len(np.unique(y_true)) > 1:
        roc_result = log_plot(plot_roc, "roc_curve.png")
        if roc_result:
            fpr, tpr, thresholds = roc_result
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            mlflow.log_table(roc_df, f"{base_metrics_path}/roc_curve.json")
            mlflow.log_metrics({
                f"{base_metrics_path}/roc_fpr_mean": float(np.mean(fpr)),
                f"{base_metrics_path}/roc_tpr_mean": float(np.mean(tpr)),
                f"{base_metrics_path}/roc_thresholds_mean": float(np.mean(thresholds))
            })
    else:
        logger.warning(f"Not enough unique values in y_true to compute ROC curve for {base_metrics_path}")

    # Precision-Recall Curve
    def plot_pr(ax):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        ax.plot(recall, precision, label='Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve\n{trackingConfig.name}\n{subtitle}')
        ax.legend()
        return precision, recall, thresholds
    
    pr_result = log_plot(plot_pr, "pr_curve.png")
    if pr_result:
        precision, recall, thresholds = pr_result
        pr_df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(thresholds, np.nan)  # Add NaN to match length of precision/recall
        })
        mlflow.log_table(pr_df, f"{base_metrics_path}/pr_curve.json")
        mlflow.log_metrics({
            f"{base_metrics_path}/pr_precision_mean": float(np.mean(precision)),
            f"{base_metrics_path}/pr_recall_mean": float(np.mean(recall)),
            f"{base_metrics_path}/pr_thresholds_mean": float(np.mean(thresholds))
        })

    # Distribution of Predictions
    def plot_dist(ax):
        sns.histplot(y_pred, kde=True, ax=ax)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Predictions\n{trackingConfig.name}\n{subtitle}')
    
    log_plot(plot_dist, "pred_distribution.png")

    # Confusion Matrix
    def plot_cm(ax):
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix\n{trackingConfig.name}\n{subtitle}')
        return cm

    cm_result = log_plot(plot_cm, "confusion_matrix.png")
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
