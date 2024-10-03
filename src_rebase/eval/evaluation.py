import io
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, log_loss, matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, Union
import hail as hl
import ray  # New import for Ray
from pyspark.sql import functions as F, DataFrame as SparkDataFrame
from skopt import BayesSearchCV  
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

from data.sample_processor import SampleProcessor
from config import SamplingConfig, TrackingConfig
from data.genotype_processor import GenotypeProcessor

# Initialize Ray at the module level
ray.init(ignore_reinit_error=True)

def get_feature_importances(model, feature_labels):
    """Get feature importances from fitted model and create SHAP explainer"""
    if hasattr(model, 'coef_'):
        model_coefficient_df = pd.DataFrame()
        if len(model.coef_.shape) > 1:
            try:
                model_coefficient_df[f"feature_importances__control"] = model.coef_[0]
                model_coefficient_df[f"feature_importances__case"] = model.coef_[1]
            except IndexError:
                model_coefficient_df[f"feature_importances"] = model.coef_[0]
        else:
            model_coefficient_df[f"feature_importances"] = model.coef_.flatten()
    elif hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        model_coefficient_df = pd.DataFrame()
        model_coefficient_df[f"feature_importances"] = model.feature_importances_
    else:
        model_coefficient_df = None

    if isinstance(model_coefficient_df, pd.DataFrame):
        model_coefficient_df.index = feature_labels
        model_coefficient_df.index.name = "feature_name"

    return model_coefficient_df

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    y_pred_binary = (y_pred > 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "brier": brier_score_loss(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred),
        "precision": precision_recall_fscore_support(y_true, y_pred_binary, average=None)[0],
        "recall": precision_recall_fscore_support(y_true, y_pred_binary, average=None)[1]
    }

def log_mlflow_metrics(metrics, best_params, model, X_test, y_pred_test, df_y_pred_train, df_y_pred_test, model_coefficient_df):
    """Log metrics and artifacts to MLflow."""
    signature = infer_signature(X_test, y_pred_test)
    with mlflow.start_run():
        mlflow.set_tag("model", model.__class__.__name__)
        mlflow.log_params(model.get_params())
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value['test'])
            mlflow.log_metric(f"train_{metric_name}", metric_value['train'])
        mlflow.log_table(data=df_y_pred_test, artifact_file="test_predictions.json")
        mlflow.log_table(data=df_y_pred_train, artifact_file="train_predictions.json")
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        if model_coefficient_df is not None:
            mlflow.log_table(data=model_coefficient_df, artifact_file="feature_importances.json")

def evaluate_model(model, search_spaces, X_train, y_train, X_test, y_test, trackingConfig: TrackingConfig, n_iter=10):
    """Perform Bayesian hyperparameter optimization for a model."""
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
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        search.fit(X_train, y_train)
        
        y_pred_train = search.predict_proba(X_train)[:, 1]
        y_pred_test = search.predict_proba(X_test)[:, 1]

        df_y_pred_train = y_train.to_frame().assign(y_pred=y_pred_train).reset_index()
        df_y_pred_test = y_test.to_frame().assign(y_pred=y_pred_test).reset_index()
        
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        metrics = {metric: {'train': train_metrics[metric], 'test': test_metrics[metric]} 
                   for metric in train_metrics.keys()}
        
        best_params = {key.replace('classifier__', ''): value for key, value in search.best_params_.items()}

        feature_labels = X_train.columns
        model_coefficient_df = get_feature_importances(search.best_estimator_['classifier'], feature_labels)

        mlflow.set_tracking_uri(trackingConfig.tracking_uri)
        mlflow.set_experiment(trackingConfig.experiment_name)
        log_mlflow_metrics(metrics, best_params, model, X_test, y_pred_test, df_y_pred_train, df_y_pred_test, model_coefficient_df)

        return metrics, best_params
    except Exception as e:
        raise e

def prepare_data(sample_processor, genotype_processor, data, train_samples, test_samples):
    logger = logging.getLogger(__name__)

    # Extract sample IDs mapped to VCF samples
    train_sample_ids = list(train_samples.values())
    test_sample_ids = list(test_samples.values())

    logger.info("Fetching genotypes for training and testing samples.")
    train_genotypes = genotype_processor.fetch_genotypes(data=data, sample_ids=train_sample_ids, return_spark=True)
    test_genotypes = genotype_processor.fetch_genotypes(data=data, sample_ids=test_sample_ids, return_spark=True)

    logger.info("Converting Spark DataFrame to Pandas DataFrame for ML processing.")
    try:
        X_train = train_genotypes.toPandas().set_index('sample_id')
        X_test = test_genotypes.toPandas().set_index('sample_id')
        
        y_train = sample_processor.get_labels(train_sample_ids)
        y_test = sample_processor.get_labels(test_sample_ids)

        # Ensure that the indices align
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to convert Spark DataFrame to Pandas DataFrame: {e}")
        raise e

def bootstrap_models(
    sample_processor: SampleProcessor,
    genotype_processor: GenotypeProcessor,
    data: Union[hl.MatrixTable, SparkDataFrame],
    samplingConfig: SamplingConfig,
    trackingConfig: TrackingConfig,
    stack: Dict,
    random_state=42
) -> pd.DataFrame:
    """
    Perform bootstrapping of models based on the configuration.

    Args:
        sample_processor (SampleProcessor): Instance of SampleProcessor.
        genotype_processor (GenotypeProcessor): Instance of GenotypeProcessor.
        data (Union[hl.MatrixTable, SparkDataFrame]): The processed dataset.
        samplingConfig (SamplingConfig): Configuration object containing bootstrap_iterations and other settings.
        trackingConfig (TrackingConfig): Configuration object containing tracking parameters.
        stack (Dict): Dictionary of models and their hyperparameter distributions.
        random_state (int): Seed for random number generation.

    Returns:
        pd.DataFrame: Aggregated performance metrics across all iterations and models.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting bootstrapping of models.")

    results = []

    for iteration in range(samplingConfig.bootstrap_iterations):
        logger.info(f"Bootstrapping iteration {iteration + 1}/{samplingConfig.bootstrap_iterations}")

        # Adjust random_state to ensure different splits for each iteration
        current_random_state = random_state + iteration

        # Draw a new train-test split for each bootstrap iteration
        train_test_sample_ids = sample_processor.draw_train_test_split(
            test_size=samplingConfig.test_size,
            random_state=current_random_state
        )

        train_samples = train_test_sample_ids['train']['samples']
        test_samples = train_test_sample_ids['test']['samples']

        logger.info(f"Number of training samples: {len(train_samples)}")
        logger.info(f"Number of testing samples: {len(test_samples)}")

        X_train, X_test, y_train, y_test = prepare_data(sample_processor, genotype_processor, data, train_samples, test_samples)

        if X_train is None:
            continue  # Skip this iteration if data preparation failed

        # List to hold Ray task references
        tasks = []

        # Iterate over each model in the stack and submit evaluation tasks to Ray
        for model, search_spaces in stack.items():
            logger.info(f"Submitting training tasks for model: {model.__class__.__name__}")

            evaluate_model(
                model=model,
                search_spaces=search_spaces,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                trackingConfig=trackingConfig,
                n_iter=10  # You can adjust this based on your needs
            )
        

        # Retrieve results from Ray
        try:
            ray_results = ray.get(tasks)
        except Exception as e:
            logger.error(f"Ray task retrieval failed: {e}")
            continue

    # Shutdown Ray after all iterations are complete
    ray.shutdown()
    logger.info("Completed bootstrapping of models.")

    # Convert results to DataFrame for aggregation
    results_df = pd.DataFrame(results)

    return results_df

def create_and_log_visualizations(y_true, y_pred, trackingConfig):
    """Create and log visualizations directly to MLflow."""
    
    def log_plot(plot_func, artifact_name):
        plt.figure(figsize=(10, 6))
        plot_func()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        mlflow.log_image(buf, artifact_name)
        plt.close()

    # ROC Curve
    def plot_roc():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve\n{trackingConfig.name}')

    log_plot(plot_roc, "roc_curve.png")

    # Precision-Recall Curve
    def plot_pr():
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve\n{trackingConfig.name}')

    log_plot(plot_pr, "pr_curve.png")

    # Distribution of Predictions
    def plot_dist():
        sns.histplot(y_pred, kde=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title(f'Distribution of Predictions\n{trackingConfig.name}')

    log_plot(plot_dist, "pred_distribution.png")

    # Confusion Matrix
    def plot_cm():
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix\n{trackingConfig.name}')

    log_plot(plot_cm, "confusion_matrix.png")