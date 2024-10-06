import io
import logging
from typing import Dict, Union

import hail as hl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from mlflow.models import infer_signature
from pyspark.sql import DataFrame as SparkDataFrame
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

from config import SamplingConfig, TrackingConfig
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from eval.feature_selection import BayesianFeatureSelector
# Initialize Ray at the module level
ray.init(ignore_reinit_error=True)

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
        "log_loss": -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
        "accuracy": y_true == y_pred_binary
    }
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "auc": roc_auc_score(y_true, y_pred),
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

def log_mlflow_metrics(metrics, best_params, model, X_test, y_test, y_pred_test, df_y_pred_train, df_y_pred_test, model_coefficient_df, trackingConfig):
    """Log metrics, artifacts, and visualizations to MLflow."""
    signature = infer_signature(X_test, y_pred_test)
    
    mlflow.log_params(best_params)
    
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
    mlflow.log_table(data=train_sample_metrics, artifact_file="train_sample_metrics.json")
    mlflow.log_table(data=test_sample_metrics, artifact_file="test_sample_metrics.json")
    
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    if model_coefficient_df is not None:
        mlflow.log_table(data=model_coefficient_df, artifact_file="feature_importances.json")

    create_and_log_visualizations(y_test, y_pred_test, trackingConfig)


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

        feature_labels = X_train.columns
        model_coefficient_df = get_feature_importances(search.best_estimator_['classifier'], feature_labels)

        # Set up MLflow tracking
        mlflow.set_tracking_uri(trackingConfig.tracking_uri)
        mlflow.set_experiment(trackingConfig.experiment_name)

        # Start a run for this model evaluation
        with mlflow.start_run(run_name=f"Model_{model.__class__.__name__}", nested=True) as run:
            mlflow.log_params(best_params)
            mlflow.set_tag("model", model.__class__.__name__)
            
            log_mlflow_metrics(metrics, best_params, search.best_estimator_['classifier'], X_test, y_test, y_pred_test, df_y_pred_train, df_y_pred_test, model_coefficient_df, trackingConfig)

        return metrics, best_params, run.info.run_id
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
        # Convert to pandas and pivot the data
        X_train = train_genotypes.toPandas().pivot(index='sample_id', columns='variant_id', values='GT_processed')
        X_test = test_genotypes.toPandas().pivot(index='sample_id', columns='variant_id', values='GT_processed')
        
        y_train = sample_processor.get_labels(train_sample_ids)
        y_test = sample_processor.get_labels(test_sample_ids)

        # Ensure that the indices align
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
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
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting bootstrapping of models.")

    results = []

    # Convert data to Spark DataFrame if it's a MatrixTable and cache it
    if isinstance(data, hl.MatrixTable):
        logger.info("Converting MatrixTable to Spark DataFrame and caching.")
        full_dataset = genotype_processor.to_spark_df(data).cache()
    elif isinstance(data, SparkDataFrame):
        logger.info("Caching input Spark DataFrame.")
        full_dataset = data.cache()
    else:
        raise TypeError("Input data must be either a Hail MatrixTable or a Spark DataFrame.")

    # Trigger an action to materialize the cache
    logger.info(f"Cached full dataset with {full_dataset.count()} rows.")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    mlflow.set_experiment(trackingConfig.experiment_name)

    for iteration in range(samplingConfig.bootstrap_iterations):
        with mlflow.start_run(run_name=f"Bootstrap_Iteration_{iteration}") as parent_run:
            logger.info(f"Bootstrapping iteration {iteration + 1}/{samplingConfig.bootstrap_iterations}")

            mlflow.log_param("iteration", iteration)
            mlflow.log_param("random_state", random_state + iteration)

            train_test_sample_ids = sample_processor.draw_train_test_split(
                test_size=samplingConfig.test_size,
                random_state=random_state + iteration
            )

            train_samples = train_test_sample_ids['train']['samples']
            test_samples = train_test_sample_ids['test']['samples']

            logger.info(f"Number of training samples: {len(train_samples)}")
            logger.info(f"Number of testing samples: {len(test_samples)}")

            # Use prepare_data function to get X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = prepare_data(
                sample_processor, 
                genotype_processor, 
                full_dataset, 
                train_samples, 
                test_samples
            )

            # Feature Selection Step
            logger.info("Performing feature selection using Bayesian Feature Selector.")
            max_attempts = 5
            attempt = 0
            selected_features = []

            while len(selected_features) == 0 and attempt < max_attempts:
                feature_selector = BayesianFeatureSelector(
                    num_iterations=1000,
                    lr=0.01,
                    confidence_level=0.95,
                    num_samples=1000,
                    batch_size=128,
                    verbose=True
                )
                feature_selector.fit(X_train, y_train)
                selected_features = feature_selector.selected_features_
                
                if len(selected_features) == 0:
                    logger.warning(f"No features selected on attempt {attempt + 1}. Retrying...")
                    attempt += 1

            if len(selected_features) == 0:
                logger.error("Failed to select features after maximum attempts. Proceeding with all features.")
                selected_features = X_train.columns.tolist()
            logger.info(f"Number of selected features: {len(selected_features)}")

            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Iterate over each model in the stack and evaluate
            for model, search_spaces in stack.items():
                model_name = model.__class__.__name__
                logger.info(f"Evaluating model: {model_name}")

                metrics, best_params, child_run_id = evaluate_model(
                    model=model,
                    search_spaces=search_spaces,
                    X_train=X_train_selected,
                    y_train=y_train,
                    X_test=X_test_selected,
                    y_test=y_test,
                    trackingConfig=trackingConfig,
                    n_iter=10
                )

                mlflow.log_metric(f"{model_name}_test_auc", metrics['aggregate']['test']['auc'])
                mlflow.log_metric(f"{model_name}_test_f1", metrics['aggregate']['test']['f1'])
                mlflow.log_param(f"{model_name}_run_id", child_run_id)

    # Unpersist the cached dataset
    full_dataset.unpersist()

    logger.info("Completed bootstrapping of models.")
    return pd.DataFrame()  # Return an empty DataFrame or collect results as needed

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def create_and_log_visualizations(y_true, y_pred, trackingConfig):
    """Create and log visualizations directly to MLflow."""
    
    def log_plot(plot_func, artifact_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_func(ax)
        mlflow.log_figure(fig, artifact_name)
        plt.close(fig)

    # ROC Curve
    def plot_roc(ax):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve\n{trackingConfig.name}')

    log_plot(plot_roc, "plots/roc_curve.png")

    # Precision-Recall Curve
    def plot_pr(ax):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve\n{trackingConfig.name}')

    log_plot(plot_pr, "plots/pr_curve.png")

    # Distribution of Predictions
    def plot_dist(ax):
        sns.histplot(y_pred, kde=True, ax=ax)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Predictions\n{trackingConfig.name}')

    log_plot(plot_dist, "plots/pred_distribution.png")

    # Confusion Matrix
    def plot_cm(ax):
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix\n{trackingConfig.name}')

    log_plot(plot_cm, "plots/confusion_matrix.png")