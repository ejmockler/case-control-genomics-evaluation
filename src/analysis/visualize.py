import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
import mlflow
from typing import List, Dict, Any
from config import TrackingConfig

def aggregate_and_plot_summary(trackingConfig: TrackingConfig, output_dir: str = "summary_plots"):
    """
    Aggregate ROC metrics across all runs for each model and generate summary visualizations.
    
    Args:
        trackingConfig (TrackingConfig): Configuration for tracking.
        output_dir (str, optional): Directory to save summary plots. Defaults to "summary_plots".
    """
 
    for model, runs in model_runs.items():
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Compute mean and std of tpr
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Plot mean ROC with std deviation
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=4, alpha=0.8)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                        label=r"$\pm$ 1 std. dev.")
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Mean ROC Curve for {model}')
        ax.legend(loc="lower right")
        plt.tight_layout()
        
        # Save and log the summary ROC plot
        summary_plot_path = os.path.join(output_dir, f"{model}_roc_summary.svg")
        plt.savefig(summary_plot_path)
        mlflow.log_artifact(summary_plot_path)
        plt.close(fig)
        
        # Generate and log normalized confusion matrix
        normalized_cm = generate_normalized_confusion_matrix(runs, model)
        if normalized_cm is not None:
            cm_plot_path = os.path.join(output_dir, f"sanitize_mlflow_name(model)_confusion_matrix_summary.svg")
            plot_confusion_matrix(normalized_cm, model, cm_plot_path)
            mlflow.log_artifact(cm_plot_path)
    
def generate_normalized_confusion_matrix(runs, model_name):
    """
    Generate a normalized confusion matrix aggregated across runs for a specific model.
    
    Args:
        runs (List[Run]): List of MLflow runs for the model.
        model_name (str): Name of the model.
    
    Returns:
        np.ndarray: Normalized confusion matrix.
    """
    client = MlflowClient()
    cm_total = np.array([[0, 0], [0, 0]])
    for run in runs:
        artifact = f"{model_name}_confusion_matrix.json"
        try:
            local_path = client.download_artifacts(run.info.run_id, artifact)
            cm_data = pd.read_json(local_path).iloc[0]
            tn = cm_data.get('true_negatives', 0)
            fp = cm_data.get('false_positives', 0)
            fn = cm_data.get('false_negatives', 0)
            tp = cm_data.get('true_positives', 0)
            cm = np.array([[tn, fp], [fn, tp]])
            cm_total += cm
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to retrieve confusion matrix for run {run.info.run_id}: {e}")
            continue
    
    if cm_total.sum() == 0:
        logging.getLogger(__name__).warning(f"No confusion matrix data available for model '{model_name}'.")
        return None
    
    # Normalize the confusion matrix
    cm_normalized = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]
    return cm_normalized

def plot_confusion_matrix(cm, model_name, save_path):
    """
    Plot and save a normalized confusion matrix.
    
    Args:
        cm (np.ndarray): Normalized confusion matrix.
        model_name (str): Name of the model.
        save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Normalized Confusion Matrix for {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



