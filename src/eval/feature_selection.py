import os
import time
import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks, peak_prominences, peak_widths
from joblib import Parallel, delayed
import torch
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pyro.contrib.autoname as autoname

@dataclass
class FeatureSelectionResult:
    selected_features: pd.Index
    num_variants: int
    total_variants: int
    credible_interval: float
    selected_credible_interval: Optional[float] = None

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
import mlflow
from sklearn.metrics import accuracy_score, f1_score

class BayesianFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, unique_id: str, num_iterations=1000, lr=1e-4, credible_interval=0.95, num_samples=1000,
                 batch_size=512, verbose=False, patience=800, covariance_type='independent',
                 validation_split=0.2, checkpoint_path="checkpoint.params", max_features=200):
        self.unique_id = unique_id
        self.logger = logging.getLogger(__name__)
        self.num_iterations = num_iterations
        self.lr = lr
        self.covariance_type = covariance_type
        self.credible_interval = credible_interval
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.validation_split = validation_split
        self.max_features = max_features
        self.checkpoint_path = f"{os.path.splitext(checkpoint_path)[0]}_{unique_id}.params"
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("Using CUDA (GPU) acceleration.")
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("Using CPU.")
        
        self.logger.info(f"Using device: {self.device}")

    def fit(self, X, y):
        scaler = MinMaxScaler()
        X_tensor = torch.tensor(scaler.fit_transform(X.values) if isinstance(X, pd.DataFrame) 
                                else scaler.fit_transform(X), 
                                dtype=torch.float64, device=self.device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, 
                                dtype=torch.float64, device=self.device)
        
        split_idx = int(X_tensor.size(0) * (1 - self.validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), 
                                batch_size=self.batch_size, shuffle=False)
        
        if isinstance(y, pd.Series):
            sample_ids = y.index.values
        elif isinstance(y, pd.DataFrame):
            # If y is a DataFrame, assume it's a single column
            sample_ids = y.index.values
        else:
            sample_ids = None
        
        if sample_ids is not None:
            val_sample_ids = sample_ids[split_idx:]
            # Initialize validation_results_ DataFrame
            self.validation_results_ = pd.DataFrame(index=val_sample_ids)
            self.validation_results_['label'] = y_val.cpu().numpy()
            self.validation_results_['sum_correct'] = 0
            self.validation_results_['count'] = 0
        else:
            self.validation_results_ = None

        def model(X, y=None):
            D = X.size(1)
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=torch.tensor(1.0, dtype=torch.float64, device=self.device)))
            lam = pyro.sample('lam', dist.HalfCauchy(scale=torch.tensor(1.0, dtype=torch.float64, device=self.device)).expand([D]).to_event(1))
            c2 = pyro.sample('c2', dist.InverseGamma(concentration=torch.tensor(1.0, dtype=torch.float64, device=self.device), 
                                                     rate=torch.tensor(1.0, dtype=torch.float64, device=self.device)).expand([D]).to_event(1))
            sigma = tau_0 * lam * torch.sqrt(c2)
            
            if self.covariance_type == 'independent':
                beta = pyro.sample('beta', dist.Normal(torch.zeros(D, dtype=torch.float64, device=self.device), sigma).to_event(1))
            elif self.covariance_type == 'multivariate':
                cov_matrix = torch.diag(sigma ** 2)
                beta = pyro.sample('beta', dist.MultivariateNormal(loc=torch.zeros(D, dtype=torch.float64, device=self.device), 
                                                                   covariance_matrix=cov_matrix))
            else:
                raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")
            
            intercept = pyro.sample('intercept', dist.Normal(torch.tensor(0., dtype=torch.float64, device=self.device), 
                                                             torch.tensor(10., dtype=torch.float64, device=self.device)))
            logits = intercept + X @ beta

            with pyro.plate('data', X.size(0)):
                pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

        guide = AutoNormal(model)

        optimizer = ClippedAdam({"lr": self.lr, "clip_norm": 1.0})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        best_val_loss = float("inf")
        patience_counter = 0

        def compute_proba(current_params, X, covariance_type='independent'):
            beta = current_params['AutoNormal.locs.beta']
            intercept = current_params['AutoNormal.locs.intercept']
            
            if covariance_type == 'independent':
                logits = intercept + torch.matmul(X, beta)
            elif covariance_type == 'multivariate':
                beta_mean = beta.mean(dim=0)
                logits = intercept + torch.matmul(X, beta_mean)
            else:
                raise ValueError(f"Unsupported covariance_type: {covariance_type}")
            
            proba = torch.sigmoid(logits)
            return proba.detach().cpu().numpy()

        if self.verbose:
            progress_bar = tqdm(range(1, self.num_iterations + 1), desc="Training", unit="iter")
        else:
            progress_bar = range(1, self.num_iterations + 1)

        start_time = time.time()
        for epoch in progress_bar:
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                loss = svi.step(X_batch, y_batch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / len(train_loader.dataset)

            val_loss = 0.0
            for X_val_batch, y_val_batch in val_loader:
                val_loss += svi.evaluate_loss(X_val_batch, y_val_batch)
            avg_val_loss = val_loss / len(val_loader)
            
            current_params = {k: v.clone().detach() for k, v in pyro.get_param_store().items()}
            y_val_pred_prob = compute_proba(current_params, X_val, covariance_type=self.covariance_type)
            y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_val.cpu().numpy(), y_val_pred)
            f1 = f1_score(y_val.cpu().numpy(), y_val_pred)
            
            self.log_metrics(epoch, avg_epoch_loss, avg_val_loss, accuracy, f1)

            # Update per-sample validation results
            if self.validation_results_ is not None:
                correct = (y_val_pred == y_val.cpu().numpy()).astype(int)
                # Assuming validation set order is preserved
                self.validation_results_.loc[val_sample_ids, 'sum_correct'] += correct
                self.validation_results_.loc[val_sample_ids, 'count'] += 1
            
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
                    f"Val Accuracy = {accuracy:.4f}, Val F1 = {f1:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                pyro.get_param_store().save(self.checkpoint_path)
                if self.verbose:
                    print(f"Epoch {epoch}: Validation loss improved to {best_val_loss:.4f}. Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        pyro.get_param_store().load(self.checkpoint_path)

        predictive = Predictive(model, guide=guide, num_samples=self.num_samples, return_sites=['beta'])
        posterior_samples = predictive(X_train, y_train)
        beta_samples = posterior_samples['beta'].detach().cpu().numpy()

        if beta_samples.ndim == 3:
            beta_samples = beta_samples.squeeze(1)
        elif beta_samples.ndim == 2 and beta_samples.shape[1] == 1:
            beta_samples = beta_samples.squeeze(1)

        # Aggregate per-sample metrics and log to MLflow
        if self.validation_results_ is not None:
            self.validation_results_['mean_accuracy'] = self.validation_results_['sum_correct'] / self.validation_results_['count']
            self.validation_results_['std_accuracy'] = np.sqrt(
                self.validation_results_['mean_accuracy'] * (1 - self.validation_results_['mean_accuracy']) / self.validation_results_['count']
            )
            self.validation_results_['draw_count'] = self.validation_results_['count']
            
            # Select relevant columns
            aggregated_results = self.validation_results_[['label', 'mean_accuracy', 'std_accuracy', 'draw_count']]
            aggregated_results.reset_index(inplace=True)
            aggregated_results.rename(columns={'index': 'sample_id'}, inplace=True)
            
            # Log the validation results
            mlflow.log_table(
                data=aggregated_results, 
                artifact_file="validation_aggregated_results.json"
            )

        # **New Integration: Using detect_data_threshold**
        # self.logger.info("Detecting data-driven threshold for feature selection.")
        # try:
        #     # Aggregate over posterior samples (e.g., compute the mean)
        #     mean_beta = np.mean(beta_samples, axis=0)

        #     # Pass the aggregated 1D array to detect_data_threshold
        #     threshold = detect_data_threshold(mean_beta, bandwidth='silverman', plot=False, logger=self.logger)
        #     self.selected_credible_interval = threshold  # Store the detected threshold
        #     self.logger.info(f"Detected threshold: {threshold:.4f}")
        #     mlflow.log_metric("selected_credible_interval", threshold)

        #     # Select features where absolute mean_beta exceeds the threshold
        #     selected = np.abs(mean_beta) >= threshold

        #     # Check if the number of selected features exceeds the maximum allowed
        #     if np.sum(selected) > self.max_features:
        #         raise ValueError(f"Number of selected features ({np.sum(selected)}) exceeds max_features ({self.max_features})")

        #     if isinstance(X, pd.DataFrame):
        #         self.selected_features_ = X.columns[selected]
        #     else:
        #         self.selected_features_ = np.arange(X.shape[1])[selected]
        # except Exception as e:
            # self.logger.error(f"Threshold detection failed: {e}")
            # Fallback to existing percentile-based selection
            # self.logger.warning("Falling back to percentile-based credible interval selection.")
        lower_bound = np.percentile(beta_samples, (1 - self.credible_interval) / 2 * 100, axis=0)
        upper_bound = np.percentile(beta_samples, (1 + self.credible_interval) / 2 * 100, axis=0)

        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound

        non_zero = (lower_bound > 0) | (upper_bound < 0)
        
        if isinstance(X, pd.DataFrame):
            self.selected_features_ = X.columns[non_zero]
        else:
            self.selected_features_ = np.arange(X.shape[1])[non_zero]
        
        self.selected_credible_interval = self.credible_interval  # Retain the original interval

        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_'):
            raise ValueError("The model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.selected_features_]

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, accuracy: float, f1: float):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        mlflow.log_metric("val_f1_score", f1, step=epoch)
        
        if epoch == 1:
            mlflow.log_param("unique_id", self.unique_id)
            mlflow.log_param("covariance_type", self.covariance_type)
            mlflow.log_param("num_iterations", self.num_iterations)
            mlflow.log_param("learning_rate", self.lr)
            mlflow.log_param("credible_interval", self.credible_interval)
            mlflow.log_param("num_samples", self.num_samples)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("patience", self.patience)
            mlflow.log_param("validation_split", self.validation_split)
            mlflow.log_param("checkpoint_path", self.checkpoint_path)


@lru_cache(maxsize=32)
def estimate_kde_sklearn(data_tuple: Tuple[float, ...], bandwidth: str = 'silverman') -> Tuple[np.ndarray, np.ndarray]:
    data = np.array(data_tuple)
    if bandwidth == 'silverman':
        bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5.)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(data[:, np.newaxis])
    x_grid = np.linspace(min(data), max(data), 500)[:, np.newaxis]
    log_density = kde.score_samples(x_grid)
    kde_values = np.exp(log_density)
    return x_grid.flatten(), kde_values

def compute_bic(n_components: int, data: np.ndarray) -> float:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(data)
    return gmm.bic(data)

def estimate_number_of_modes_parallel(data: np.ndarray, max_components: int = 20) -> int:
    data_reshaped = data.reshape(-1, 1)
    n_components_range = range(1, max_components + 1)
    bic_scores = Parallel(n_jobs=-1)(
        delayed(compute_bic)(n, data_reshaped) for n in n_components_range
    )
    best_n_components = n_components_range[np.argmin(bic_scores)]
    return best_n_components

def detect_peaks_adaptive(
    kde_values: np.ndarray,
    x_grid: np.ndarray,
    expected_num_peaks: int
) -> Tuple[np.ndarray, dict]:
    peaks, _ = find_peaks(kde_values)
    if len(peaks) == 0:
        return peaks, {}
    
    prominences = peak_prominences(kde_values, peaks)[0]
    widths = peak_widths(kde_values, peaks)[0]
    
    properties = {
        'prominences': prominences,
        'widths': widths
    }
    
    prominence_threshold = np.median(prominences)
    width_threshold = np.median(widths)
    
    significant_indices = (prominences >= prominence_threshold) & (widths >= width_threshold)
    significant_peaks = peaks[significant_indices]
    significant_properties = {key: val[significant_indices] for key, val in properties.items()}
    
    if len(significant_peaks) > expected_num_peaks:
        sorted_indices = np.argsort(-significant_properties['prominences'])
        significant_peaks = significant_peaks[sorted_indices[:expected_num_peaks]]
        for key in significant_properties:
            significant_properties[key] = significant_properties[key][sorted_indices[:expected_num_peaks]]
    
    return significant_peaks, significant_properties

def determine_threshold_vectorized(x_grid: np.ndarray, kde_values: np.ndarray, peaks: np.ndarray) -> Optional[float]:
    if len(peaks) >= 2:
        sorted_peaks = peaks[np.argsort(-kde_values[peaks])]
        first_peak, second_peak = sorted_peaks[:2]
        start, end = sorted([first_peak, second_peak])
        valley_region = kde_values[start:end]
        if valley_region.size == 0:
            return None
        valley_index = np.argmin(valley_region) + start
        return x_grid[valley_index]
    return None

def plot_kde(
    x_grid: np.ndarray,
    kde_values: np.ndarray,
    peaks: np.ndarray,
    threshold: Optional[float],
    data: np.ndarray
):
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, kde_values, label='KDE')
    plt.hist(data, bins=30, density=True, alpha=0.3, label='Histogram')
    plt.plot(x_grid[peaks], kde_values[peaks], "x", label='Detected Peaks')
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Detected Threshold: {threshold:.4f}')
    plt.xlabel('Data Values')
    plt.ylabel('Density')
    plt.title('Data Distribution with Detected Threshold')
    plt.legend()
    plt.show()

def detect_data_threshold(
    data: np.ndarray,
    bandwidth: str = 'silverman',
    plot: bool = False,
    logger: Optional[logging.Logger] = None,
    max_modes: int = 5
) -> float:
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    if len(data) < 2:
        raise ValueError("At least two data points are required.")

    if np.isclose(data.min(), data.max()):
        raise ValueError("Data values are nearly constant; threshold detection is not meaningful.")

    data = data.astype(np.float32)
    x_grid, kde_values = estimate_kde_sklearn(tuple(data), bandwidth=bandwidth)

    num_modes = estimate_number_of_modes_parallel(data, max_components=max_modes)
    logger.info(f"Estimated number of modes: {num_modes}")

    if num_modes >= 2:
        logger.info("Multimodal distribution detected; proceeding with adaptive peak detection.")
        peaks, properties = detect_peaks_adaptive(kde_values, x_grid, expected_num_peaks=num_modes)
        logger.info(f"Number of significant peaks detected: {len(peaks)}")

        threshold = determine_threshold_vectorized(x_grid, kde_values, peaks)
        if threshold is not None:
            logger.info(f"Detected threshold at value: {threshold:.4f}")
        else:
            raise ValueError("Unable to determine threshold from peaks.")
    else:
        raise ValueError("Unimodal distribution detected or insufficient modes; unable to determine threshold.")

    if plot:
        plot_kde(x_grid, kde_values, peaks, threshold, data)

    return threshold