import os
import time
# Enable fallback for CPU since Cauchy is not supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import find_peaks, peak_prominences, peak_widths
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm


@dataclass
class FeatureSelectionResult:
    selected_features: pd.Index
    num_variants: int
    total_variants: int
    confidence_level: float

        
class BayesianFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Bayesian Feature Selector using Sparse Bayesian Learning (SBL) with Pyro.
    Each instance must have a unique identifier to prevent sample site name collisions.
    """
    def __init__(self, unique_id: str, num_iterations=1000, lr=1e-4, confidence_level=0.95, num_samples=1000,
                 batch_size=512, verbose=False, patience=800,
                 validation_split=0.2, checkpoint_path="checkpoint.params"):
        """
        Initializes the BayesianFeatureSelector with the given hyperparameters.
        
        Parameters:
        - num_iterations: Number of training iterations.
        - lr: Learning rate for the optimizer.
        - confidence_level: Confidence level for credible intervals.
        - num_samples: Number of posterior samples to draw.
        - batch_size: Size of mini-batches for training.
        - verbose: If True, prints training progress.
        - patience: Number of epochs with no improvement before early stopping.
        - validation_split: Fraction of data to use for validation.
        - checkpoint_path: Path to save model checkpoints.
        - unique_id: A unique string identifier for the model instance.
        """
        self.unique_id = unique_id  # Store the unique identifier
        self.logger = logging.getLogger(__name__)
        self.num_iterations = num_iterations
        self.lr = lr
        self.confidence_level = confidence_level
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.validation_split = validation_split
        # Append unique_id to checkpoint_path to ensure uniqueness
        self.checkpoint_path = f"{os.path.splitext(checkpoint_path)[0]}_{unique_id}.params"
        
        # Set the device based on availability
        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     if self.verbose:
        #         print("Using MPS (GPU) acceleration with CPU fallback.")
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
        """
        Fits the Bayesian Feature Selector to the data.

        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame).
        - y: Target vector (numpy array or pandas Series).

        Returns:
        - self: Fitted estimator.
        """
        # Convert input data to PyTorch tensors and move to the appropriate device
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, 
                                dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, 
                                dtype=torch.float32, device=self.device)

       
        # Split data into training and validation sets
        split_idx = int(X_tensor.size(0) * (1 - self.validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Create DataLoaders for mini-batch processing
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), 
                                batch_size=self.batch_size, shuffle=False)

        # Define the Bayesian model with unique sample site names and improved parameterization
        def model(X_batch, y_batch):
            B, D = X_batch.size(0), X_batch.size(1)
            
            with autoname.scope(prefix=self.unique_id):
                # Global shrinkage parameter
                tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=1.0))
                
                # Local shrinkage parameters
                with pyro.plate(f'features', D):
                    lam = pyro.sample('lam', dist.HalfCauchy(scale=1.0))
                
                # Auxiliary variables for improved heavy-tailed modeling
                with pyro.plate(f'features_c2', D):
                    c2 = pyro.sample('c2', dist.InverseGamma(0.5, 0.5))
                
                # Scale for beta coefficients
                sigma = tau_0 * lam * torch.sqrt(c2)
                
                # Sample beta coefficients with sparsity
                beta = pyro.sample('beta', dist.Normal(
                    loc=torch.zeros(D, device=self.device),
                    scale=sigma
                ).to_event(1))
                
                # Sample intercept
                intercept = pyro.sample('intercept', dist.Normal(0., 10.))
                
                # Compute logits
                if len(beta.shape) == 1:
                    logits = intercept + X_batch.matmul(beta)
                else:
                    logits = intercept[:, None] + torch.matmul(beta, X_batch.T)
                
                # Observation likelihood
                with pyro.plate(f'data', B):
                    pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y_batch)

        # Use AutoMultivariateNormal guide
        guide = AutoLowRankMultivariateNormal(model)

        # Initialize the optimizer with adjusted settings
        optimizer = ClippedAdam({"lr": self.lr, "clip_norm": 5.0})
        
        # Set up the SVI object with TraceMeanField_ELBO loss
        svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
        
        # Clear previous parameter store
        pyro.clear_param_store()

        # Initialize early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop with optional verbosity
        if self.verbose:
            progress_bar = tqdm(range(1, self.num_iterations + 1), desc="Training", unit="iter")
        else:
            progress_bar = range(1, self.num_iterations + 1)

        start_time = time.time()
        for epoch in progress_bar:
            epoch_loss = 0.0
            # Training: iterate over batches and update parameters
            for X_batch, y_batch in train_loader:
                loss = svi.step(X_batch, y_batch)
                epoch_loss += loss

            # Validation: compute average validation loss
            val_loss = 0.0
            for X_val_batch, y_val_batch in val_loader:
                val_loss += svi.evaluate_loss(X_val_batch, y_val_batch)
            avg_val_loss = val_loss / len(val_loader)

            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader.dataset)

            # Log metrics using MLflow
            self.log_metrics(epoch, avg_epoch_loss, avg_val_loss)

            # Update progress bar if verbose
            if self.verbose:
                if epoch % 10 == 0 or epoch == 1:
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / epoch
                    remaining_iters = self.num_iterations - epoch
                    estimated_remaining_time = remaining_iters * avg_time_per_iter
                    progress_bar.set_postfix({
                        'train_loss': f"{avg_epoch_loss:.4f}",
                        'val_loss': f"{avg_val_loss:.4f}",
                        'ETA': f"{estimated_remaining_time / 60:.2f} min"
                    })
                else:
                    progress_bar.set_postfix({'train_loss': f"{avg_epoch_loss:.4f}"})

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save model checkpoint
                pyro.get_param_store().save(self.checkpoint_path)
                if self.verbose:
                    print(f"Epoch {epoch}: Validation loss improved to {best_val_loss:.4f}. Checkpoint saved.")
            else:
                patience_counter += 1
                if self.verbose:
                    print(f"Epoch {epoch}: Validation loss did not improve. Patience counter: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        # Load the best parameters
        pyro.get_param_store().load(self.checkpoint_path)

        # Extract beta samples from the posterior using the prefixed name
        beta_key = f'{self.unique_id}/beta'

        # Use Predictive for posterior sampling
        predictive = Predictive(model, guide=guide, num_samples=self.num_samples,
                                return_sites=[beta_key], parallel=True)
        posterior_samples = predictive(X_train, y_train)
        beta_samples = posterior_samples[beta_key].detach().cpu().numpy()

        # Remove the extra dimension from parallel sampling if necessary
        if beta_samples.ndim == 3:
            # Shape: [num_samples, 1, D] -> [num_samples, D]
            beta_samples = beta_samples.squeeze(1)
        elif beta_samples.ndim == 2 and beta_samples.shape[1] == 1:
            # Shape: [num_samples, 1] -> [num_samples]
            beta_samples = beta_samples.squeeze(1)

        # Compute credible intervals
        lower_bound = np.percentile(beta_samples, (1 - self.confidence_level) / 2 * 100, axis=0)
        upper_bound = np.percentile(beta_samples, (1 + self.confidence_level) / 2 * 100, axis=0)

        # Store credible intervals
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound

        # Select features where the credible interval does not include zero
        non_zero = (lower_bound > 0) | (upper_bound < 0)
        
        if isinstance(X, pd.DataFrame):
            self.selected_features_ = X.columns[non_zero]
        else:
            self.selected_features_ = np.arange(X.shape[1])[non_zero]

        return self

    def transform(self, X):
        """
        Transforms the input data by selecting the chosen features.

        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame).

        Returns:
        - X_transformed: Transformed feature matrix with selected features.
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("The model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.selected_features_]

    def log_metrics(self, epoch, train_loss, val_loss):
        """
        Logs training and validation losses to MLflow.

        Parameters:
        - epoch: Current epoch number.
        - train_loss: Training loss for the epoch.
        - val_loss: Validation loss for the epoch.
        """
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

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
    fallback_percentile: float = 95,
    max_modes: int = 20
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
            logger.warning("Unable to determine threshold from peaks; falling back to percentile-based threshold.")
            threshold = np.percentile(data, fallback_percentile)
            logger.info(f"Using {fallback_percentile}th percentile as threshold: {threshold:.4f}")
    else:
        logger.info("Unimodal distribution detected or insufficient modes; using percentile-based threshold.")
        threshold = np.percentile(data, fallback_percentile)
        logger.info(f"Using {fallback_percentile}th percentile as threshold: {threshold:.4f}")

    if plot:
        plot_kde(x_grid, kde_values, peaks, threshold, data)

    return threshold