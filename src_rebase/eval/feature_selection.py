import logging
from typing import Tuple, Optional
import os
# Enable fallback for CPU since cauchy is not supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time

class BayesianFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_iterations=1000, lr=1e-6, confidence_level=0.95, num_samples=1000,
                 batch_size=512, verbose=False, patience=800, fallback_percentile=95, max_modes=5,
                 validation_split=0.2, checkpoint_path="checkpoint.params"):
        self.num_iterations = num_iterations
        self.lr = lr
        self.confidence_level = confidence_level
        self.num_samples = num_samples  # Number of samples for posterior
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.fallback_percentile = fallback_percentile
        self.max_modes = max_modes
        self.validation_split = validation_split
        self.checkpoint_path = checkpoint_path

        # Set the device (GPU or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            if self.verbose:
                print("Using MPS (GPU) acceleration.")
        else:
            self.device = torch.device('cpu')
            if self.verbose:
                print("MPS not available. Using CPU.")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target variable y must be provided.")

        N, D = X.shape
        # Move data to the device
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32, device=self.device)

        # Split into training and validation sets
        split_idx = int(N * (1 - self.validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Create DataLoaders for mini-batch processing
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define the model and guide with batch_size as an argument
        def model(X_batch, y_batch):
            batch_size = X_batch.size(0)
            D = X_batch.size(1)

            # Sample the global and local variables
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=torch.tensor(1.0, device=self.device)))
            with pyro.plate('features', D):
                lam = pyro.sample('lam', dist.HalfCauchy(scale=torch.ones(D, device=self.device)))

            sigma = lam * tau_0
            beta = pyro.sample('beta', dist.Normal(loc=torch.zeros(D, device=self.device),
                                                scale=sigma).to_event(1))
            intercept = pyro.sample('intercept', dist.Normal(torch.tensor(0., device=self.device),
                                                            torch.tensor(10., device=self.device)))

            # Check if there is an extra dimension in beta (i.e., multiple posterior samples)
            if beta.dim() == 2:  # beta.shape: (num_samples, D)
                # Expand X_batch to match beta's sample dimension
                X_batch = X_batch.unsqueeze(0)  # X_batch shape: (1, batch_size, D)
                intercept = intercept.unsqueeze(-1).unsqueeze(-1)  # intercept shape: (num_samples, 1, 1)
                
                # Compute logits: shape (num_samples, batch_size)
                logits = intercept + (X_batch * beta.unsqueeze(1)).sum(-1)
                
                # Expand y_batch to match the sample dimension: shape (1, batch_size)
                y_batch = y_batch.unsqueeze(0)
                
                # Handle observation likelihood with multiple samples
                with pyro.plate('data', size=batch_size):
                    pyro.sample('obs', dist.Bernoulli(logits=logits).to_event(1), obs=y_batch)
            else:
                # Single sample case: beta.shape: (D,)
                logits = intercept + X_batch.matmul(beta)  # shape: (batch_size,)

                # Handle observation likelihood for single sample
                with pyro.plate('data', size=batch_size):
                    pyro.sample('obs', dist.Bernoulli(logits=logits).to_event(1), obs=y_batch)

        def guide(X_batch, y_batch):
            batch_size = X_batch.size(0)
            tau_loc = pyro.param('tau_loc', torch.tensor(0.0, device=self.device))
            tau_scale = pyro.param('tau_scale', torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)
            tau_0 = pyro.sample('tau_0', dist.LogNormal(tau_loc, tau_scale))
            
            lam_loc = pyro.param('lam_loc', torch.zeros(D, device=self.device))
            lam_scale = pyro.param('lam_scale', torch.ones(D, device=self.device), constraint=dist.constraints.positive)
            with pyro.plate('features', D):
                lam = pyro.sample('lam', dist.LogNormal(lam_loc, lam_scale))
            
            beta_loc = pyro.param('beta_loc', torch.zeros(D, device=self.device))
            beta_scale = pyro.param('beta_scale', torch.ones(D, device=self.device), constraint=dist.constraints.positive)
            beta = pyro.sample('beta', dist.Normal(beta_loc, beta_scale).to_event(1))
            
            intercept_loc = pyro.param('intercept_loc', torch.tensor(0.0, device=self.device))
            intercept_scale = pyro.param('intercept_scale', torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)
            intercept = pyro.sample('intercept', dist.Normal(intercept_loc, intercept_scale))

        # Use ClippedAdam optimizer for stability
        optimizer = AdamW({'lr': self.lr})

        # Set up the inference algorithm
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # Clear the parameter store
        pyro.clear_param_store()

        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop with tqdm progress bar
        if self.verbose:
            progress_bar = tqdm(range(1, self.num_iterations + 1), desc="Training", unit="iter")
        else:
            progress_bar = range(1, self.num_iterations + 1)

        start_time = time.time()
        for i in progress_bar:
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                loss = svi.step(X_batch, y_batch)
                epoch_loss += loss

            # Validation
            total_val_loss = 0.0
            for X_val_batch, y_val_batch in val_loader:
                val_loss = svi.evaluate_loss(X_val_batch, y_val_batch)
                total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(val_loader)

            if self.verbose:
                if i % 10 == 0 or i == 1:
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / i
                    remaining_iters = self.num_iterations - i
                    estimated_remaining_time = remaining_iters * avg_time_per_iter
                    progress_bar.set_postfix({
                        'train_loss': f"{epoch_loss:.4f}",
                        'val_loss': f"{avg_val_loss:.4f}",
                        'ETA': f"{estimated_remaining_time / 60:.2f} min"
                    })
                else:
                    progress_bar.set_postfix({'train_loss': f"{epoch_loss:.4f}"})

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save model checkpoint
                pyro.get_param_store().save(self.checkpoint_path)
                if self.verbose:
                    print(f"Iteration {i}: Validation loss improved to {best_val_loss:.4f}. Checkpoint saved.")
            else:
                patience_counter += 1
                if self.verbose:
                    print(f"Iteration {i}: Validation loss did not improve. Patience counter: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        # Load the best parameters
        pyro.get_param_store().load(self.checkpoint_path)

        # Use Predictive for posterior sampling
        predictive = Predictive(model, guide=guide, num_samples=self.num_samples,
                                return_sites=["beta", "intercept"])
        posterior_samples = predictive(X_train, y_train)

        # Extract beta samples from the posterior
        beta_samples = posterior_samples['beta'].detach().cpu().numpy()
        # Remove the extra dimension from parallel sampling
        beta_samples = np.squeeze(beta_samples, axis=1)

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
            self.selected_features_ = np.arange(D)[non_zero]

        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_'):
            raise ValueError("The model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, self.selected_features_]

def estimate_kde(
    data: np.ndarray,
    bandwidth: str = 'silverman'
) -> Tuple[np.ndarray, np.ndarray]:
    kde = gaussian_kde(data, bw_method=bandwidth)
    x_grid = np.linspace(min(data), max(data), 1000)
    kde_values = kde.evaluate(x_grid)
    return x_grid, kde_values

def estimate_number_of_modes(
    data: np.ndarray,
    max_components: int = 100
) -> int:
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, max_components + 1)
    data_reshaped = data.reshape(-1, 1)
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data_reshaped)
        bic_score = gmm.bic(data_reshaped)
        bic.append(bic_score)
        if bic_score < lowest_bic:
            lowest_bic = bic_score
            best_n_components = n_components
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

def determine_threshold(
    x_grid: np.ndarray,
    kde_values: np.ndarray,
    peaks: np.ndarray
) -> Optional[float]:
    if len(peaks) >= 2:
        peak_heights = kde_values[peaks]
        sorted_peaks = peaks[np.argsort(-peak_heights)]
        first_peak = sorted_peaks[0]
        second_peak = sorted_peaks[1]
        start, end = sorted([first_peak, second_peak])
        valley_region = kde_values[start:end]
        if len(valley_region) == 0:
            return None
        valley_index = np.argmin(valley_region) + start
        threshold = x_grid[valley_index]
        return threshold
    else:
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
    max_modes: int = 5
) -> float:
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(data) < 2:
        raise ValueError("At least two data points are required.")

    if np.isclose(min(data), max(data)):
        raise ValueError("Data values are nearly constant; threshold detection is not meaningful.")

    x_grid, kde_values = estimate_kde(data, bandwidth=bandwidth)

    num_modes = estimate_number_of_modes(data, max_components=max_modes)
    logger.info(f"Estimated number of modes: {num_modes}")

    if num_modes >= 2:
        logger.info("Multimodal distribution detected; proceeding with adaptive peak detection.")
        peaks, properties = detect_peaks_adaptive(kde_values, x_grid, expected_num_peaks=num_modes)
        logger.info(f"Number of significant peaks detected: {len(peaks)}")

        threshold = determine_threshold(x_grid, kde_values, peaks)
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
        peaks, _ = detect_peaks_adaptive(kde_values, x_grid, expected_num_peaks=num_modes)
        plot_kde(x_grid, kde_values, peaks, threshold, data)

    return threshold