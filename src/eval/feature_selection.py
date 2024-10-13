from functools import lru_cache
import os
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO, Predictive
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
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
    credible_interval: float

import os
import logging
import time
from typing import Union

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import mlflow
from numpyro import distributions as dist
import numpyro
from numpyro.infer import SVI, TraceMeanField_ELBO, Predictive
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal

class BayesianFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Bayesian Feature Selector using Sparse Bayesian Learning (SBL) with NumPyro.
    Each instance must have a unique identifier to prevent sample site name collisions.
    Supports both independent Normal and MultivariateNormal distributions for feature weights.
    """
    def __init__(self, unique_id: str, covariance_type: str = 'independent', num_iterations=1000, lr=1e-4,
                 credible_interval=0.95, num_samples=1000, batch_size=512, verbose=False,
                 patience=800, validation_split=0.2, checkpoint_path="checkpoint.npz"):
        """
        Initializes the BayesianFeatureSelector with the given hyperparameters.
        
        Parameters:
        - unique_id: A unique string identifier for the model instance.
        - covariance_type: Type of covariance for feature weights ('independent' or 'multivariate').
        - num_iterations: Number of training iterations.
        - lr: Learning rate for the optimizer.
        - credible_interval: Confidence level for credible intervals.
        - num_samples: Number of posterior samples to draw.
        - batch_size: Size of mini-batches for training.
        - verbose: If True, prints training progress.
        - patience: Number of epochs with no improvement before early stopping.
        - validation_split: Fraction of data to use for validation.
        - checkpoint_path: Path to save model checkpoints.
        """
        self.unique_id = unique_id  # Store the unique identifier
        self.logger = logging.getLogger(__name__)
        self.num_iterations = num_iterations
        self.lr = lr
        self.credible_interval = credible_interval
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.validation_split = validation_split
        self.checkpoint_path = f"{os.path.splitext(checkpoint_path)[0]}_{unique_id}.npz"
        
        # Validate covariance_type
        if covariance_type not in ['independent', 'multivariate']:
            raise ValueError("covariance_type must be either 'independent' or 'multivariate'")
        self.covariance_type = covariance_type
        
        self.logger.info("NumPyro uses JAX's device backend automatically.")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, pd.DataFrame]):
        """
        Fits the Bayesian Feature Selector to the data.

        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame).
        - y: Target vector (numpy array or pandas Series).

        Returns:
        - self: Fitted estimator.
        """
        # Convert input data to JAX numpy arrays
        if isinstance(X, pd.DataFrame):
            X_np = jnp.array(X.values)
            feature_names = X.columns
        else:
            X_np = jnp.array(X)
            feature_names = np.arange(X_np.shape[1])
        
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_np = jnp.array(y.values).flatten()
        else:
            y_np = jnp.array(y).flatten()

        def model(X, y=None):
            """
            Defines the probabilistic model for Bayesian feature selection.
            
            Parameters:
            - X: Feature matrix.
            - y: Target vector.
            """
            D = X.shape[1]
            tau_0 = numpyro.sample('tau_0', dist.HalfCauchy(scale=1.0))
            lam = numpyro.sample('lam', dist.HalfCauchy(scale=1.0).expand([D]).to_event(1))
            c2 = numpyro.sample('c2', dist.InverseGamma(concentration=0.5, rate=0.5).expand([D]).to_event(1))
            sigma = tau_0 * lam * jnp.sqrt(c2)
            
            if self.covariance_type == 'independent':
                # Independent Normal distributions for each beta
                beta = numpyro.sample('beta', dist.Normal(jnp.zeros(D), sigma).to_event(1))
            elif self.covariance_type == 'multivariate':
                # Multivariate Normal with diagonal covariance
                # Alternatively, for full covariance, you can define a covariance matrix.
                # Here, we'll use a diagonal covariance for simplicity.
                cov_matrix = jnp.diag(sigma ** 2)
                beta = numpyro.sample('beta', dist.MultivariateNormal(loc=jnp.zeros(D), covariance_matrix=cov_matrix))
            else:
                raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")
            
            intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
            logits = intercept + X @ beta
            with numpyro.plate('data', X.shape[0]):
                numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
        
        # Split data into training and validation sets
        split_idx = int(X_np.shape[0] * (1 - self.validation_split))
        X_train, X_val = X_np[:split_idx], X_np[split_idx:]
        y_train, y_val = y_np[:split_idx], y_np[split_idx:]
        
        # Manually create batches for training and validation
        train_batches = self._create_batches(X_train, y_train, self.batch_size, shuffle=True)
        val_batches = self._create_batches(X_val, y_val, self.batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = Adam(self.lr)
        
        # Initialize the guide (variational distribution)
        # For multivariate covariance, AutoLowRankMultivariateNormal is suitable.
        # For independent normals, a different guide might be more efficient
        if self.covariance_type == 'multivariate':
            guide = AutoLowRankMultivariateNormal(model)
        else:
            guide = AutoNormal(model)

        # Initialize SVI state with a unique random key for each instance
        seed = int.from_bytes(os.urandom(4), 'little')
        
        # Initialize SVI
        svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
        
        # Initialize SVI state with the first batch
        initial_batch_X, initial_batch_y = train_batches[0]
        svi_state = svi.init(rng_key=jax.random.PRNGKey(seed), X=initial_batch_X, y=initial_batch_y)
        
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
            for batch_X, batch_y in train_batches:
                svi_state, loss = svi.update(svi_state, X=batch_X, y=batch_y)
                epoch_loss += jax.device_get(loss)
            
            avg_train_loss = epoch_loss / split_idx
            
            # Validation loss
            val_loss = 0.0
            for batch_X_val, batch_y_val in val_batches:
                batch_val_loss = svi.evaluate(svi_state, X=batch_X_val, y=batch_y_val)
                val_loss += jax.device_get(batch_val_loss)
            avg_val_loss = val_loss / (X_val.shape[0] or 1)

            # Log metrics using MLflow
            self.log_metrics(epoch, avg_train_loss, avg_val_loss)
            
            # Update progress bar if verbose
            if self.verbose:
                if epoch % 10 == 0 or epoch == 1:
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / epoch
                    remaining_iters = self.num_iterations - epoch
                    estimated_remaining_time = remaining_iters * avg_time_per_iter
                    progress_bar.set_postfix({
                        'train_loss': f"{avg_train_loss:.4f}",
                        'val_loss': f"{avg_val_loss:.4f}",
                        'ETA': f"{estimated_remaining_time / 60:.2f} min"
                    })
                else:
                    progress_bar.set_postfix({'train_loss': f"{avg_train_loss:.4f}"})
            
            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the optimized parameters
                optimized_params = svi.get_params(svi_state)
                np.savez(self.checkpoint_path, **{k: jax.device_get(v) for k, v in optimized_params.items()})
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
        if os.path.exists(self.checkpoint_path):
            checkpoint = np.load(self.checkpoint_path)
            loaded_params = {k: jnp.array(v) for k, v in checkpoint.items()}
        else:
            loaded_params = svi.get_params(svi_state)
        
        # Take posterior samples
        predictive = Predictive(model, params=loaded_params, num_samples=self.num_samples)
        posterior_samples = predictive(jax.random.PRNGKey(seed + 1), X=X_train, y=y_train)
        beta_samples = posterior_samples['beta']
        
        # Convert JAX arrays to NumPy for percentile calculation
        beta_samples_np = np.array(beta_samples)
        
        # Compute credible intervals
        lower_bound = np.percentile(beta_samples_np, (1 - self.credible_interval) / 2 * 100, axis=0)
        upper_bound = np.percentile(beta_samples_np, (1 + self.credible_interval) / 2 * 100, axis=0)
        
        # Select features where the credible interval does not include zero
        non_zero = (lower_bound > 0) | (upper_bound < 0)
        
        if isinstance(X, pd.DataFrame):
            self.selected_features_ = feature_names[non_zero]
        else:
            self.selected_features_ = np.arange(X_np.shape[1])[non_zero]
        
        # Store credible intervals
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
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
            feature_indices = self.selected_features_
            return X[:, feature_indices]
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """
        Logs training and validation losses to MLflow.

        Parameters:
        - epoch: Current epoch number.
        - train_loss: Training loss for the epoch.
        - val_loss: Validation loss for the epoch.
        """
        mlflow.log_metric("train_loss", float(train_loss), step=epoch)
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)
    
    @staticmethod
    def _create_batches(X: jnp.ndarray, y: jnp.ndarray, batch_size: int, shuffle: bool = False):
        """
        Creates batches of data manually compatible with JAX arrays.

        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        - batch_size: Size of each batch.
        - shuffle: Whether to shuffle the data before batching.

        Returns:
        - List of tuples containing batched X and y.
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # If the last batch is smaller, pad it to match batch_size
            current_batch_size = batch_X.shape[0]
            if current_batch_size < batch_size:
                pad_size = batch_size - current_batch_size
                # Randomly select samples to pad
                pad_indices = np.random.choice(batch_X.shape[0], size=pad_size, replace=True)
                pad_X = batch_X[pad_indices]
                pad_y = batch_y[pad_indices]
                batch_X = jnp.concatenate([batch_X, pad_X], axis=0)
                batch_y = jnp.concatenate([batch_y, pad_y], axis=0)
            
            batches.append((batch_X, batch_y))
        return batches

    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        """
        Loads model parameters from a checkpoint.

        Parameters:
        - checkpoint_path: Path to the checkpoint file.

        Returns:
        - params: Dictionary of model parameters.
        """
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        params = {k: jnp.array(v) for k, v in checkpoint.items()}
        return params

    
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