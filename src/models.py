import threading
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from pyro.contrib import autoname
from pyro.infer.autoguide import AutoMultivariateNormal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
from skopt.space import Real, Integer

from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore ConvergenceWarning from hyperparameter optimization
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class RadialBasisSVC(SVC):
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )

class SparseBayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, num_iterations=1000, lr=1e-3, verbose=False, unique_id=''):
        self.num_iterations = num_iterations
        self.lr = lr
        self.verbose = verbose
        self.is_fitted_ = False
        self.unique_id = unique_id
        
        # Set the device based on availability
        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def fit(self, X, y):
        # Ensure target is binary
        y_unique = np.unique(y)
        if len(y_unique) != 2:
            raise ValueError("SparseBayesianLogisticRegression only supports binary classification.")
        self.classes_ = y_unique

        # Convert data to tensors and move to device
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32, device=self.device)

        # Define optimizer with gradient clipping
        optimizer = ClippedAdam({'lr': self.lr, 'clip_norm': 5.0})

        # Use AutoMultivariateNormal
        with autoname.scope(prefix=self.unique_id):
            self.guide = AutoMultivariateNormal(self.model).to(self.device)

            # Set up the SVI object with TraceMeanField_ELBO
            svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())

        # Training loop
        progress_bar = tqdm(range(self.num_iterations), desc="Training", disable=not self.verbose)
        for i in progress_bar:
            loss = svi.step(X_tensor, y_tensor)
            if self.verbose:
                progress_bar.set_postfix({'Loss': f'{loss / len(X_tensor):.4f}'})

        # Extract parameters from the AutoGuide
        self.extract_params()

        self.is_fitted_ = True
        return self

    def extract_params(self):
        # Extract the posterior means from the guide
        # Account for the prefix in parameter names
        self.beta_loc_ = self.guide.locs[f"{self.unique_id}.beta"].detach().cpu().numpy()
        self.intercept_loc_ = self.guide.locs[f"{self.unique_id}.intercept"].detach().cpu().item()

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            beta_loc = torch.tensor(self.beta_loc_, device=self.device)
            intercept_loc = torch.tensor(self.intercept_loc_, device=self.device)

            logits = intercept_loc + X_tensor.matmul(beta_loc)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Return probability for both classes
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = proba.argmax(axis=1)
        return np.array([self.classes_[idx] for idx in class_indices])

    def get_params(self, deep=True):
        return {'num_iterations': self.num_iterations, 'lr': self.lr, 'verbose': self.verbose, 'unique_id': self.unique_id}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def model(self, X, y):
        with autoname.scope(prefix=self.unique_id):
            # Get dimensions
            N, D = X.shape

            # Global shrinkage parameter
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=1.0))

            with pyro.plate('features', D):
                # Local shrinkage parameters
                lam = pyro.sample('lam', dist.HalfCauchy(scale=1.0))
                # Auxiliary variables
                c2 = pyro.sample('c2', dist.InverseGamma(0.5, 0.5))
                # Scale for beta coefficients
                sigma = tau_0 * lam * torch.sqrt(c2)
                # Beta coefficients
                beta = pyro.sample('beta', dist.Normal(
                    loc=torch.zeros(D, device=self.device),
                    scale=sigma
                ))

            # Intercept
            intercept = pyro.sample('intercept', dist.Normal(0., 10.))

            # Compute logits
            logits = intercept + X.matmul(beta)

            # Observation likelihood
            with pyro.plate(f'data', N):
                pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

stack = {
    LogisticRegression(
        penalty="elasticnet",
        solver="saga",
    ): {"C": Real(1e-4, 1e4, prior="log-uniform"), "l1_ratio": Real(0, 1), "max_iter": Integer(100, 1000)},
    MultinomialNB(): {"alpha": Real(1e-4, 1e4, prior="log-uniform")},
    # SparseBayesianLogisticRegression(verbose=False): {"num_iterations": Integer(100, 1000), "lr": Real(1e-6, 1e-1, prior="log-uniform")},
    RadialBasisSVC(probability=True): {"C": Real(1e-4, 1e4, prior="log-uniform"), 'gamma': Real(1e-4, 1e-1, prior="log-uniform")},
    lgb.LGBMClassifier(): {"learning_rate": Real(1e-6, 1e-1, prior="log-uniform"), "max_depth": Integer(3, 10), "n_estimators": Integer(100, 1000)},
    KNeighborsClassifier(): {"n_neighbors": Integer(1, 10)},
    MLPClassifier(): {"hidden_layer_sizes": Integer(10, 100), "alpha": Real(1e-4, 1e-1, prior="log-uniform")},
}