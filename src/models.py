import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from skopt.space import Integer, Real

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
    def __init__(self, num_iterations=1000, lr=0.01, verbose=False):
        self.num_iterations = num_iterations
        self.lr = lr
        self.verbose = verbose
        self.is_fitted_ = False

    def fit(self, X, y):
        # Store unique classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SparseBayesianLogisticRegression only supports binary classification.")

        N, D = X.shape
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)

        # Define the model and guide
        # Similar to the BayesianFeatureSelector but focusing on prediction

        # Use ClippedAdam optimizer for stability
        optimizer = ClippedAdam({'lr': self.lr})

        # Set up the inference algorithm
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Clear the parameter store
        pyro.clear_param_store()

        # Training loop
        for i in range(self.num_iterations):
            loss = svi.step(X_tensor, y_tensor)
            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss}')

        # Store the learned parameters
        self.beta_loc_ = pyro.param('beta_loc').detach().numpy()
        self.beta_scale_ = pyro.param('beta_scale').detach().numpy()
        self.intercept_loc_ = pyro.param('intercept_loc').item()
        self.intercept_scale_ = pyro.param('intercept_scale').item()
        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
        beta_loc = torch.tensor(self.beta_loc_, dtype=torch.float32)
        intercept_loc = torch.tensor(self.intercept_loc_, dtype=torch.float32)
        logits = intercept_loc + X_tensor.matmul(beta_loc)
        probs = torch.sigmoid(logits).detach().numpy()
        return np.vstack([1 - probs, probs]).T  # shape (n_samples, 2)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def get_params(self, deep=True):
        return {'num_iterations': self.num_iterations, 'lr': self.lr, 'verbose': self.verbose}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def model(self, X, y):
        N, D = X.shape
        tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=1.0))
        with pyro.plate('features', D):
            lam = pyro.sample('lam', dist.HalfCauchy(scale=torch.ones(D)))
        sigma = lam * tau_0
        beta = pyro.sample('beta', dist.Normal(loc=torch.zeros(D), scale=sigma).to_event(1))
        intercept = pyro.sample('intercept', dist.Normal(0., 10.))
        logits = intercept + X.matmul(beta)
        with pyro.plate('data', N):
            pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

    def guide(self, X, y):
        N, D = X.shape
        # Global variables
        tau_loc = pyro.param('tau_loc', torch.tensor(0.0))
        tau_scale = pyro.param('tau_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
        tau_0 = pyro.sample('tau_0', dist.LogNormal(tau_loc, tau_scale))

        # Feature-specific variables
        lam_loc = pyro.param('lam_loc', torch.zeros(D))
        lam_scale = pyro.param('lam_scale', torch.ones(D), constraint=dist.constraints.positive)
        with pyro.plate('features', D):
            lam = pyro.sample('lam', dist.LogNormal(lam_loc, lam_scale))

        # Beta coefficients
        beta_loc = pyro.param('beta_loc', torch.zeros(D))
        beta_scale = pyro.param('beta_scale', torch.ones(D), constraint=dist.constraints.positive)
        beta = pyro.sample('beta', dist.Normal(beta_loc, beta_scale).to_event(1))

        # Intercept
        intercept_loc = pyro.param('intercept_loc', torch.tensor(0.0))
        intercept_scale = pyro.param('intercept_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
        intercept = pyro.sample('intercept', dist.Normal(intercept_loc, intercept_scale))

stack = {
    LogisticRegression(
        penalty="l1",
        solver="saga",
    ): {"C": Real(1e-14, 1e-7, prior="log-uniform")},
    BernoulliNB(): {"alpha": Real(1e-6, 1, prior="log-uniform")},
    SparseBayesianLogisticRegression(): {"num_iterations": Integer(100, 1000), "lr": Real(1e-6, 1e-1, prior="log-uniform")},
    RadialBasisSVC(probability=True): {"C": Real(1e-14, 1e-7, prior="log-uniform")},
    LGBMClassifier(): {"learning_rate": Real(1e-3, 1e-1, prior="log-uniform"), "max_depth": Integer(3, 10), "n_estimators": Integer(100, 1000)},
}
