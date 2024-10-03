import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn_rvm import EMRVC
import torch
from xgboost import XGBClassifier

from skopt.space import Categorical, Integer, Real
import pandas as pd


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

        # Define the model
        def model(X, y):
            tau_0 = pyro.sample('tau_0', dist.HalfCauchy(scale=1.0))
            with pyro.plate('features', D):
                lam = pyro.sample('lam', dist.HalfCauchy(scale=torch.ones(1)))
            sigma = lam * tau_0
            beta = pyro.sample('beta', dist.Normal(loc=torch.zeros(D), scale=sigma).to_event(1))
            intercept = pyro.sample('intercept', dist.Normal(0., 10.))
            logits = intercept + X.matmul(beta)
            with pyro.plate('data', N):
                pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

        # Define the guide (variational distribution)
        def guide(X, y):
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

        # Use ClippedAdam optimizer for stability
        optimizer = ClippedAdam({'lr': self.lr})

        # Set up the inference algorithm
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

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


class RadialBasisSVC(SVC):
    def __init__(self, **kwargs):
        super().__init__(kernel="rbf", **kwargs)

class LinearSVC(SVC):
    def __init__(self, **kwargs):
        super().__init__(kernel="linear", **kwargs)

stack = {
    # LinearSVC(probability=True, kernel="linear"): {
    #     "tol": Real(1e-6, 1e-3, prior="log-uniform"),
    #     "C": Real(1e-3, 10, prior="log-uniform"),
    # },
    # RadialBasisSVC(probability=True, kernel="rbf"): {
    #     "tol": Real(1e-6, 1e-3, prior="log-uniform"),
    #     "C": Real(1e-3, 10, prior="log-uniform"),
    #     "gamma": Categorical(["scale", "auto"]),
    # },
    # LogisticRegression(penalty="l1", solver="saga", l1_ratio=1): {
    #     "C": Real(1e-6, 1, prior="log-uniform"),
    #     #   "l1_ratio": Real(1e-6, 1, prior="log-uniform"),
    # },
    LogisticRegression(
        penalty="l1",
        solver="saga",
    ): {"C": Real(1e-14, 1e-7, prior="log-uniform")},
    BernoulliNB(): {"alpha": Real(1e-6, 1, prior="log-uniform")},
    SparseBayesianLogisticRegression(): {"num_iterations": Integer(100, 1000), "lr": Real(1e-6, 1e-1, prior="log-uniform")},
    # AdaBoostClassifier(): {
    #     "n_estimators": Integer(25, 75),
    #     "learning_rate": Real(1e-3, 1, prior="log-uniform"),
    # },
    # XGBClassifier(): {
    #     "learning_rate": Real(1e-3, 1, prior="log-uniform"),
    #     "n_estimators": Integer(10, 100),
    # },
    # RandomForestClassifier(n_jobs=-1): {
    #     "n_estimators": Integer(10, 100),
    # },
}
