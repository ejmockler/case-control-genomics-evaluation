from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from skopt.space import Categorical, Integer, Real


class RadialBasisSVC(SVC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LinearSVC(SVC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


stack = {
    LinearSVC(probability=True, kernel="linear"): {
        "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        "C": Real(1e-3, 10, prior="log-uniform"),
    },
    RadialBasisSVC(probability=True, kernel="rbf"): {
        "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        "C": Real(1e-3, 10, prior="log-uniform"),
        "gamma": Categorical(["scale", "auto"]),
    },
    LogisticRegression(penalty="l2", solver="saga"): {
        "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        "C": Real(1e-3, 10, prior="log-uniform"),
    },
    BernoulliNB(): {"alpha": Real(1e-6, 1, prior="log-uniform")},
    AdaBoostClassifier(): {
        "n_estimators": Integer(25, 75),
        "learning_rate": Real(1e-3, 1, prior="log-uniform"),
    },
    XGBClassifier(): {
        "learning_rate": Real(1e-3, 1, prior="log-uniform"),
        "n_estimators": Integer(10, 100),
    },
    RandomForestClassifier(n_jobs=-1): {
        "n_estimators": Integer(10, 100),
    },
}
