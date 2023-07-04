from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from skopt.space import Categorical, Integer, Real


class RadialBasisSVC(SVC):
    pass  ## TODO properly init:
    ## RuntimeError: scikit-learn estimators should always specify their parameters in the signature of their __init__ (no varargs). <class 'models.RadialBasisSVC'> with constructor (self, *args, **kwargs) doesn't  follow this convention.


stack = {
    LinearSVC(): {
        "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        "C": Real(1e-3, 10, prior="log-uniform"),
    },
    RadialBasisSVC(probability=True): {
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
    RandomForestClassifier(): {
        "n_estimators": Integer(75, 200),
    },
}
