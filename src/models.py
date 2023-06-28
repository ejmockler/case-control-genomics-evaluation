from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from skopt.space import Categorical, Integer, Real

RadialBasisSVC = SVC
RadialBasisSVC.__name__ = "RadialBasisSVC"

stack = {
    LinearSVC(): {
        "tol": Real(1e-6, 1e1, prior="log-uniform"),
        "C": Real(1e-4, 1e1, prior="log-uniform"),
    },
    RadialBasisSVC(probability=True, kernel="rbf"): {
        "tol": Real(1e-4, 1e1, prior="log-uniform"),
        "C": Real(1e-4, 1e1, prior="log-uniform"),
        "gamma": Categorical(["scale", "auto"]),
    },
    LogisticRegression(penalty="l2", solver="saga"): {
        "tol": Real(1e-6, 1e1, prior="log-uniform"),
        "C": Real(1e-4, 1e1, prior="log-uniform"),
    },
    MultinomialNB(): {"alpha": Real(1e-10, 1e1, prior="log-uniform")},
    AdaBoostClassifier(): {
        "n_estimators": Integer(25, 75),
        "learning_rate": Real(1e-6, 1e1, prior="log-uniform"),
    },
    XGBClassifier(): {  # single thread to avoid memory issues (segfaults) in multiprocessing
        "learning_rate": Real(1e-6, 1e1, prior="log-uniform"),
        "n_estimators": Integer(10, 100),
    },
    RandomForestClassifier(): {
        "n_estimators": Integer(75, 200),
    },
}