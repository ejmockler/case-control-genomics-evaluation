from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from skopt.space import Categorical, Integer, Real


class RadialBasisSVC(SVC):
    pass


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
    BernoulliNB(): {"alpha": Real(1e-10, 1e1, prior="log-uniform")},
    AdaBoostClassifier(): {
        "n_estimators": Integer(25, 75),
        "learning_rate": Real(1e-6, 1e1, prior="log-uniform"),
    },
    XGBClassifier(): {
        "learning_rate": Real(1e-6, 1e1, prior="log-uniform"),
        "n_estimators": Integer(10, 100),
    },
    RandomForestClassifier(): {
        "n_estimators": Integer(75, 200),
    },
}
