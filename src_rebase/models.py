from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn_rvm import EMRVC
from xgboost import XGBClassifier

from skopt.space import Categorical, Integer, Real


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
