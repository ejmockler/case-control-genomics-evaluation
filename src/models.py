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
    def __init__(
        self,
        *,
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
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )


class LinearSVC(SVC):
    def __init__(
        self,
        *,
        C=1.0,
        kernel="linear",
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
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )


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
    LogisticRegression(penalty="elasticnet", solver="saga"): {
        "C": Real(1e-6, 1, prior="log-uniform"),
        "l1_ratio": Real(1e-6, 1, prior="log-uniform"),
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
