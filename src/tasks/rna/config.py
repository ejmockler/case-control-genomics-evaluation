from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt.space import Categorical, Integer, Real

config = {
    "data": {
        "counts_file": "../transcriptomics/4_matrix/AnswerALS-651-T-v1-release6_raw-counts.csv",
        "clinical_file": "../adhoc analysis/ALS Consortium WGS Metadata 03112022.xlsx",
        "sex_genotype_column": "Sex Genotype",
        "case_label": "CASE",
        "control_label": "CTRL",
    },
    "clinicalTable": {
        "filters": "pct_european>=0.85",  # Example filter
        "idColumn": "ExternalSubjectId",
    },
    "external_samples": {
        "metadata": [
            # {
            #     "setType": "crossval",
            #     "path": "path/to/external_sample_file1.tsv",
            #     "label": "control",
            #     "idColumn": "Sample name",
            #     "filters": "`Superpopulation code`=='EUR'",
            # },
            # {
            #     "setType": "Cases vs. Controls (Ethnically-Variable)",
            #     "path": "path/to/external_sample_file2.tsv",
            #     "label": "case",
            #     "idColumn": "ExternalSampleId",
            #     "filters": "`Subject Group`.str.contains('ALS Spectrum MND') & `pct_european`<0.85",
            # },
            # Add more external sample configurations as needed
        ],
    },
    "preprocessing": {
        "remove_invariant_features": True,
        "normalization_method": "deseq2",  # Options: 'deseq2', 'log', 'none'
        "feature_selection": {
            "method": "variance_threshold",
            "params": {"threshold": 0.0},  # Adjust as needed
        },
    },
    "ml_workflow": {
        "test_size": 0.2,
        "n_bootstraps": 2,
        "n_jobs": -1,
        "random_state": 42,
        "balanced_accuracy_threshold": 0.8,
    },
    "models": {
        # "LinearSVC": {
        #     "class": SVC,
        #     "fixed_params": {"probability": True, "kernel": "linear"},
        #     "search_space": {
        #         "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        #         "C": Real(1e-3, 10, prior="log-uniform"),
        #     },
        # },
        # "RadialBasisSVC": {
        #     "class": SVC,
        #     "fixed_params": {"probability": True, "kernel": "rbf"},
        #     "search_space": {
        #         "tol": Real(1e-6, 1e-3, prior="log-uniform"),
        #         "C": Real(1e-3, 10, prior="log-uniform"),
        #         "gamma": Categorical(["scale", "auto"]),
        #     },
        # },
        "LogisticRegression": {
            "class": LogisticRegression,
            "fixed_params": {"penalty": "elasticnet", "solver": "saga"},
            "search_space": {
                "C": Real(1e-6, 1, prior="log-uniform"),
                "l1_ratio": Real(1e-6, 1, prior="log-uniform"),
            },
        },
        "BernoulliNB": {
            "class": BernoulliNB,
            "fixed_params": {},
            "search_space": {"alpha": Real(1e-6, 1, prior="log-uniform")},
        },
        # "AdaBoostClassifier": {
        #     "class": AdaBoostClassifier,
        #     "fixed_params": {},
        #     "search_space": {
        #         "n_estimators": Integer(25, 75),
        #         "learning_rate": Real(1e-3, 1, prior="log-uniform"),
        #     },
        # },
        # "XGBClassifier": {
        #     "class": XGBClassifier,
        #     "fixed_params": {},
        #     "search_space": {
        #         "learning_rate": Real(1e-3, 1, prior="log-uniform"),
        #         "n_estimators": Integer(10, 100),
        #     },
        # },
        # "RandomForestClassifier": {
        #     "class": RandomForestClassifier,
        #     "fixed_params": {"n_jobs": -1},
        #     "search_space": {
        #         "n_estimators": Integer(10, 100),
        #     },
        # },
    },
    "hyperparameter_optimization": {
        "n_iter": 5,
        "cv": 5,
        "scoring": "balanced_accuracy",
        "n_jobs": -1,
        "verbose": 1,
    },
    "analysis": {
        "plot_feature_importance": True,
        "n_top_features": 20,
        "save_results": True,
        "results_dir": "path/to/results",
    },
}
