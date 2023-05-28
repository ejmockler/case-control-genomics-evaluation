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

from env import neptune_api_token

RadialBasisSVC = SVC
RadialBasisSVC.__name__ = "RadialBasisSVC"

config = {
    "vcfLike": {
        "path": "../notebook/Variant_report_NUPs_fixed_2022-03-28.xlsx",  # variant call table with annotations
        "sheet": "all cases vs all controls",  # sheet name if Excel spreadsheet
        "indexColumn": [
            "chrom",
            "position",
            "Gene",
        ],  # header that indexes variants (set as list with multiple columns)
        "binarize": True,  # binarize variants to 0/1, or sum to weigh allele frequency
        "minAlleleFrequency": 0.05,  # filter out variants with allele frequency less than this
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "Nucleoporin genes, well-classified cases",  # name of the experiment
        "entity": "ejmockler",
        "project": "ALS-NUPs-WellClassified",
        "plotAllSampleImportances": True,  # plot all sample feature importances in Neptune
        "token": neptune_api_token,
    },
    "clinicalTable": {
        "path": "../notebook/ACWM.xlsx",  # clinical data as Excel spreadsheet
        "idColumn": "ExternalSampleId",  # genotype ID header
        "uniqueIdColumn": "ExternalSubjectId",  # unique ID for each patient
        "labelColumn": "Subject Group",  # header that has case/control labels
        "controlLabels": [
            "Non-Neurological Control"
        ],  # these labels include external sample IDs (like 1000 Genomes)
        "caseLabels": [],
        "controlAlias": "control",
        "caseAlias": "case",
        "filters": "pct_european>=0.85",  # filter out nonhomogenous samples with less than 85% European ancestry
    },
    "externalTables": {
        "path": [
            "../notebook/igsr-1000 genomes phase 3 release.tsv",
            "../notebook/accurateCases.csv",
        ],  # external sample table
        "label": [
            "control",
            "case",
        ],  # case | control | mixed (mixed labels are held out as an external test set)
        "idColumn": ["Sample name", "id"],  # sample ID header
        "filters": [
            "`Superpopulation code`=='EUR' & `Population name`!='Finnish'",
            "",
        ],  # remove finnish samples due to unusual homogeneity (verify w/ PCA)
    },
    "sampling": {
        "bootstrapIterations": 60,
        "startFrom": 42,  # start from this bootstrap iteration
        "crossValIterations": 10,  # number of validations per bootstrap iteration
        "holdoutSplit": 0.1,
    },
    "modelStack": {
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
        # TabNetClassifier: {
        #     "n_d": Integer(8, 64),
        #     "n_a": Integer(8, 64),
        #     "n_steps": Integer(3, 10),
        #     "lambda_sparse": Real(1e-4, 1e+1, prior="log-uniform"),
        # },
        MultinomialNB(): {"alpha": Real(1e-10, 1e1, prior="log-uniform")},
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
    },
}
