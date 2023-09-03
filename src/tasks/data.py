from dataclasses import dataclass
from typing import Optional
from joblib import Parallel, delayed
from prefect import task
from sklearn.metrics import roc_auc_score, roc_curve
from shap import Explainer
from shap.maskers import Masker
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle

from config import config


@dataclass
class Genotype:
    genotype: pd.DataFrame
    ids: list
    name: str


@dataclass
class GenotypeData:
    """ "Initializes storage for genotypes and their transformation methods."""

    # TODO add allele model method
    case: Genotype
    holdout_case: Genotype
    control: Genotype
    holdout_control: Genotype

    def __post_init__(self):
        self._check_duplicates()

    def _check_duplicates(self):
        # prepare a dict to hold hashable representations of each dataframe
        df_dict = {
            "case_genotypes": [tuple(row) for row in self.case.genotype.values],
            "control_genotypes": [tuple(row) for row in self.control.genotype.values],
            "holdout_case_genotypes": [
                tuple(row) for row in self.holdout_case.genotype.values
            ],
            "holdout_control_genotypes": [
                tuple(row) for row in self.holdout_control.genotype.values
            ],
        }

        # prepare a dict to hold duplicates
        dup_dict = {}

        # check each list against all others for duplicates
        for df_name, genotypes in df_dict.items():
            other_genotypes = [
                geno
                for name, genos in df_dict.items()
                if name != df_name
                for geno in genos
            ]
            duplicates = set(genotypes) & set(other_genotypes)
            if duplicates:
                dup_dict[df_name] = list(
                    duplicates
                )  # Convert set back to list to show duplicates

        # if any duplicates found, raise an assertion error with details
        if dup_dict:
            raise AssertionError(
                f"Duplicate genotypes exist in the following groups: {dup_dict}"
            )

    def filter_allele_frequencies(self, min_allele_frequency):
        all_genotypes = pd.concat(
            [geno.genotype.dropna(how="any", axis=0) for geno in self.genotype_groups],
            axis=1,
        )
        filtered_genotypes = all_genotypes.loc[
            all_genotypes.gt(0).sum(axis=1).divide(len(all_genotypes.columns))
            >= min_allele_frequency
        ]
        print(
            f"Filtered {len(all_genotypes) - len(filtered_genotypes)} alleles with frequency below {'{:.3%}'.format(min_allele_frequency)}"
        )
        print(f"Kept {len(filtered_genotypes)} alleles")

        self.case.genotype = [
            Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
            for geno in self.case.genotype
        ]
        self.control.genotype = [
            Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
            for geno in self.control.genotype
        ]

        if len(self.holdout_case.genotype) > 0:
            self.holdout_case.genotype = [
                Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
                for geno in self.holdout_case.genotype
            ]
        if len(self.holdout_control.genotype) > 0:
            self.holdout_control.genotype = [
                Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
                for geno in self.holdout_control.genotype
            ]


@dataclass
class SampleData:
    """Initializes storage for sample data. Vectors are samples x features."""

    ids: list[str]
    labels: list[str]
    vectors: np.ndarray


@dataclass
class FoldResult(SampleData):
    """Initializes storage for sample results."""

    probabilities: list[float]
    global_feature_explanations: Optional[pd.DataFrame]
    local_feature_explanations: Optional[pd.DataFrame]
    shap_explainer: Optional[Explainer]
    shap_masker: Optional[Masker]
    fitted_optimizer: Optional[object]

    def __post_init__(self):
        self.calculate_AUC()
        self.calculate_positive_ratios()
        self.calculate_accuracy()

    def calculate_AUC(self):
        self.auc = [
            roc_auc_score(
                label,
                (probability[:, 1] if len(probability.shape) > 1 else probability),
            )
            for label, probability in zip(self.labels, self.probabilities)
        ]

    def calculate_positive_ratios(self):
        self.fpr = np.linspace(0, 1, 100)  # Define common set of FPR values
        fpr, tpr, thresholds = roc_curve(
            self.labels,
            (
                self.probabilities[:, 1]
                if len(self.probabilities.shape) > 1
                else self.probabilities
            ),
        )
        self.tpr = np.interp(self.fpr, fpr, tpr)

    def calculate_accuracy(self):
        self.case_accuracy = np.divide(
            np.sum((np.around(self.probabilities)[self.labels == 1] == 1).astype(int)),
            np.sum((self.labels == 1).astype(int)),
        )
        self.control_accuracy = np.divide(
            np.sum((np.around(self.probabilities)[self.labels == 0] == 0).astype(int)),
            np.sum((self.labels == 0).astype(int)),
        )
        self.accuracy = np.divide(
            np.sum([self.case_accuracy, self.control_accuracy]), 2
        )


@dataclass
class EvaluationResult:
    """Initializes storage for a cross-validated model; contains evaluation metrics and training data."""

    test: list[FoldResult] = []
    holdout: list[FoldResult] = []
    train: list[SampleData] = []

    def average(self):
        self.calculate_average_AUC()
        self.calculate_average_positive_ratios()
        self.calculate_average_global_feature_explanations()
        self.calculate_average_local_feature_explanations()
        self.calculate_average_accuracies()

    def calculate_average_AUC(self):
        self.average_test_auc = np.mean([fold.auc for fold in self.test])
        self.average_holdout_auc = np.mean([fold.auc for fold in self.holdout])

    def calculate_average_positive_ratios(self):
        self.average_test_tpr = np.mean([fold.tpr for fold in self.test], axis=0)
        self.average_holdout_tpr = np.mean([fold.tpr for fold in self.holdout], axis=0)
        self.average_test_fpr = np.mean([fold.fpr for fold in self.test], axis=0)
        self.average_holdout_fpr = np.mean([fold.fpr for fold in self.holdout], axis=0)

    def calculate_average_global_feature_explanations(self):
        if all(fold.global_feature_explanations is not None for fold in self.test):
            concatenated = pd.concat([fold.global_feature_explanations for fold in self.test])
            self.average_global_feature_explanations = concatenated.groupby('feature_name').agg(['average', 'standard_deviation'])

    def calculate_average_local_feature_explanations(self):
        # For test set
        all_test_explanations = []
        all_test_ids = []
        for fold in self.test:
            if fold.local_feature_explanations is not None:
                all_test_explanations.append(fold.local_feature_explanations)
                all_test_ids.extend(fold.ids)
        if all_test_explanations:
            all_test_explanations_df = pd.concat(all_test_explanations, keys=all_test_ids, names=["sample_id"])
            self.average_test_local_feature_explanations = all_test_explanations_df.groupby('sample_id').agg(['average', 'standard_deviation'])

        # For holdout set
        all_holdout_explanations = []
        all_holdout_ids = []
        for fold in self.holdout:
            if fold.local_feature_explanations is not None:
                all_holdout_explanations.append(fold.local_feature_explanations)
                all_holdout_ids.extend(fold.ids)
        if all_holdout_explanations:
            all_holdout_explanations_df = pd.concat(all_holdout_explanations, keys=all_holdout_ids, names=["sample_id"])
            self.average_holdout_local_feature_explanations = all_holdout_explanations_df.groupby('sample_id').agg(['average', 'standard_deviation'])


    def calculate_average_accuracies(self):
        if self.test:
            self.average_test_accuracy = np.mean([fold.accuracy for fold in self.test])

        if self.holdout:
            self.average_holdout_accuracy = np.mean(
                [fold.accuracy for fold in self.holdout]
            )

        if self.test:
            self.average_test_case_accuracy = np.mean(
                [fold.case_accuracy for fold in self.test]
            )
            self.average_test_control_accuracy = np.mean(
                [fold.control_accuracy for fold in self.test]
            )

        if self.holdout:
            self.average_holdout_case_accuracy = np.mean(
                [fold.case_accuracy for fold in self.holdout]
            )
            self.average_holdout_control_accuracy = np.mean(
                [fold.control_accuracy for fold in self.holdout]
            )

        all_accuracies = []
        if self.test:
            all_accuracies.extend([fold.accuracy for fold in self.test])
        if self.holdout:
            all_accuracies.extend([fold.accuracy for fold in self.holdout])
        if all_accuracies:
            self.overall_average_accuracy = np.mean(all_accuracies)

    def create_sample_results_dataframe(self):
        # Extract probabilities and ids from test and holdout folds
        all_probabilities = [
            probability[1]  # Assuming probability is a 2-element list or tuple
            for fold in self.test + self.holdout
            for probability in fold.probabilities
        ]

        all_ids = [id for fold in self.test + self.holdout for id in fold.ids]

        # Create DataFrame
        sampleResultsDataframe = pd.DataFrame.from_dict(
            {"probability": all_probabilities, "id": all_ids},
            dtype=object,
        ).set_index("id")

        return sampleResultsDataframe

    # ... other methods to calculate or update attributes


@dataclass
class BootstrapResult:
    """Initializes storage for model bootstrap."""

    model_name: str
    iteration_results: list[EvaluationResult] = []


@dataclass
class ClassificationResults:
    """Initializes storage for all model results across bootstraps."""

    models: list[BootstrapResult] = []
    sample_results: pd.DataFrame = pd.DataFrame()
    feature_importance: pd.DataFrame = pd.DataFrame()
    hyperparameters: pd.DataFrame = pd.DataFrame()


@task()
def load_fold_dataframe(args):
    field, runID = args
    try:
        if field == "testLabels" or field == "featureImportance/modelCoefficients":
            return pd.concat(
                [pd.read_csv(f"{field}/{runID}_{i}.csv") for i in range(1, 11)]
            )
        elif "average" in field.lower():
            return pd.read_csv(f"{field}/{runID}_average.csv")
        else:
            return pd.read_csv(f"{field}/{runID}.csv")
    except:
        pass


def serializeBootstrapResults(modelResult, sampleResults):
    for j in range(
        config["sampling"]["lastIteration"],
        config["sampling"]["bootstrapIterations"],
    ):
        current = modelResult[j]
        sample_result_args = [
            (fold, k, sampleID, current, modelResult)
            for fold in range(config["sampling"]["crossValIterations"])
            for k, sampleID in enumerate(
                [*current["testIDs"][fold], *current["holdoutIDs"][fold]]
            )
        ]
        Parallel(n_jobs=-1, backend="threading")(
            delayed(processSampleResult)(*args) for args in sample_result_args
        )
    for sampleID in modelResult["samples"].keys():
        if sampleID not in sampleResults:
            # label, probability
            sampleResults[sampleID] = [
                modelResult["labels"][sampleID],
                modelResult["samples"][sampleID],
            ]
        else:
            sampleResults[sampleID][1] = np.append(
                sampleResults[sampleID][1], modelResult["samples"][sampleID]
            )
    return sampleResults


def serializeResultsDataframe(sampleResults):
    for modelName in sampleResults:
        for sampleID in sampleResults[modelName]:
            # add accuracy
            sampleResults[modelName][sampleID].append(
                np.mean(
                    np.mean(
                        [
                            (
                                np.around(probability[1])
                                if len(probability.shape) > 0
                                else np.around(probability)
                            )
                            == sampleResults[sampleID][0]  # label index
                            for probability in np.hstack(
                                np.array(sampleResults[sampleID][1])
                            )  # probability index
                        ]
                    ),
                )
            )

    sampleResultsDataFrame = pd.DataFrame(
        [
            [sampleID, modelName, *sampleResults[modelName][sampleID]]
            for sampleID in sampleResults[modelName]
            for modelName in sampleResults
        ],
        columns=["id", "model", "label", "probability", "accuracy"],
    )
    sampleResultsDataFrame["meanProbability"] = sampleResultsDataFrame[
        "probability"
    ].map(
        lambda x: np.mean(
            np.array(x)[:, 1]
            if np.array(x).ndim > 1 and np.array(x).shape[1] > 1
            else np.mean(np.array(x))
        )
    )
    sampleResultsDataFrame = sampleResultsDataFrame.set_index(
        ["id", "model"], drop=False
    )
    np.set_printoptions(threshold=np.inf)
    return sampleResultsDataFrame


def processSampleResult(fold, k, sampleID, current, results):
    probability = (
        (
            current["probabilities"][fold][k]
            if len(current["probabilities"][fold][k]) <= 1
            else current["probabilities"][fold][k][1]
        )
        if k < len(current["testIDs"][fold])
        else (
            current["holdoutProbabilities"][fold][k - len(current["testIDs"][fold])]
            if len(
                current["holdoutProbabilities"][fold][k - len(current["testIDs"][fold])]
            )
            <= 1
            else current["holdoutProbabilities"][fold][
                k - len(current["testIDs"][fold])
            ][1]
        )
    )

    label = (
        current["testLabels"][fold][k]
        if k < len(current["testIDs"][fold])
        else current["holdoutLabels"][fold][k - len(current["testIDs"][fold])]
    )

    if sampleID in results["samples"]:
        results["samples"][sampleID] = np.append(
            results["samples"][sampleID], probability
        )

    else:
        results["samples"][sampleID] = np.atleast_1d(probability)
        results["labels"][sampleID] = label
    return results


@task()
def recoverPastRuns(modelStack, results=None):
    bootstrapFolders = os.listdir(
        f"projects/{config['tracking']['project']}/bootstraps"
    )
    bootstrapFolders = [int(folder) for folder in bootstrapFolders]
    bootstrapFolders.sort()

    if results == None:
        results = []

    for i, model in enumerate(modelStack):
        modelName = model.__class__.__name__
        if i + 1 > len(results):
            results.append({})
        if "samples" not in results[i]:
            results[i]["samples"] = {}
        if "labels" not in results[i]:
            results[i]["labels"] = {}
        if config["sampling"]["lastIteration"] > 0:
            # if run was interrupted, and bootstrapping began after the first iteration (and incomplete runs deleted)
            for j in tqdm(
                range(
                    0,
                    config["sampling"]["lastIteration"],
                ),
                unit="bootstrap iteration",
            ):
                results[i][j] = {}
                currentBootstrap = bootstrapFolders[j]
                modelFolders = os.listdir(
                    f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}"
                )
                currentModel = modelFolders[modelFolders.index(modelName)]
                currentFiles = os.listdir(
                    f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}"
                )
                results[i]["model"] = modelName
                for fileName in currentFiles:
                    if "testCount" in fileName:
                        results[i][j]["testCount"] = float(fileName.split("_")[1])
                    elif "trainCount" in fileName:
                        results[i][j]["trainCount"] = float(fileName.split("_")[1])
                    elif "holdoutCount" in fileName:
                        results[i][j]["holdoutCount"] = float(fileName.split("_")[1])
                    elif "meanAUC" in fileName:
                        results[i][j]["averageTestAUC"] = float(fileName.split("_")[1])
                    elif "meanHoldoutAUC" in fileName:
                        results[i][j]["averageHoldoutAUC"] = float(
                            fileName.split("_")[1]
                        )
                    elif (
                        "testLabels" in fileName
                        or "trainLabels" in fileName
                        or "holdoutLabels" in fileName
                        or "testIDs" in fileName
                        or "trainIDs" in fileName
                        or "holdoutIDs" in fileName
                    ):
                        sampleKeyFiles = os.listdir(
                            f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}"
                        )
                        sampleKeyFiles.sort(
                            key=lambda fileName: int(fileName.split(".")[0])
                        )  # ascending order by k-fold
                        results[i][j][fileName] = [
                            np.ravel(
                                pd.read_csv(
                                    f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}/{keys}",
                                    index_col=0,
                                )
                            )
                            for keys in sampleKeyFiles  # data column identical to dir name
                        ]
                    elif "sampleResults" in fileName:
                        results[i][j]["probabilities"] = pd.read_csv(
                            f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}",
                            index_col="id",
                        )
                        if "holdoutIDs" in results[i][j]:
                            results[i][j]["holdoutProbabilities"] = (
                                results[i][j]["probabilities"]
                                .loc[
                                    np.unique(
                                        np.hstack(results[i][j]["holdoutIDs"])
                                    ).tolist()
                                ]
                                .to_numpy()
                            )
                            results[i][j]["probabilities"] = (
                                results[i][j]["probabilities"]
                                .loc[np.hstack(results[i][j]["testIDs"]).tolist()]
                                .to_numpy()
                            )
                    elif "featureImportance" in fileName:
                        importanceFiles = os.listdir(
                            f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}"
                        )
                        for importanceType in importanceFiles:
                            if importanceType == "modelCoefficients":
                                coefficientFiles = os.listdir(
                                    f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}/{importanceType}"
                                )
                                coefficientFiles.sort(
                                    key=lambda fileName: int(fileName.split(".")[0])
                                )  # ascending order by k-fold
                                results[i][j]["globalExplanations"] = [
                                    pd.read_csv(
                                        f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}/{importanceType}/{foldCoefficients}",
                                        index_col="feature_name",
                                    )
                                    for foldCoefficients in coefficientFiles
                                ]
                            elif importanceType == "shapelyExplanations":
                                results[i][j][
                                    "averageShapelyExplanations"
                                ] = pd.read_csv(
                                    f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/averageLocalExplanations.csv",
                                    index_col="feature_name",
                                )
                                if "averageHoldoutShapelyExplanations" in results[i][j]:
                                    results[i][j][
                                        "averageHoldoutShapelyExplanations"
                                    ] = pd.read_csv(
                                        f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/averageHoldoutLocalExplanations.csv",
                                        index_col="feature_name",
                                    )
                    elif "hyperparameters" in fileName:
                        hyperparameterFiles = os.listdir(
                            f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}"
                        )
                        if "fittedOptimizer.pkl" in hyperparameterFiles:
                            with open(
                                f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/{fileName}/fittedOptimizer.pkl",
                                "rb",
                            ) as pickleFile:
                                results[i][j]["fittedOptimizer"] = pickle.load(
                                    pickleFile
                                )

                results[i][j]["testFPR"] = np.linspace(
                    0, 1, 100
                )  # common FPR to interpolate with other models
                testFPR, testTPR, thresholds = roc_curve(
                    np.hstack(results[i][j]["testLabels"]),
                    np.hstack(results[i][j]["probabilities"]),
                )
                # interpolate TPR at common FPR values
                tpr_interpolated = np.interp(results[i][j]["testFPR"], testFPR, testTPR)
                results[i][j]["averageTestTPR"] = tpr_interpolated
                if "holdoutProbabilities" in results[i][j]:
                    results[i][j]["holdoutFPR"] = np.linspace(0, 1, 100)
                    holdoutFPR, holdoutTPR, thresholds = roc_curve(
                        np.hstack(results[i][j]["holdoutLabels"]),
                        np.hstack(results[i][j]["holdoutProbabilities"]),
                    )
                    tpr_interpolated = np.interp(
                        results[i][j]["holdoutFPR"], holdoutFPR, holdoutTPR
                    )
                    results[i][j]["averageHoldoutTPR"] = tpr_interpolated
                results[i][j]["model"] = modelName
                allTestIDs = {
                    id
                    for foldIDs in results[i][j]["testIDs"]
                    + results[i][j]["holdoutIDs"]
                    for id in foldIDs
                }

                for sampleID in allTestIDs:
                    if any(
                        sampleID in idList for idList in results[i][j]["holdoutIDs"]
                    ):
                        currentProbabilities = results[i][j]["holdoutProbabilities"]
                        currentLabels = np.array(
                            [
                                label
                                for labelList in results[i][j]["holdoutLabels"]
                                for label in labelList
                            ]
                        )
                        currentIDs = np.array(
                            [
                                id
                                for idList in results[i][j]["holdoutIDs"]
                                for id in idList
                            ]
                        )
                    else:
                        currentProbabilities = results[i][j]["probabilities"]
                        currentLabels = np.array(
                            [
                                label
                                for labelList in results[i][j]["testLabels"]
                                for label in labelList
                            ]
                        )
                        currentIDs = np.array(
                            [id for idList in results[i][j]["testIDs"] for id in idList]
                        )
                    if sampleID in currentIDs:
                        if sampleID not in results[i]["samples"]:
                            results[i]["samples"][sampleID] = np.ravel(
                                currentProbabilities[np.where(currentIDs == sampleID)]
                            )
                            results[i]["labels"][sampleID] = currentLabels[
                                np.argmax(currentIDs == sampleID)
                            ]
                        else:
                            results[i]["samples"][sampleID] = np.append(
                                results[i]["samples"][sampleID],
                                np.ravel(
                                    currentProbabilities[
                                        np.where(currentIDs == sampleID)
                                    ]
                                ),
                            )
    return results
