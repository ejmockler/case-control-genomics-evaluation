from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Optional, Union
from joblib import Parallel, delayed
from prefect import task
from sklearn.metrics import roc_auc_score, roc_curve
from shap import Explainer
from shap.maskers import Masker
from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.typing as npt
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
        }
        
        if self.holdout_case.genotype.empty:
            df_dict.update({"holdout_case_genotypes": [
                tuple(row) for row in self.holdout_case.genotype.values
            ],})
        if self.holdout_control.genotype.empty:
            df_dict.update({"holdout_control_genotypes": [
                tuple(row) for row in self.holdout_control.genotype.values
            ],})

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

    set: str
    ids: list[str]
    labels: list[str]
    vectors: np.ndarray


@dataclass
class AggregatedMetric:
    """Base class for aggregate evaluation metrics with standard deviation."""

    data: Union[npt.NDArray[np.floating], list[float]]
    ndims: int = 0

    @cached_property
    def mean(self) -> np.floating:
        return np.mean(self.data, axis=self.ndims)

    @cached_property
    def std_dev(self) -> np.floating:
        return np.std(self.data, axis=self.ndims)


@dataclass
class FoldResult(SampleData):
    """Initializes storage for sample results."""

    probabilities: np.ndarray  # each element is a 2-element sublist of binary probabilities
    global_feature_explanations: Optional[pd.DataFrame] = None
    shap_explanation: Optional[object] = None
    shap_explainer: Optional[Explainer] = None
    shap_masker: Optional[Masker] = None
    fitted_optimizer: Optional[object] = None

    def __post_init__(self):
        self.calculate_AUC()
        self.calculate_positive_ratios()
        self.calculate_accuracy()

    @cached_property
    def local_case_explanations(self):
        localCaseExplanationsDataframe = pd.DataFrame(
            data=self.shap_explanation.values[:, :, 1],
            index=self.ids,
            columns=self.shap_explanation.feature_names,
        )
        localCaseExplanationsDataframe.index.name = "sample_id"
        return localCaseExplanationsDataframe

    @cached_property
    def local_control_explanations(self):
        localControlExplanationsDataframe = pd.DataFrame(
            data=self.shap_explanation.values[:, :, 1],
            index=self.ids,
            columns=self.shap_explanation.feature_names,
        )
        localControlExplanationsDataframe.index.name = "sample_id"
        return localControlExplanationsDataframe

    def calculate_AUC(self):
        self.auc = roc_auc_score(self.labels, self.probabilities[:, 1])

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
        # Predicted labels based on probabilities
        predicted_labels = np.around(self.probabilities[:, 1])

        # Number of correct predictions
        correct_predictions = np.sum(predicted_labels == self.labels)

        # Overall accuracy
        self.accuracy = correct_predictions / len(self.labels)

        # For the case accuracy
        positive_labels = self.labels == 1
        self.case_accuracy = np.sum(predicted_labels[positive_labels] == 1) / np.sum(
            positive_labels
        )

        # For the control accuracy
        negative_labels = self.labels == 0
        self.control_accuracy = np.sum(predicted_labels[negative_labels] == 0) / np.sum(
            negative_labels
        )


@dataclass
class EvaluationResult:
    """Initializes storage for a cross-validated model; contains evaluation metrics and training data."""

    train: list[SampleData] = field(default_factory=list)
    test: list[FoldResult] = field(default_factory=list)
    holdout: list[FoldResult] = field(default_factory=list)

    def average(self):
        self.calculate_average_AUC()
        self.calculate_average_positive_ratios()
        self.calculate_average_accuracies()

    def calculate_average_AUC(self):
        self.average_test_auc = np.mean([fold.auc for fold in self.test])
        self.average_holdout_auc = np.mean([fold.auc for fold in self.holdout])

    def calculate_average_positive_ratios(self):
        self.average_test_tpr = np.mean([fold.tpr for fold in self.test], axis=0)
        self.average_holdout_tpr = np.mean([fold.tpr for fold in self.holdout], axis=0)
        self.average_test_fpr = np.mean([fold.fpr for fold in self.test], axis=0)
        self.average_holdout_fpr = np.mean([fold.fpr for fold in self.holdout], axis=0)

    @staticmethod
    def aggregate_explanations(explanations):
        """Aggregate the explanations across folds."""
        concatenated_df = pd.concat(explanations)
        grouped = concatenated_df.groupby(level=0)
        means = grouped.mean()
        stds = grouped.std()
        return pd.concat([means, stds], axis=1, keys=["mean", "std"])

    @cached_property
    def average_global_feature_explanations(self):
        if all(fold.global_feature_explanations is not None for fold in self.test):
            concatenated = pd.concat(
                [fold.global_feature_explanations for fold in self.test]
            )
            return concatenated.groupby("feature_name").agg(
                ["mean", "std"],
            )

    @cached_property
    def average_test_local_case_explanations(self):
        all_test_explanations = []
        for fold in self.test:
            if fold.shap_explanation is not None:
                all_test_explanations.append(fold.local_case_explanations)
        if all_test_explanations:
            return self.aggregate_explanations(all_test_explanations)

    @cached_property
    def average_holdout_local_case_explanations(self):
        if self.holdout:
            all_holdout_explanations = []
            for fold in self.holdout:
                if fold.shap_explanation is not None:
                    all_holdout_explanations.append(fold.local_case_explanations)
            if all_holdout_explanations:
                return self.aggregate_explanations(all_holdout_explanations)

    @cached_property
    def average_test_local_control_explanations(self):
        all_test_explanations = []
        for fold in self.test:
            if fold.shap_explanation is not None:
                all_test_explanations.append(fold.local_control_explanations)
        if all_test_explanations:
            return self.aggregate_explanations(all_test_explanations)

    @cached_property
    def average_holdout_local_control_explanations(self):
        if self.holdout:
            all_holdout_explanations = []
            for fold in self.holdout:
                if fold.shap_explanation is not None:
                    all_holdout_explanations.append(fold.local_control_explanations)
            if all_holdout_explanations:
                return self.aggregate_explanations(all_holdout_explanations)

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

    @cached_property
    def sample_results_dataframe(self):
        # Extract probabilities and ids from test and holdout folds
        all_probabilities = [
            probability[1]  # Assuming probability exists for each binary class
            for fold in self.test + (self.holdout if self.holdout else [])
            for probability in fold.probabilities
        ]

        all_labels = [
            label
            for fold in self.test + (self.holdout if self.holdout else [])
            for label in fold.labels
        ]

        all_ids = [
            id
            for fold in self.test + (self.holdout if self.holdout else [])
            for id in fold.ids
        ]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "probability": all_probabilities,
                "prediction": np.around(all_probabilities).astype(int),
                "label": all_labels,
                "id": all_ids,
            }
        )

        # Calculate correctness of each classification
        df["accuracy"] = (df["prediction"] == df["label"]).astype(int)

        # Group by ID and compute aggregated values
        aggregation_dict = {
            "probability": [("mean", "mean"), ("std", "std")],
            "accuracy": [("mean", "mean"), ("std", "std")],
            "prediction": [("most_frequent", lambda x: x.value_counts().index[0])],
            "label": [("first", "first")],  # Get the first label for each id
            "id": [("draw_count", "size")],  # Count occurrences of each id
        }
        sampleResultsDataframe = df.groupby("id").agg(aggregation_dict)

        # Flatten MultiIndex columns for easier referencing
        sampleResultsDataframe.columns = [
            f"{col[0]}_{col[1]}" if col[1] not in ["first", "most_frequent"] else col[0]
            for col in sampleResultsDataframe.columns
        ]

        # Rename 'label_first' back to 'label'
        sampleResultsDataframe = sampleResultsDataframe.rename(
            columns={"label_first": "label", "id_draw_count": "draw_count"}
        )
        return sampleResultsDataframe

    @staticmethod
    def get_unique_samples(foldResults: Iterable[FoldResult]):
        sampleDict = {}
        for fold in foldResults:
            for i, id in enumerate(fold.ids):
                if id not in sampleDict:
                    sampleDict[id] = fold.labels[i]
        return sampleDict

    @cached_property
    def holdout_dict(self):
        return self.get_unique_samples(self.holdout)

    @cached_property
    def test_dict(self):
        return self.get_unique_samples(self.test)

    @cached_property
    def train_dict(self):
        return self.get_unique_samples(self.train)


@dataclass
class BootstrapResult:
    """Initializes storage for model bootstrap."""

    model_name: str
    iteration_results: list[EvaluationResult] = field(default_factory=list)

    @cached_property
    def sample_results_dataframe(self):
        all_probabilities = []
        all_labels = []
        all_ids = []

        # Iterate over each EvaluationResult in iteration_results
        for eval_result in self.iteration_results:
            test_folds = eval_result.test
            holdout_folds = eval_result.holdout if eval_result.holdout else []

            all_probabilities.extend(
                [
                    probability[1]  # Assuming probability exists for each binary class
                    for fold in test_folds + holdout_folds
                    for probability in fold.probabilities
                ]
            )

            all_labels.extend(
                [label for fold in test_folds + holdout_folds for label in fold.labels]
            )

            all_ids.extend(
                [id for fold in test_folds + holdout_folds for id in fold.ids]
            )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "probability": all_probabilities,
                "prediction": np.around(all_probabilities).astype(int),
                "label": all_labels,
                "id": all_ids,
            }
        )

        # Calculate correctness of each classification
        df["accuracy"] = (df["prediction"] == df["label"]).astype(int)

        # Group by ID and compute aggregated values
        aggregation_dict = {
            "probability": [("mean", "mean"), ("std", "std")],
            "accuracy": [("mean", "mean"), ("std", "std")],
            "prediction": [("most_frequent", lambda x: x.value_counts().index[0])],
            "label": [("first", "first")],  # Get the first label for each id
            "id": [("draw_count", "size")],  # Count occurrences of each id
        }
        sampleResultsDataframe = df.groupby("id").agg(aggregation_dict)

        # Flatten MultiIndex columns for easier referencing
        sampleResultsDataframe.columns = [
            "_".join(col) for col in sampleResultsDataframe.columns
        ]
        # Rename 'label_first' back to 'label' and 'id_draw_count' to 'draw_count'
        sampleResultsDataframe = sampleResultsDataframe.rename(
            columns={"label_first": "label", "id_draw_count": "draw_count"}
        )

        return sampleResultsDataframe

    def get_aggregated_attribute(self, attribute_name, agg_func=None, level=0):
        if getattr(self.iteration_results[0], attribute_name) is None:
            return None

        means_list = [
            getattr(res, attribute_name).xs("mean", axis=1, level=level)
            for res in self.iteration_results
            if getattr(res, attribute_name) is not None
        ]
        stds_list = [
            getattr(res, attribute_name).xs("std", axis=1, level=level)
            for res in self.iteration_results
            if getattr(res, attribute_name) is not None
        ]

        if agg_func is None:
            return self.aggregate_global_explanations(means_list, stds_list)
        else:
            return agg_func(means_list, stds_list)

    @cached_property
    def average_global_feature_explanations(self):
        return self.get_aggregated_attribute(
            "average_global_feature_explanations", level=1
        )

    @cached_property
    def average_test_local_case_explanations(self):
        return self.get_aggregated_attribute(
            "average_test_local_case_explanations",
            agg_func=self.aggregate_local_explanations,
        )

    @cached_property
    def average_test_local_control_explanations(self):
        return self.get_aggregated_attribute(
            "average_test_local_control_explanations",
            agg_func=self.aggregate_local_explanations,
        )

    @cached_property
    def average_holdout_local_case_explanations(self):
        return self.get_aggregated_attribute(
            "average_holdout_local_case_explanations",
            agg_func=self.aggregate_local_explanations,
        )

    @cached_property
    def average_holdout_local_control_explanations(self):
        return self.get_aggregated_attribute(
            "average_holdout_local_control_explanations",
            agg_func=self.aggregate_local_explanations,
        )

    def aggregate_global_explanations(self, means_list, stds_list):
        """Aggregate global feature importances across multiple EvaluationResults."""
        concatenated_means = pd.concat(means_list, axis=1)
        concatenated_stds = pd.concat(stds_list, axis=1)

        # Convert STDs to variances
        concatenated_variances = concatenated_stds**2

        # Calculate mean of means and mean of variances
        overall_mean = concatenated_means.mean(axis=1)
        overall_variance = concatenated_variances.mean(axis=1)

        # Convert variance back to STD
        overall_std = np.sqrt(overall_variance)

        return pd.concat([overall_mean, overall_std], axis=1, keys=["mean", "std"])

    def aggregate_local_explanations(self, means_list, stds_list):
        """
        Aggregate local feature explanations across multiple EvaluationResults.
        """
        concatenated_means = pd.concat(means_list, axis=1)
        concatenated_stds = pd.concat(stds_list, axis=1)

        # Convert STDs to variances
        concatenated_variances = concatenated_stds**2

        # Group by column names to calculate the mean of means and mean of variances for each feature
        overall_mean = concatenated_means.groupby(
            by=concatenated_means.columns, axis=1
        ).mean()
        overall_variance = concatenated_variances.groupby(
            by=concatenated_variances.columns, axis=1
        ).mean()

        # Convert variance back to STD
        overall_std = np.sqrt(overall_variance)

        # Using MultiIndex for columns
        multi_columns = pd.MultiIndex.from_product(
            [["mean", "std"], overall_mean.columns], names=["stat", "feature"]
        )
        aggregated = pd.concat([overall_mean, overall_std], axis=1)
        aggregated.columns = multi_columns

        return aggregated

    def calculate_average_accuracies(self):
        for res in self.iteration_results:
            res.average()
        # Test accuracies
        all_test_accuracies = [
            res.average_test_accuracy for res in self.iteration_results
        ]
        self.average_test_accuracy = np.mean(all_test_accuracies)

        all_test_case_accuracies = [
            res.average_test_case_accuracy for res in self.iteration_results
        ]
        self.average_test_case_accuracy = np.mean(all_test_case_accuracies)

        all_test_control_accuracies = [
            res.average_test_control_accuracy for res in self.iteration_results
        ]
        self.average_test_control_accuracy = np.mean(all_test_control_accuracies)

        # Holdout accuracies
        all_holdout_accuracies = [
            res.average_holdout_accuracy
            for res in self.iteration_results
            if res.holdout
        ]
        if all_holdout_accuracies:
            self.average_holdout_accuracy = np.mean(all_holdout_accuracies)

            all_holdout_case_accuracies = [
                res.average_holdout_case_accuracy
                for res in self.iteration_results
                if res.holdout
            ]
            self.average_holdout_case_accuracy = np.mean(all_holdout_case_accuracies)

            all_holdout_control_accuracies = [
                res.average_holdout_control_accuracy
                for res in self.iteration_results
                if res.holdout
            ]
            self.average_holdout_control_accuracy = np.mean(
                all_holdout_control_accuracies
            )

    def get_unique_samples(self, setType):
        sampleDict = {}
        for res in self.iteration_results:
            for id in list(getattr(res, f"{setType}_dict").keys()):
                if id not in sampleDict:
                    sampleDict[id] = getattr(res, f"{setType}_dict")[id]
        return sampleDict

    @cached_property
    def holdout_dict(self):
        return self.get_unique_samples("holdout")

    @cached_property
    def test_dict(self):
        return self.get_unique_samples("test")

    @cached_property
    def train_dict(self):
        return self.get_unique_samples("train")


@dataclass
class ClassificationResults:
    """Initializes storage for all model results across bootstraps."""

    modelResults: list[BootstrapResult] = field(default_factory=list)


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
