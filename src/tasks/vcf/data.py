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
    """Associates DataFrame of sample genotypes (variants x samples) with metadata: a list of sample IDs & the sample set name."""

    genotype: pd.DataFrame
    ids: list
    name: str


@dataclass
class GenotypeData:
    """Initializes storage for genotypes and their transformation methods."""

    # TODO add allele model method
    case: Genotype
    holdout_case: dict
    control: Genotype
    holdout_control: dict

    def __post_init__(self):
        self._check_duplicates()

    @staticmethod
    def get_allele_frequencies(genotypeDataframe):
        # sum counts of non-zero alleles over all samples, normalize by number of samples
        return (
            genotypeDataframe.gt(0).sum(axis=1).divide(len(genotypeDataframe.columns))
        )

    def _check_duplicates(self):
        # Prepare a dict to hold column labels of each dataframe
        df_dict = {
            "case_genotypes": list(self.case.genotype.columns),
            "control_genotypes": list(self.control.genotype.columns),
        }

        df_dict["holdout_case_genotypes"] = list()
        for holdout_case in self.holdout_case.values():
            if not holdout_case.genotype.empty:
                df_dict["holdout_case_genotypes"].extend(
                    list(holdout_case.genotype.columns)
                )

        df_dict["holdout_control_genotypes"] = list()
        for holdout_control in self.holdout_control.values():
            if not holdout_control.genotype.empty:
                df_dict["holdout_control_genotypes"].extend(
                    list(holdout_control.genotype.columns)
                )

        # Prepare a dict to hold duplicates
        dup_dict = {}

        # Check each column list against all others for duplicates
        for df_name, columns in df_dict.items():
            other_columns = [
                col for name, cols in df_dict.items() if name != df_name for col in cols
            ]
            duplicates = set(columns) & set(other_columns)
            if duplicates:
                dup_dict[df_name] = list(duplicates)  # Store duplicates in a list

        # If any duplicates are found, raise an assertion error with details
        if dup_dict:
            raise AssertionError(
                f"Duplicate column labels exist in the following groups: {dup_dict}"
            )


@dataclass
class SampleData:
    """Initializes storage for sample data. Vectors are samples x features."""

    name: str  # 'test', 'train' or holdout set name
    ids: list[str]
    labels: list[str]
    vectors: np.ndarray
    geneCount: int


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

    probabilities: (
        np.ndarray
    )  # each element is a 2-element sublist of binary probabilities
    global_feature_explanations: pd.DataFrame = pd.DataFrame()
    shap_explanation: Optional[object] = None
    shap_explainer: Optional[Explainer] = None
    shap_masker: Optional[Masker] = None
    optimizer_results: Optional[list] = None
    hyperparameters: Optional[dict] = None

    def __post_init__(self):
        if "excess" not in self.name:
            self.calculate_AUC()
        self.calculate_positive_ratios()
        self.calculate_accuracy()

    def append_allele_frequencies(self, genotypeData: GenotypeData):
        if (
            isinstance(self.global_feature_explanations, pd.DataFrame)
            and not self.global_feature_explanations.empty
        ):
            # Determine target data based on sample set name
            targetCaseData = (
                {"test": genotypeData.case}
                if self.name == "test"
                else genotypeData.holdout_case
            )
            targetControlData = (
                {"test": genotypeData.control}
                if self.name == "test"
                else genotypeData.holdout_control
            )

            setName = self.name
            # Append MAF (Minor Allele Frequencies) to the global feature explanations DataFrame
            self.global_feature_explanations[f"{setName}__case_maf"] = (
                self.global_feature_explanations.index.map(
                    genotypeData.get_allele_frequencies(
                        targetCaseData[setName].genotype[
                            self.ids[np.where(self.labels == 1)]
                        ]
                    ).to_dict()
                )
            )
            # Check for zygosity configuration and calculate RAF (Rare Allele Frequencies)
            if config["vcfLike"]["zygosity"]:
                # Assuming zygosity is coded as 0, 1, 2 (homozygous reference, heterozygous, homozygous and/or rare variant)
                # Calculate frequency of rare variant (zygosity == 2)
                case_raf = (
                    targetCaseData[setName].genotype[
                        self.ids[np.where(self.labels == 1)]
                    ]
                    == 2
                ).mean(axis=1)
                self.global_feature_explanations[f"{setName}__case_raf"] = case_raf

            self.global_feature_explanations[f"{setName}__control_maf"] = (
                self.global_feature_explanations.index.map(
                    genotypeData.get_allele_frequencies(
                        targetControlData[setName].genotype[
                            self.ids[np.where(self.labels == 0)]
                        ]
                    ).to_dict()
                )
            )
            if config["vcfLike"]["zygosity"]:
                control_raf = (
                    targetControlData[setName].genotype[
                        self.ids[np.where(self.labels == 0)]
                    ]
                    == 2
                ).mean(axis=1)
                self.global_feature_explanations[f"{setName}__control_raf"] = (
                    control_raf
                )

    @cached_property
    def local_explanations(self):
        localExplanationsDataframe = pd.DataFrame(
            data=self.shap_explanation.values[:, :, 1],
            index=self.ids,
            columns=self.shap_explanation.feature_names,
        )
        localExplanationsDataframe.index.name = "sample_id"
        return localExplanationsDataframe

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
    holdout: list[dict] = field(default_factory=list)
    excess: list[FoldResult] = field(default_factory=list)

    def average(self):
        self.calculate_average_AUC()
        self.calculate_average_positive_ratios()
        self.calculate_average_accuracies()

    def calculate_average_AUC(self):
        self.average_test_auc = np.mean([fold.auc for fold in self.test])
        self.average_holdout_auc = self.calculate_holdout_averages("auc")

    def calculate_average_positive_ratios(self):
        self.average_test_tpr = np.mean([fold.tpr for fold in self.test], axis=0)
        self.average_holdout_tpr = self.calculate_holdout_averages("tpr")
        self.average_test_fpr = np.mean([fold.fpr for fold in self.test], axis=0)
        self.average_holdout_fpr = self.calculate_holdout_averages("fpr")

    def calculate_holdout_averages(self, metric):
        # calculate mean across all folds for each holdout set
        holdout_averages = {}
        for setName in self.holdout[0]:
            if setName not in holdout_averages:
                holdout_averages[setName] = []
            holdout_averages[setName].extend(
                [getattr(fold[setName], metric) for fold in self.holdout]
            )

        # Calculate mean across all folds for each holdout set
        for setName in holdout_averages:
            if metric in ["tpr", "fpr"]:  # Metrics that require axis=0 for np.mean
                holdout_averages[setName] = np.mean(holdout_averages[setName], axis=0)
            else:
                holdout_averages[setName] = np.mean(holdout_averages[setName])

        return holdout_averages

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
        concatenated = pd.DataFrame()
        if all(fold.global_feature_explanations is not None for fold in self.test):
            concatenated = pd.concat(
                [fold.global_feature_explanations for fold in self.test]
                + [concatenated]
            )
        if all(
            fold[setName].global_feature_explanations is not None
            for setName in self.holdout[0]
            for fold in self.holdout
        ):
            columnsMAF = [
                colName for colName in concatenated.columns if "maf" in colName
            ]
            aggDict = {
                "feature_importances": ["mean", "std"],
                **{col: ["mean", "std"] for col in columnsMAF},
            }
            if config["vcfLike"]["zygosity"]:
                columnsRAF = [
                    colName for colName in concatenated.columns if "raf" in colName
                ]
                aggDict.update({**{col: ["mean", "std"] for col in columnsRAF}})
            return concatenated.groupby("feature_name").agg(aggDict)

    @cached_property
    def average_test_local_explanations(self):
        all_test_explanations = []
        for fold in self.test:
            if fold.shap_explanation is not None:
                all_test_explanations.append(fold.local_explanations)
        if all_test_explanations:
            return self.aggregate_explanations(all_test_explanations)

    @cached_property
    def average_holdout_local_explanations(self):
        if self.holdout:
            holdout_explanations_by_set = {}
            for setName in self.holdout[0]:
                current_holdout_explanations = []
                for fold in self.holdout:
                    if fold[setName].shap_explanation is not None:
                        current_holdout_explanations.append(
                            fold[setName].local_explanations
                        )
                if holdout_explanations_by_set[setName]:
                    holdout_explanations_by_set[setName] = self.aggregate_explanations(
                        current_holdout_explanations
                    )
            self.aggregate_explanations(current_holdout_explanations)
        return holdout_explanations_by_set

    def calculate_average_accuracies(self):
        if self.test:
            self.average_test_accuracy = np.mean([fold.accuracy for fold in self.test])
            self.average_test_case_accuracy = np.mean(
                [fold.case_accuracy for fold in self.test]
            )
            self.average_test_control_accuracy = np.mean(
                [fold.control_accuracy for fold in self.test]
            )

        if self.holdout:
            if not hasattr(self, "average_holdout_accuracy"):
                self.average_holdout_accuracy = {}
                self.average_holdout_case_accuracy = {}
                self.average_holdout_control_accuracy = {}
            for setName in self.holdout[0]:
                all_accuracies = []
                case_accuracies = []
                control_accuracies = []
                for fold in self.holdout:
                    all_accuracies.append(fold[setName].accuracy)
                    case_accuracies.append(fold[setName].case_accuracy)
                    control_accuracies.append(fold[setName].control_accuracy)
                # Calculate the mean across all folds for each holdout set
                self.average_holdout_accuracy[setName] = np.mean(all_accuracies)
                self.average_holdout_case_accuracy[setName] = np.mean(case_accuracies)
                self.average_holdout_control_accuracy[setName] = np.mean(
                    control_accuracies
                )

        if self.excess:
            self.average_excess_accuracy = np.mean(
                [fold.accuracy for fold in self.excess]
            )

        all_accuracies = []
        if self.test:
            all_accuracies.extend([fold.accuracy for fold in self.test])
        if self.holdout:
            for setName in self.holdout[0]:
                all_accuracies.extend([fold[setName].accuracy for fold in self.holdout])
        if self.excess:
            all_accuracies.extend([fold.accuracy for fold in self.excess])
        if all_accuracies:
            self.overall_average_accuracy = np.mean(all_accuracies)

    def aggregate_sample_results(self, folds, is_holdout=False):
        np.set_printoptions(threshold=np.inf)

        if not is_holdout:
            return self.aggregate_folds(folds)
        else:
            holdout_dfs = {}
            # assuming all holdout sets are evaluated for every fold
            for setName in folds[0]:
                holdout_dfs[setName] = self.aggregate_folds(
                    [fold[setName] for fold in folds]
                )
            return holdout_dfs

    def aggregate_folds(self, folds):
        all_probabilities = []
        all_labels = []
        all_ids = []

        for fold in folds:
            all_probabilities.extend(
                [probability[1] for probability in fold.probabilities]
            )
            all_labels.extend(fold.labels)
            all_ids.extend(fold.ids)

        df = pd.DataFrame(
            {
                "probability": all_probabilities,
                "probabilities": all_probabilities,
                "prediction": np.around(all_probabilities).astype(int),
                "label": all_labels,
                "id": all_ids,
            }
        )

        df["accuracy"] = (df["prediction"] == df["label"]).astype(int)

        aggregation_dict = {
            "probability": [("mean", "mean"), ("std", "std")],
            "probabilities": [("list", list)],
            "prediction": [("most_frequent", lambda x: x.value_counts().index[0])],
            "label": [("first", "first")],  # Get the first label for each id
            "accuracy": [("mean", "mean"), ("std", "std")],
            "id": [("draw_count", "size")],  # Count occurrences of each id
        }

        sampleResultsDataframe = df.groupby("id").agg(aggregation_dict)
        sampleResultsDataframe.columns = [
            "_".join(col).strip() for col in sampleResultsDataframe.columns.values
        ]

        sampleResultsDataframe = sampleResultsDataframe.rename(
            columns={"label_first": "label", "id_draw_count": "draw_count"}
        )

        return sampleResultsDataframe.sort_index()

    @cached_property
    def test_results_dataframe(self):
        return self.aggregate_sample_results(self.test)

    @cached_property
    def holdout_results_dataframe(self):
        return self.aggregate_sample_results(self.holdout, is_holdout=True)

    @cached_property
    def excess_results_dataframe(self):
        return self.aggregate_sample_results(self.excess)

    @staticmethod
    def get_unique_samples(folds: Iterable[FoldResult], is_holdout=False):
        if not is_holdout:
            sampleDict = {}
            for fold in folds:
                for i, id_ in enumerate(fold.ids):
                    if id_ not in sampleDict:
                        sampleDict[id_] = fold.labels[i]
            return sampleDict
        else:
            holdoutSampleDict = {}
            for setName in folds[0]:
                if setName not in holdoutSampleDict:
                    holdoutSampleDict[setName] = {}
                for (
                    fold
                ) in folds:  # folds is expected to be a list of dicts for holdout
                    for i, id_ in enumerate(fold[setName].ids):
                        if id_ not in holdoutSampleDict[setName]:
                            holdoutSampleDict[setName][id_] = fold[setName].labels[i]
            return holdoutSampleDict

    @cached_property
    def holdout_dict(self):
        return self.get_unique_samples(self.holdout, is_holdout=True)

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

    def aggregate_sample_results(self, fold_type):
        if fold_type == "holdout":
            return self.aggregate_holdout_sample_results()
        else:
            return self.aggregate_fold_sample_results(fold_type)

    def aggregate_fold_sample_results(self, fold_type):
        """Aggregate results for test and excess data, which do not involve subsets."""
        all_probabilities, all_labels, all_ids = [], [], []
        for eval_result in self.iteration_results:
            folds = getattr(eval_result, fold_type, [])
            for fold in folds:
                all_probabilities.extend(
                    fold.probabilities[:, 1]
                    if fold.probabilities.ndim > 1
                    else fold.probabilities
                )
                all_labels.extend(fold.labels)
                all_ids.extend(fold.ids)

        return self.create_results_dataframe(all_probabilities, all_labels, all_ids)

    def aggregate_holdout_sample_results(self):
        """Special handling for aggregating holdout data across subsets."""
        dfs = {}
        for setName in self.iteration_results[0].holdout[0]:
            all_probabilities, all_labels, all_ids = [], [], []
            for eval_result in self.iteration_results:
                for fold in eval_result.holdout:
                    all_probabilities.extend(
                        fold[setName].probabilities[:, 1]
                        if fold[setName].probabilities.ndim > 1
                        else fold[setName].probabilities
                    )
                    all_labels.extend(fold[setName].labels)
                    all_ids.extend(fold[setName].ids)
                dfs[setName] = self.create_results_dataframe(
                    all_probabilities, all_labels, all_ids
                )
        return dfs

    def create_results_dataframe(self, probabilities, labels, ids):
        """Creates a results dataframe from provided lists of probabilities, labels, and ids."""
        df = pd.DataFrame(
            {
                "probability": probabilities,
                "prediction": np.around(probabilities).astype(int),
                "label": labels,
                "id": ids,
            }
        )
        df["accuracy"] = (df["prediction"] == df["label"]).astype(int)

        aggregation_dict = {
            "probability": [("mean", "mean"), ("std", "std")],
            "prediction": [("most_frequent", lambda x: x.value_counts().index[0])],
            "label": [("first", "first")],
            "accuracy": [("mean", "mean"), ("std", "std")],
            "id": [("draw_count", "size")],
        }

        df = df.groupby("id").agg(aggregation_dict)

        df.columns = ["_".join(col).strip() for col in df.columns.values]

        return df.rename(
            columns={"label_first": "label", "id_draw_count": "draw_count"}
        )

    @cached_property
    def test_results_dataframe(self):
        return self.aggregate_sample_results("test")

    @cached_property
    def holdout_results_dataframe(self):
        return self.aggregate_sample_results("holdout")

    @cached_property
    def excess_results_dataframe(self):
        return self.aggregate_sample_results("excess")

    def get_aggregated_attribute(self, attribute_name, agg_func=None, level=0):
        # Initialize a dict to hold results if dealing with holdout data
        results_dict = {}

        # Check if we're dealing with structured holdout data
        is_holdout_structure = (
            attribute_name.startswith("holdout")
            and "holdout" in self.iteration_results[0].__dict__
        )

        if is_holdout_structure:
            # Loop through each subset within the holdout data
            for setName in self.iteration_results[0].holdout[0].keys():
                means_list = []
                stds_list = []

                # Aggregate data for each holdout subset
                for res in self.iteration_results:
                    if res.holdout:
                        for fold in res.holdout:
                            if setName in fold:
                                data = getattr(fold[setName], attribute_name, None)
                                if isinstance(data, pd.DataFrame) and not data.empty:
                                    means_list.append(
                                        data.xs("mean", axis=1, level=level)
                                    )
                                    stds_list.append(
                                        data.xs("std", axis=1, level=level)
                                    )
                if not means_list:
                    return None
                # Use the provided aggregation function or the default
                if agg_func:
                    results_dict[setName] = agg_func(means_list, stds_list)
                else:
                    results_dict[setName] = self.aggregate_global_explanations(
                        means_list, stds_list
                    )
        else:
            # Handle non-holdout attributes as before
            means_list, stds_list = [], []
            for res in self.iteration_results:
                data = getattr(res, attribute_name, None)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    means_list.append(data.xs("mean", axis=1, level=level))
                    stds_list.append(data.xs("std", axis=1, level=level))
            if not means_list:
                return None
            if agg_func:
                return agg_func(means_list, stds_list)
            else:
                return self.aggregate_global_explanations(means_list, stds_list)

        return results_dict if is_holdout_structure else None

    def aggregate_global_explanations(self, means_list, stds_list):
        concatenated_variances = pd.concat(stds_list, axis=0)
        concatenated_means = pd.concat(means_list, axis=0)

        # Now apply sqrt to the aggregated variances to get the standard deviation
        results_std = concatenated_variances.apply(np.sqrt)

        # Define the columns to be aggregated and the methods of aggregation
        columns_to_aggregate = means_list[0].columns
        agg_dict = {}

        for col in columns_to_aggregate:
            if "maf" in col or "raf" in col:
                agg_dict[col] = [
                    "mean",
                    "std",
                ]  # Only storing these labels for later use
            elif col == "feature_importances":
                agg_dict[col] = ["mean", "std", "min", "max", "median"]

        # Prepare the final dataframe with proper labels for mean and std
        final_results = pd.DataFrame()
        for col in agg_dict:
            if "mean" in agg_dict[col]:
                final_results[f"mean_{col}"] = (
                    concatenated_means[col].groupby("feature_name").mean()
                )
            if "std" in agg_dict[col]:
                final_results[f"std_{col}"] = (
                    concatenated_variances[col].groupby("feature_name").mean()
                )
            if "min" in agg_dict[col]:
                final_results[f"min_{col}"] = (
                    concatenated_means[col].groupby("feature_name").min()
                )
            if "max" in agg_dict[col]:
                final_results[f"max_{col}"] = (
                    concatenated_means[col].groupby("feature_name").max()
                )
            if "median" in agg_dict[col]:
                final_results[f"median_{col}"] = (
                    concatenated_means[col].groupby("feature_name").median()
                )

        return final_results

    @cached_property
    def average_global_feature_explanations(self):
        return self.get_aggregated_attribute(
            "average_global_feature_explanations", level=1
        )

    @cached_property
    def average_test_local_explanations(self):
        return self.get_aggregated_attribute(
            "average_test_local_explanations",
            agg_func=self.aggregate_local_explanations,
        )

    @cached_property
    def average_holdout_local_explanations(self):
        return self.get_aggregated_attribute(
            "average_holdout_local_explanations",
            agg_func=self.aggregate_local_explanations,
        )

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
        self.average_test_accuracy = np.mean(
            [res.average_test_accuracy for res in self.iteration_results]
        )
        self.average_test_case_accuracy = np.mean(
            [res.average_test_case_accuracy for res in self.iteration_results]
        )
        self.average_test_control_accuracy = np.mean(
            [res.average_test_control_accuracy for res in self.iteration_results]
        )

        # Initialize dictionaries to hold aggregated accuracies for holdout sets
        holdout_accuracies = {}
        holdout_case_accuracies = {}
        holdout_control_accuracies = {}

        for res in self.iteration_results:
            if res.holdout:
                for setName in res.average_holdout_accuracy.keys():
                    if setName not in holdout_accuracies:
                        holdout_accuracies[setName] = []
                        holdout_case_accuracies[setName] = []
                        holdout_control_accuracies[setName] = []

                    holdout_accuracies[setName].append(
                        res.average_holdout_accuracy[setName]
                    )
                    holdout_case_accuracies[setName].append(
                        res.average_holdout_case_accuracy[setName]
                    )
                    holdout_control_accuracies[setName].append(
                        res.average_holdout_control_accuracy[setName]
                    )

        # Calculate the mean across all bootstrap iterations for each holdout set
        self.average_holdout_accuracy = {
            setName: np.mean(accuracies)
            for setName, accuracies in holdout_accuracies.items()
        }
        self.average_holdout_case_accuracy = {
            setName: np.mean(accuracies)
            for setName, accuracies in holdout_case_accuracies.items()
        }
        self.average_holdout_control_accuracy = {
            setName: np.mean(accuracies)
            for setName, accuracies in holdout_control_accuracies.items()
        }

    def get_unique_samples(self, setType):
        if setType != "holdout":
            # For test and train, where data is not subdivided into subsets
            sampleDict = {}
            for res in self.iteration_results:
                sampleDict.update(getattr(res, f"{setType}_dict", {}))
            return sampleDict
        else:
            # For holdout, organize samples by subset
            holdoutSampleDict = {}
            for res in self.iteration_results:
                if hasattr(res, "holdout_dict"):
                    for setName, subsetSamples in res.holdout_dict.items():
                        if setName not in holdoutSampleDict:
                            holdoutSampleDict[setName] = {}
                        for id, value in subsetSamples.items():
                            if id not in holdoutSampleDict[setName]:
                                holdoutSampleDict[setName][id] = value
            return holdoutSampleDict

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
    """Initializes storage for model result across all bootstraps."""

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
def filterTable(table, filterString):
    if not filterString:
        return table
    print(f"Filtering: {filterString}")
    filteredTable = table.query(filterString, engine="python")
    return filteredTable


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
                                results[i][j]["averageShapelyExplanations"] = (
                                    pd.read_csv(
                                        f"projects/{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentModel}/averageLocalExplanations.csv",
                                        index_col="feature_name",
                                    )
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
