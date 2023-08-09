from dataclasses import dataclass
from prefect import task
from sklearn.metrics import roc_curve
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
    case_genotype_group: Genotype
    holdout_case_genotype_group: Genotype
    control_genotype_group: Genotype
    holdout_control_genotype_group: Genotype
    filtered_clinical_data: pd.DataFrame

    def __post_init__(self):
        self._check_duplicates()

    def _check_duplicates(self):
        # prepare a dict to hold column names from each dataframe
        df_dict = {
            "case_genotypes": self.case_genotype_group.genotypes,
            "control_genotypes": self.control_genotype_group.genotypes,
            "holdout_case_genotypes": self.holdout_case_genotype_group.genotypes,
            "holdout_control_genotypes": self.holdout_control_genotype_group.genotypes,
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
                dup_dict[df_name] = duplicates

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

        self.case_genotype_group.genotypes = [
            Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
            for geno in self.case_genotype_group.genotypes
        ]
        self.control_genotype_group.genotypes = [
            Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
            for geno in self.control_genotype_group.genotypes
        ]

        if len(self.holdout_case_genotype_group.genotypes) > 0:
            self.holdout_case_genotype_group.genotypes = [
                Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
                for geno in self.holdout_case_genotype_group.genotypes
            ]
        if len(self.holdout_control_genotype_group.genotypes) > 0:
            self.holdout_control_genotype_group.genotypes = [
                Genotype(filtered_genotypes.loc[:, geno.genotype.columns], geno.ids)
                for geno in self.holdout_control_genotype_group.genotypes
            ]


@task()
def recoverPastRuns(modelStack, results=None):
    # TODO debug, refactor remote logging and downloads

    # runsTable = projectTracker.fetch_runs_table().to_pandas()
    # pastRuns = runsTable[
    #     runsTable["bootstrapIteration"] < config["sampling"]["lastIteration"]
    # ]
    # label_args_list = [
    #     ("testLabels", runID) for runID in pastRuns["sys/id"].unique()
    # ]
    # for runID in pastRuns["sys/id"].unique():
    #     download_file(runID, "testLabels", "csv", config)
    #     download_file(runID, "sampleResults", "csv", config)
    #     download_file(runID, "featureImportance/modelCoefficients", "csv", config)
    #     download_file(runID, "featureImportance/shapelyExplanations/average", "csv", config)
    if config["tracking"]["remote"]:
        # get bootstrap runs for model
        # currentRuns = pastRuns.loc[
        #     (pastRuns["bootstrapIteration"] == j)
        #     & (pastRuns["model"] == modelName)
        # ]
        # results[i][j]["trainCount"] = np.around(
        #     currentRuns["nTrain"].unique()[0]
        # )
        # results[i][j]["testCount"] = np.around(
        #     currentRuns["nTest"].unique()[0]
        # )
        # samplesResultsByFold = [
        #     load_fold_dataframe(("sampleResults", runID))
        #     for runID in currentRuns["sys/id"].unique()
        # ]
        # loadedSamples = pd.concat(samplesResultsByFold).set_index(
        #     "id", drop=True
        # )
        # # unique run ID ordering matches label_args_list
        # currentRunIDIndices = np.where(
        #     pastRuns["sys/id"].unique() == currentRuns.loc["sys/id"]
        # )
        # loadedLabels = [
        #     load_fold_dataframe(args)
        #     for k in currentRunIDIndices
        #     for args in label_args_list[k]
        # ]
        # for sampleID in loadedSamples.index:
        #     if sampleID not in results[i]["samples"]:
        #         results[i]["samples"][sampleID] = loadedSamples.loc[
        #             sampleID
        #         ]["probability"].to_numpy()
        #     else:
        #         results[i]["samples"][sampleID] = np.append(
        #             loadedSamples.loc[sampleID][
        #                 "probability"
        #             ].to_numpy(),
        #             results["samples"][sampleID],
        #         )
        #     if sampleID not in results[i]["labels"]:
        #         results[i]["labels"][sampleID] = loadedSamples.loc[
        #             sampleID
        #         ]["label"].unique()[
        #             0
        #         ]  # all labels should be same for sample ID
        # results[i][j]["testLabels"] = loadedLabels
        # results[i][j]["probabilities"] = samplesResultsByFold
        # try:
        #     results[i][j]["globalExplanations"] = [
        #         load_fold_dataframe(
        #             ("featureImportance/modelCoefficients", runID)
        #         )
        #         for runID in currentRuns["sys/id"].unique()
        #     ]
        # except:
        #     pass
        # try:
        #     results[i][j][
        #         "averageShapelyExplanations"
        #     ] = load_fold_dataframe(
        #         ("featureImportance/shapelyExplanations/average", runID)
        #     )
        # except:
        #     pass
        pass

    if not config["tracking"]["remote"]:
        bootstrapFolders = os.listdir(
            f"projects/{config['tracking']['project']}/bootstraps"
        )
        # convert to int
        bootstrapFolders = [int(folder) for folder in bootstrapFolders]
        bootstrapFolders.sort()
        # assert (
        #     max(bootstrapFolders) == config["sampling"]["lastIteration"]
        # )  # TODO automatically determine last iteration by max
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
                            results[i][j]["holdoutCount"] = float(
                                fileName.split("_")[1]
                            )
                        elif "meanAUC" in fileName:
                            results[i][j]["averageTestAUC"] = float(
                                fileName.split("_")[1]
                            )
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
                                    if (
                                        "averageHoldoutShapelyExplanations"
                                        in results[i][j]
                                    ):
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
                    tpr_interpolated = np.interp(
                        results[i][j]["testFPR"], testFPR, testTPR
                    )
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
                                [
                                    id
                                    for idList in results[i][j]["testIDs"]
                                    for id in idList
                                ]
                            )
                        if sampleID in currentIDs:
                            if sampleID not in results[i]["samples"]:
                                results[i]["samples"][sampleID] = np.ravel(
                                    currentProbabilities[
                                        np.where(currentIDs == sampleID)
                                    ]
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
