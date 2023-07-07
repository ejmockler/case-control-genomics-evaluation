import asyncio
from inspect import isclass
import pickle
import os
import sys
import numpy as np
import neptune
import pandas as pd
import ray
import plotly.express as px
import matplotlib
from tqdm import tqdm

matplotlib.use("agg")

from prefect_ray.task_runners import RayTaskRunner
from prefect.task_runners import ConcurrentTaskRunner
from prefect import flow, task
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from tasks.input import (
    fromMultiprocessDict,
    processInputFiles,
    prepareDatasets,
    toMultiprocessDict,
)
from tasks.predict import (
    beginTracking,
    evaluate,
    serializeDataFrame,
    trackResults,
    processSampleResult,
)
from tasks.visualize import (
    plotAUC,
    plotCalibration,
    plotConfusionMatrix,
    plotOptimizer,
    trackVisualizations,
)
from neptune.types import File
from config import config
from models import stack as modelStack

import gc
from joblib import Parallel, delayed


@flow(task_runner=RayTaskRunner())
def classify(
    runNumber,
    model,
    hyperParameterSpace,
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    clinicalData,
    innerCvIterator,
    outerCvIterator,
    results,
    config,
    track=True,
):
    trainIDs = set()
    testIDs = set()
    holdoutIDs = set()
    results[runNumber] = {}
    embedding = prepareDatasets(
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
        verbose=(True if runNumber == 0 else False),
    )

    clinicalIDs = list()
    sampleIndex = (
        embedding["sampleIndex"]
        if len(embedding["holdoutSamples"]) == 0
        else embedding["sampleIndex"] + embedding["holdoutSampleIndex"]
    )

    for id in sampleIndex:
        clinicalIDs.extend(id.split("__"))

    totalSampleCount = len(embedding["samples"])
    caseCount = np.count_nonzero(embedding["labels"])
    print(f"{totalSampleCount} samples\n")
    print(f"{caseCount} cases\n")
    print(f"{totalSampleCount - caseCount} controls\n")
    if len(embedding["holdoutSamples"]) > 0:
        holdoutCaseCount = np.count_nonzero(embedding["holdoutLabels"])
        print(f"{len(embedding['holdoutSamples'])} holdout samples\n")
        print(f"{holdoutCaseCount} holdout cases\n")
        print(
            f"{len(embedding['holdoutSamples']) - holdoutCaseCount} holdout controls\n"
        )

    current = {}
    # check if model is initialized
    if isclass(model):
        if model.__name__ == "TabNetClassifier":
            #  model = model(verbose=False, optimizer_fn=Lion)
            pass
    print(f"Iteration {runNumber+1} with model {model.__class__.__name__}")

    if track:
        runID = beginTracking.submit(
            model, runNumber, embedding, clinicalData, clinicalIDs, config
        )
    gc.collect()

    # outer cross-validation
    crossValIndices = list(
        outerCvIterator.split(embedding["samples"], embedding["labels"])
    )
    current["trainIndices"] = [train for train, _ in crossValIndices]
    current["testIndices"] = [test for _, test in crossValIndices]
    trainIDs.update(
        *[
            np.array(embedding["sampleIndex"])[indices]
            for indices in current["trainIndices"]
        ]
    )
    testIDs.update(
        *[
            np.array(embedding["sampleIndex"])[indices]
            for indices in current["testIndices"]
        ]
    )
    holdoutIDs.update(*np.array(embedding["holdoutSampleIndex"]))

    evaluate_args = [
        (
            embedding["samples"][trainIndices],
            embedding["labels"][trainIndices],
            embedding["samples"][testIndices],
            embedding["labels"][testIndices],
            model,
            embedding,
            hyperParameterSpace,
            innerCvIterator,
            np.array([embedding["sampleIndex"][i] for i in trainIndices]),
            np.array([embedding["sampleIndex"][i] for i in testIndices]),
            embedding["variantIndex"],
            config,
        )
        for trainIndices, testIndices in zip(
            current["trainIndices"], current["testIndices"]
        )
    ]

    # run models sequentially with SHAP to avoid memory issues
    # if config["model"]["calculateShapelyExplanations"]:
    outerCrossValResults = []
    for args in evaluate_args:
        result = evaluate(*args)
        outerCrossValResults.append(result)
        gc.collect()
    outerCrossValResults = list(map(list, zip(*outerCrossValResults)))
    # else:
    # outerCrossValResults = zip(
    #     *Parallel(n_jobs=-1)(delayed(evaluate)(*args) for args in evaluate_args)
    # )

    # TODO implement object to structure these results
    resultNames = [
        "globalExplanations",
        "localExplanations",
        "holdoutLocalExplanations",
        "probabilities",
        "holdoutProbabilities",
        "predictions",
        "holdoutPredictions",
        "testLabels",
        "trainLabels",
        "holdoutLabels",
        "trainIDs",
        "testIDs",
        "holdoutIDs",
        "fittedOptimizer",
        "shapExplainer",
        "shapMasker",
    ]
    current = {
        **current,
        **{name: result for name, result in zip(resultNames, outerCrossValResults)},
    }

    current["testAUC"] = [
        roc_auc_score(
            labels,
            (probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities),
        )
        for labels, probabilities in zip(
            current["testLabels"], current["probabilities"]
        )
    ]
    current["averageTestAUC"] = np.mean(current["testAUC"])

    current["testFPR"] = np.linspace(0, 1, 100)  # Define common set of FPR values
    current["testTPR"] = []
    for labels, probabilities in zip(current["testLabels"], current["probabilities"]):
        fpr, tpr, thresholds = roc_curve(
            labels,
            (probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities),
        )

        # Interpolate TPR at common FPR values
        tpr_interpolated = np.interp(current["testFPR"], fpr, tpr)
        current["testTPR"].append(tpr_interpolated)
    current["averageTestTPR"] = np.mean(current["testTPR"], axis=0)

    if len(embedding["holdoutSamples"]) > 0:
        current["holdoutAUC"] = [
            roc_auc_score(
                labels,
                (
                    probabilities[:, 1]
                    if len(probabilities.shape) > 1
                    else probabilities
                ),
            )
            for labels, probabilities in zip(
                current["holdoutLabels"], current["holdoutProbabilities"]
            )
        ]
        current["averageHoldoutAUC"] = np.mean(current["holdoutAUC"])
        current["holdoutFPR"] = np.linspace(
            0, 1, 100
        )  # Define common set of FPR values
        current["holdoutTPR"] = []
        for labels, probabilities in zip(
            current["holdoutLabels"], current["holdoutProbabilities"]
        ):
            fpr, tpr, thresholds = roc_curve(
                labels,
                (
                    probabilities[:, 1]
                    if len(probabilities.shape) > 1
                    else probabilities
                ),
            )
            tpr_interpolated = np.interp(current["holdoutFPR"], fpr, tpr)
            current["holdoutTPR"].append(tpr_interpolated)
        current["averageHoldoutTPR"] = np.mean(current["holdoutTPR"], axis=0)

    if config["model"]["calculateShapelyExplanations"]:
        current["averageShapelyExplanations"] = pd.DataFrame.from_dict(
            {
                "feature_name": [
                    name for name in current["localExplanations"][0].feature_names
                ],
                "value": [
                    np.mean(
                        np.hstack(
                            [
                                np.mean(localExplanations.values[:, featureIndex])
                                for localExplanations in current["localExplanations"]
                            ]
                        )
                    )
                    for featureIndex in range(
                        len(current["localExplanations"][0].feature_names)
                    )
                ],
                "standard_deviation": [
                    np.mean(
                        np.hstack(
                            [
                                np.std(localExplanations.values[:, featureIndex])
                                for localExplanations in current["localExplanations"]
                            ]
                        )
                    )
                    for featureIndex in range(
                        len(current["localExplanations"][0].feature_names)
                    )
                ],
            },
            dtype=object,
        ).set_index("feature_name")
        if len(embedding["holdoutSamples"]) > 0:
            current["averageHoldoutShapelyExplanations"] = pd.DataFrame.from_dict(
                {
                    "feature_name": [
                        name
                        for name in current["holdoutLocalExplanations"][0].feature_names
                    ],
                    "value": [
                        np.mean(
                            np.hstack(
                                [
                                    np.mean(localExplanations.values[:, featureIndex])
                                    for localExplanations in current[
                                        "holdoutLocalExplanations"
                                    ]
                                ]
                            )
                        )
                        for featureIndex in range(
                            len(current["holdoutLocalExplanations"][0].feature_names)
                        )
                    ],
                    "standard_deviation": [
                        np.mean(
                            np.hstack(
                                [
                                    np.std(localExplanations.values[:, featureIndex])
                                    for localExplanations in current[
                                        "holdoutLocalExplanations"
                                    ]
                                ]
                            )
                        )
                        for featureIndex in range(
                            len(current["holdoutLocalExplanations"][0].feature_names)
                        )
                    ],
                },
                dtype=object,
            ).set_index("feature_name")
    gc.collect()

    if current["globalExplanations"][0] is not None:
        df = pd.concat(current["globalExplanations"]).reset_index()
        # calculate mean
        mean_df = df.groupby("feature_name").mean()
        # calculate standard deviation
        std_df = df.groupby("feature_name").std()
        # rename std_df columns
        std_df.columns = "stdDev_" + std_df.columns
        # join mean and std dataframe
        averageGlobalExplanations = pd.concat([mean_df, std_df], axis=1)
        current["averageGlobalExplanations"] = averageGlobalExplanations

    caseAccuracy = np.mean(
        [
            np.divide(
                np.sum((predictions[labels == 1] == 1).astype(int)),
                np.sum((labels == 1).astype(int)),
            )
            for predictions, labels in zip(
                current["predictions"], current["testLabels"]
            )
        ]
    )
    controlAccuracy = np.mean(
        [
            np.divide(
                np.sum((predictions[labels == 0] == 0).astype(int)),
                np.sum((labels == 0).astype(int)),
            )
            for predictions, labels in zip(
                current["predictions"], current["testLabels"]
            )
        ]
    )

    results[runNumber] = current

    if track:
        resultsFuture = trackResults.submit(runID.result(), current, config)

        # plot AUC & hyperparameter convergence
        plotSubtitle = f"""
            {config["tracking"]["name"]}, {embedding["samples"].shape[1]} variants
            Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
            
            {np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(caseAccuracy)} accuracy, {len(embedding['labels']) - np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(controlAccuracy)} accuracy
            {int(np.around(np.mean([len(indices) for indices in current["trainIndices"]])))}±1 train, {int(np.around(np.mean([len(indices) for indices in current["testIndices"]])))}±1 test samples per x-val fold"""

        if len(current["holdoutLabels"]) > 0:
            holdoutCaseAccuracy = np.mean(
                [
                    np.divide(
                        np.sum((predictions[labels == 1] == 1).astype(int)),
                        np.sum((labels == 1).astype(int)),
                    )
                    for predictions, labels in zip(
                        current["holdoutPredictions"],
                        current["holdoutLabels"],
                    )
                ]
            )
            holdoutControlAccuracy = np.mean(
                [
                    np.divide(
                        np.sum((predictions[labels == 0] == 0).astype(int)),
                        np.sum((labels == 0).astype(int)),
                    )
                    for predictions, labels in zip(
                        current["holdoutPredictions"],
                        current["holdoutLabels"],
                    )
                ]
            )
            holdoutPlotSubtitle = f"""
                {config["tracking"]["name"]}, {embedding["samples"].shape[1]} variants
                Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
                
                Ethnically variable holdout
                {np.count_nonzero(embedding['holdoutLabels'])} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(holdoutCaseAccuracy)} accuracy, {len(embedding['holdoutLabels']) - np.count_nonzero(embedding['holdoutLabels'])} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(holdoutControlAccuracy)} accuracy
                {int(np.around(np.mean([len(indices) for indices in current["trainIndices"]])))}±1 train, {int(np.around(np.mean([len(indices) for indices in current["testIndices"]])))}±1 test samples per x-val fold"""
            holdoutVisualizationFuture = trackVisualizations.submit(
                runID.result(),
                holdoutPlotSubtitle,
                model.__class__.__name__,
                current,
                holdout=True,
                config=config,
            )
            gc.collect()

        visualizationFuture = trackVisualizations.submit(
            runID.result(),
            plotSubtitle,
            model.__class__.__name__,
            current,
            config=config,
        )

        visualizationFuture.wait()
        if len(current["holdoutLabels"]) > 0:
            holdoutVisualizationFuture.wait()
        resultsFuture.wait()

    results[runNumber]["testCount"] = len(trainIDs)
    results[runNumber]["trainCount"] = len(testIDs)
    results[runNumber]["holdoutCount"] = len(holdoutIDs)

    return results


@flow()
def bootstrap(
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    clinicalData,
    model,
    hyperParameterSpace,
    innerCvIterator,
    outerCvIterator,
    config,
    track=True,
):
    gc.collect()
    results = {}
    results["samples"] = {}
    results["labels"] = {}
    results["model"] = model.__class__.__name__

    # parallelize with workflow engine in cluster environment
    for runNumber in range(
        config["sampling"]["lastIteration"],
        config["sampling"]["bootstrapIterations"],
    ):
        # update results for every bootstrap iteration
        results = classify(
            runNumber,
            model,
            hyperParameterSpace,
            caseGenotypes,
            controlGenotypes,
            holdoutCaseGenotypes,
            holdoutControlGenotypes,
            clinicalData,
            innerCvIterator,
            outerCvIterator,
            results,
            config,
            track,
        )
        gc.collect()

    return results


@task()
def download_file(run_id, field="sampleResults", extension="csv", config=config):
    path = f"./{field}/{run_id}.{extension}"
    if not os.path.exists(field):
        os.makedirs(field, exist_ok=True)
    if os.path.isfile(path):
        return
    run = neptune.init_run(
        with_id=run_id,
        project=config["entity"] + "/" + config["project"],
        api_token=config["neptuneApiToken"],
    )
    try:
        if field == "globalFeatureImportance" or field == "testLabels":
            for i in range(11):
                path = f"./{field}/{run_id}_{i}.{extension}"
                run[f"{field}/{i}"].download(destination=path)
            averageResultsPath = f"./{field}/{run_id}_average.{extension}"
            run[f"{field}/average"].download(destination=averageResultsPath)
        else:
            run[field].download(destination=path)
    except:
        pass
    run.stop()


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
    for sampleID in sampleResults:
        # add accuracy
        sampleResults[sampleID].append(
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

    sampleResultsDataFrame = pd.DataFrame.from_dict(
        sampleResults, orient="index", columns=["label", "probability", "accuracy"]
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
    np.set_printoptions(threshold=np.inf)
    sampleResultsDataFrame.index.name = "id"
    return sampleResultsDataFrame


@flow()
def main(
    config=config,
    caseGenotypes=None,
    caseIDs=None,
    holdoutCaseGenotypes=None,
    holdoutCaseIDs=None,
    controlGenotypes=None,
    controlIDs=None,
    holdoutControlGenotypes=None,
    holdoutControlIDs=None,
    clinicalData=None,
):
    if caseGenotypes is None:
        (
            caseGenotypes,
            caseIDs,
            holdoutCaseGenotypes,
            holdoutCaseIDs,
            controlGenotypes,
            controlIDs,
            holdoutControlGenotypes,
            holdoutControlIDs,
            clinicalData,
        ) = processInputFiles(config)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator
    if config["tracking"]["remote"]:
        projectTracker = neptune.init_project(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            api_token=config["tracking"]["token"],
        )

    bootstrap_args = [
        (
            caseGenotypes,
            controlGenotypes,
            holdoutCaseGenotypes,
            holdoutControlGenotypes,
            clinicalData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            outerCvIterator,
            config,
        )
        for model, hyperParameterSpace in list(modelStack.items())
    ]

    # results = []
    # for args in bootstrap_args:
    #     results.append(bootstrap(*args))

    results = Parallel(n_jobs=-1)(delayed(bootstrap)(*args) for args in bootstrap_args)

    testLabelsProbabilitiesByModelName = dict()
    holdoutLabelsProbabilitiesByModelName = dict()
    tprFprAucByInstance = dict()
    holdoutTprFprAucByInstance = dict()
    variantCount = 0
    lastVariantCount = 0
    sampleResults = {}

    if config["sampling"]["lastIteration"] > 0:
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
                                results[i][j]["testCount"] = float(
                                    fileName.split("_")[1]
                                )
                            elif "trainCount" in fileName:
                                results[i][j]["trainCount"] = float(
                                    fileName.split("_")[1]
                                )
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
                                        .loc[
                                            np.hstack(results[i][j]["testIDs"]).tolist()
                                        ]
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
                                            key=lambda fileName: int(
                                                fileName.split(".")[0]
                                            )
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
                                sampleID in idList
                                for idList in results[i][j]["holdoutIDs"]
                            ):
                                currentProbabilities = results[i][j][
                                    "holdoutProbabilities"
                                ]
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

    for i in range(len(modelStack)):
        modelResult = results[i]
        modelName = modelResult["model"]
        sampleResults = serializeBootstrapResults(modelResult, sampleResults)
        if modelName not in testLabelsProbabilitiesByModelName:
            testLabelsProbabilitiesByModelName[modelName] = [[], []]
        if modelName not in holdoutLabelsProbabilitiesByModelName:
            holdoutLabelsProbabilitiesByModelName[modelName] = [[], []]
        if modelName not in tprFprAucByInstance:
            tprFprAucByInstance[modelName] = [[], [], 0]
        if modelName not in holdoutTprFprAucByInstance:
            holdoutTprFprAucByInstance[modelName] = [[], [], 0]
        globalExplanationsList = []
        for j in range(config["sampling"]["bootstrapIterations"]):
            bootstrapResult = modelResult[j]
            testFlattenedCrossValIndex = 0
            holdoutFlattenedCrossValIndex = 0
            for k in range(config["sampling"]["crossValIterations"]):
                # append labels
                testLabelsProbabilitiesByModelName[modelName][0] = np.hstack(
                    (
                        testLabelsProbabilitiesByModelName[modelName][0],
                        *[foldLabel for foldLabel in bootstrapResult["testLabels"][k]],
                    )
                )
                holdoutLabelsProbabilitiesByModelName[modelName][0] = np.hstack(
                    (
                        holdoutLabelsProbabilitiesByModelName[modelName][0],
                        *[
                            foldLabel
                            for foldLabel in bootstrapResult["holdoutLabels"][k]
                        ],
                    )
                )
                # append probabilities
                testLabelsProbabilitiesByModelName[modelName][1] = np.hstack(
                    [
                        testLabelsProbabilitiesByModelName[modelName][1],
                        np.array(bootstrapResult["probabilities"][k])[:, 1]
                        if len(bootstrapResult["probabilities"][k][0].shape) >= 1
                        else np.ravel(
                            bootstrapResult["probabilities"][
                                testFlattenedCrossValIndex : testFlattenedCrossValIndex
                                + len(bootstrapResult["testLabels"][k])
                            ]
                        ),  # probabilities from recovered bootstrap runs are 1D
                    ]
                )
                testFlattenedCrossValIndex += len(bootstrapResult["testLabels"][k])
                holdoutLabelsProbabilitiesByModelName[modelName][1] = np.hstack(
                    [
                        holdoutLabelsProbabilitiesByModelName[modelName][1],
                        np.array(bootstrapResult["holdoutProbabilities"][k])[:, 1]
                        if len(bootstrapResult["holdoutProbabilities"][k][0].shape) >= 1
                        else np.ravel(
                            bootstrapResult["holdoutProbabilities"][
                                holdoutFlattenedCrossValIndex : holdoutFlattenedCrossValIndex
                                + len(bootstrapResult["holdoutLabels"][k])
                            ]
                        ),
                    ]
                )
                holdoutFlattenedCrossValIndex += len(
                    bootstrapResult["holdoutLabels"][k]
                )
            if j == 0:
                tprFprAucByInstance[modelName][0] = [bootstrapResult["averageTestTPR"]]
                tprFprAucByInstance[modelName][1] = bootstrapResult["testFPR"]
                tprFprAucByInstance[modelName][2] = [bootstrapResult["averageTestAUC"]]
                if "averageHoldoutAUC" in bootstrapResult:
                    holdoutTprFprAucByInstance[modelName][0] = [
                        bootstrapResult["averageHoldoutTPR"]
                    ]
                    holdoutTprFprAucByInstance[modelName][1] = bootstrapResult[
                        "holdoutFPR"
                    ]
                    holdoutTprFprAucByInstance[modelName][2] = [
                        bootstrapResult["averageHoldoutAUC"]
                    ]
            else:
                tprFprAucByInstance[modelName][0] = np.concatenate(
                    [
                        tprFprAucByInstance[modelName][0],
                        [bootstrapResult["averageTestTPR"]],
                    ]
                )
                tprFprAucByInstance[modelName][2] = np.concatenate(
                    [
                        tprFprAucByInstance[modelName][2],
                        [bootstrapResult["averageTestAUC"]],
                    ]
                )
                if "averageHoldoutAUC" in bootstrapResult:
                    holdoutTprFprAucByInstance[modelName][0] = np.concatenate(
                        [
                            holdoutTprFprAucByInstance[modelName][0],
                            [bootstrapResult["averageHoldoutTPR"]],
                        ]
                    )
                    holdoutTprFprAucByInstance[modelName][2] = np.concatenate(
                        [
                            holdoutTprFprAucByInstance[modelName][2],
                            [bootstrapResult["averageHoldoutAUC"]],
                        ]
                    )

            if config["model"]["calculateShapelyExplanations"]:
                averageShapelyExplanationsDataFrame = (
                    pd.concat(
                        [
                            modelResult[j]["averageShapelyExplanations"]
                            for j in range(config["sampling"]["bootstrapIterations"])
                        ]
                    )
                    .reset_index()
                    .groupby("feature_name")
                    .mean()
                )
                if "averageHoldoutShapelyExplanations" in modelResult[j]:
                    averageHoldoutShapelyExplanationsDataFrame = (
                        pd.concat(
                            [
                                modelResult[j]["averageHoldoutShapelyExplanations"]
                                for j in range(
                                    config["sampling"]["bootstrapIterations"]
                                )
                            ]
                        )
                        .reset_index()
                        .groupby("feature_name")
                        .mean()
                    )
                if config["tracking"]["remote"]:
                    projectTracker["averageShapelyExplanations"].upload(
                        serializeDataFrame(averageShapelyExplanationsDataFrame)
                    )
                    if "averageHoldoutShapelyExplanations" in modelResult[j]:
                        projectTracker[
                            "averageHoldoutShapelyExplanationsDataFrame"
                        ].upload(
                            serializeDataFrame(
                                averageHoldoutShapelyExplanationsDataFrame
                            )
                        )
                else:
                    os.makedirs(
                        f"projects/{config['tracking']['project']}/averageShapelyExplanations/",
                        exist_ok=True,
                    )
                    averageShapelyExplanationsDataFrame.to_csv(
                        f"projects/{config['tracking']['project']}/averageShapelyExplanations.csv"
                    )
                    if "averageHoldoutShapelyExplanations" in modelResult[j]:
                        averageHoldoutShapelyExplanationsDataFrame.to_csv(
                            f"projects/{config['tracking']['project']}/averageHoldoutShapelyExplanations.csv"
                        )
            if "globalExplanations" in bootstrapResult and isinstance(
                bootstrapResult["globalExplanations"][0], pd.DataFrame
            ):
                variantCount = bootstrapResult["globalExplanations"][0].shape[0]
                assert lastVariantCount == variantCount or lastVariantCount == 0
                lastVariantCount = variantCount

                globalExplanationsList += bootstrapResult["globalExplanations"]

        if globalExplanationsList:
            averageGlobalExplanationsDataFrame = (
                pd.concat(globalExplanationsList)
                .reset_index()
                .groupby("feature_name")
                .mean()
            )
            if config["tracking"]["remote"]:
                projectTracker[f"averageModelCoefficients/{modelName}"].upload(
                    serializeDataFrame(averageGlobalExplanationsDataFrame)
                )
            else:
                os.makedirs(
                    f"projects/{config['tracking']['project']}/averageModelCoefficients/",
                    exist_ok=True,
                )
                averageGlobalExplanationsDataFrame.to_csv(
                    f"projects/{config['tracking']['project']}/averageModelCoefficients/{modelName}.csv"
                )

        # Calculate mean over bootstraps (axis=0) for each TPR value
        tprFprAucByInstance[modelName][0] = np.mean(
            tprFprAucByInstance[modelName][0], axis=0
        )
        # Same for AUC
        tprFprAucByInstance[modelName][2] = np.mean(
            tprFprAucByInstance[modelName][2], axis=0
        )
        if "averageHoldoutAUC" in bootstrapResult:
            holdoutTprFprAucByInstance[modelName][0] = np.mean(
                holdoutTprFprAucByInstance[modelName][0],
                axis=0,
            )
            holdoutTprFprAucByInstance[modelName][2] = np.mean(
                holdoutTprFprAucByInstance[modelName][2],
                axis=0,
            )

    sampleResultsDataFrame = serializeResultsDataframe(sampleResults)
    sampleResultsDataFrame["probability"] = sampleResultsDataFrame["probability"].map(
        lambda x: np.array2string(np.array(x), separator=",")
    )

    if config["tracking"]["remote"]:
        projectTracker["sampleResults"].upload(
            serializeDataFrame(sampleResultsDataFrame)
        )
        projectTracker["sampleResultsObject"].upload(
            File.as_pickle(sampleResultsDataFrame)
        )
    else:
        sampleResultsDataFrame.to_csv(
            f"projects/{config['tracking']['project']}/sampleResults.csv"
        )

    seenCases = [
        id for id in sampleResultsDataFrame.index if id in caseGenotypes.columns
    ]
    seenControls = [
        id for id in sampleResultsDataFrame.index if id in controlGenotypes.columns
    ]
    seenHoldoutCases = [
        id for id in sampleResultsDataFrame.index if id in holdoutCaseGenotypes.columns
    ]
    seenHoldoutControls = [
        id
        for id in sampleResultsDataFrame.index
        if id in holdoutControlGenotypes.columns
    ]

    caseAccuracy = sampleResultsDataFrame.loc[
        sampleResultsDataFrame.index.isin([*seenCases])
    ]["accuracy"].mean()
    controlAccuracy = sampleResultsDataFrame.loc[
        sampleResultsDataFrame.index.isin([*seenControls])
    ]["accuracy"].mean()
    holdoutCaseAccuracy = sampleResultsDataFrame.loc[
        sampleResultsDataFrame.index.isin([*seenHoldoutCases])
    ]["accuracy"].mean()
    holdoutControlAccuracy = sampleResultsDataFrame.loc[
        sampleResultsDataFrame.index.isin([*seenHoldoutControls])
    ]["accuracy"].mean()

    bootstrapTrainCount = int(
        np.around(
            np.mean(
                [
                    float(modelResult[j]["trainCount"])
                    for j in range(config["sampling"]["bootstrapIterations"])
                ]
            )
        )
    )
    bootstrapTestCount = int(
        np.around(
            np.mean(
                [
                    float(modelResult[j]["testCount"])
                    for j in range(config["sampling"]["bootstrapIterations"])
                ]
            )
        )
    )

    bootstrapHoldoutCount = int(
        np.around(
            np.mean(
                [
                    float(modelResult[j]["holdoutCount"])
                    for j in range(config["sampling"]["bootstrapIterations"])
                ]
            )
        )
    )

    plotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
        
        {config["tracking"]["name"]}, {variantCount} variants
        Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

        {len(seenCases)} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(caseAccuracy)} accuracy, {len(seenControls)} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(controlAccuracy)} accuracy
        {bootstrapTrainCount}±1 train, {bootstrapTestCount}±1 test samples per bootstrap iteration"""

    accuracyHistogram = px.histogram(
        sampleResultsDataFrame.loc[
            ~sampleResultsDataFrame.index.isin(
                [*seenHoldoutCases, *seenHoldoutControls]
            )
        ],
        x="accuracy",
        color="label",
        pattern_shape="label",
        range_x=[0, 1],
        title=f"""Mean sample accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
    )
    probabilityHistogram = px.histogram(
        sampleResultsDataFrame.loc[
            ~sampleResultsDataFrame.index.isin(
                [*seenHoldoutCases, *seenHoldoutControls]
            )
        ],
        x="meanProbability",
        color="label",
        title=f"""Mean sample probability, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
    )
    aucPlot = plotAUC(
        f"""
            Receiver Operating Characteristic (ROC) Curve
            {plotSubtitle}
            """,
        tprFprAucByInstance=tprFprAucByInstance,
        config=config,
    )
    calibrationPlot = plotCalibration(
        f"""
            Calibration Curve
            {plotSubtitle}
            """,
        testLabelsProbabilitiesByModelName,
        config=config,
    )
    confusionMatrixInstanceList, averageConfusionMatrix = plotConfusionMatrix(
        f"""
            Confusion Matrix
            {plotSubtitle}
            """,
        testLabelsProbabilitiesByModelName,
        config=config,
    )
    if config["model"]["hyperparameterOptimization"]:
        try:
            convergencePlot = plotOptimizer(
                f"""
                    Convergence Plot
                    {plotSubtitle}
                    """,
                {
                    model.__class__.__name__: [
                        result
                        for j in range(config["sampling"]["bootstrapIterations"])
                        for foldOptimizer in results[i][j]["fittedOptimizer"]
                        for result in foldOptimizer.optimizer_results_
                    ]
                    for i, model in enumerate(modelStack)
                },
            )
        except:
            print("Convergence plot data unavailable!", file=sys.stderr)
            convergencePlot = None

    if bootstrapHoldoutCount > 0:
        holdoutPlotSubtitle = f"""
            {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
            {config["tracking"]["name"]}, {variantCount} variants
            Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

            Ethnically variable holdout
            {len(seenHoldoutCases)} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(holdoutCaseAccuracy)} accuracy, {len(seenHoldoutControls)} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(holdoutControlAccuracy)} accuracy
            {bootstrapHoldoutCount} ethnically-matched samples"""

        holdoutAccuracyHistogram = px.histogram(
            sampleResultsDataFrame.loc[
                sampleResultsDataFrame.index.isin(
                    [*seenHoldoutCases, *seenHoldoutControls]
                )
            ],
            x="accuracy",
            color="label",
            pattern_shape="label",
            range_x=[0, 1],
            title=f"""Mean holdout accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
        )
        holdoutProbabilityHistogram = px.histogram(
            sampleResultsDataFrame.loc[
                sampleResultsDataFrame.index.isin(
                    [*seenHoldoutCases, *seenHoldoutControls]
                )
            ],
            x="meanProbability",
            color="label",
            title=f"""Mean holdout probability, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
        )
        holdoutAucPlot = plotAUC(
            f"""
                Receiver Operating Characteristic (ROC) Curve
                {holdoutPlotSubtitle}
                """,
            tprFprAucByInstance=holdoutTprFprAucByInstance,
            config=config,
        )
        holdoutCalibrationPlot = plotCalibration(
            f"""
                Calibration Curve
                {holdoutPlotSubtitle}
                """,
            holdoutLabelsProbabilitiesByModelName,
            config=config,
        )
        (
            holdoutConfusionMatrixInstanceList,
            averageHoldoutConfusionMatrix,
        ) = plotConfusionMatrix(
            f"""
                Confusion Matrix
                {plotSubtitle}
                """,
            holdoutLabelsProbabilitiesByModelName,
            config=config,
        )

    if config["tracking"]["remote"]:
        projectTracker["sampleAccuracyPlot"].upload(accuracyHistogram)
        projectTracker["sampleProbabilityPlot"].upload(probabilityHistogram)
        projectTracker["aucPlot"].upload(aucPlot)
        for name, confusionMatrix in zip(
            list(testLabelsProbabilitiesByModelName.keys()), confusionMatrixInstanceList
        ):
            projectTracker[f"confusionMatrix/{name}"].upload(confusionMatrix)
        projectTracker["averageConfusionMatrix"].upload(averageConfusionMatrix)

        if config["model"]["hyperparameterOptimization"]:
            projectTracker["calibrationPlot"].upload(File.as_image(calibrationPlot))

        if config["model"]["hyperparameterOptimization"]:
            if convergencePlot is not None:
                projectTracker["convergencePlot"].upload(File.as_image(convergencePlot))

        if bootstrapHoldoutCount > 0:
            projectTracker["sampleAccuracyPlotHoldout"].upload(holdoutAccuracyHistogram)
            projectTracker["sampleProbabilityPlotHoldout"].upload(
                holdoutProbabilityHistogram
            )
            projectTracker["aucPlotHoldout"].upload(holdoutAucPlot)
            for i, confusionMatrix in enumerate(holdoutConfusionMatrixInstanceList):
                projectTracker[f"confusionMatrixHoldout/{i+1}"].upload(confusionMatrix)
            projectTracker["averageConfusionMatrixHoldout"].upload(
                averageHoldoutConfusionMatrix
            )
            projectTracker["calibrationPlotHoldout"].upload(
                File.as_image(holdoutCalibrationPlot)
            )

        projectTracker.stop()
    else:
        accuracyHistogram.write_html(
            f"projects/{config['tracking']['project']}/accuracyPlot.html"
        )
        probabilityHistogram.write_html(
            f"projects/{config['tracking']['project']}/probabilityPlot.html"
        )
        aucPlot.savefig(
            f"projects/{config['tracking']['project']}/aucPlot.svg", bbox_inches="tight"
        )
        aucPlot.savefig(
            f"projects/{config['tracking']['project']}/aucPlot.png", bbox_inches="tight"
        )
        confusionMatrixPath = (
            f"projects/{config['tracking']['project']}/confusionMatrix"
        )
        os.makedirs(confusionMatrixPath, exist_ok=True)
        for name, confusionMatrix in zip(
            list(testLabelsProbabilitiesByModelName.keys()), confusionMatrixInstanceList
        ):
            confusionMatrix.savefig(
                f"{confusionMatrixPath}/{name}.svg",
                bbox_inches="tight",
            )
        averageConfusionMatrix.savefig(
            f"projects/{config['tracking']['project']}/averageConfusionMatrix.svg",
            bbox_inches="tight",
        )
        averageConfusionMatrix.savefig(
            f"projects/{config['tracking']['project']}/averageConfusionMatrix.png",
            bbox_inches="tight",
        )

        calibrationPlot.savefig(
            f"projects/{config['tracking']['project']}/calibrationPlot.svg",
            bbox_inches="tight",
        )
        calibrationPlot.savefig(
            f"projects/{config['tracking']['project']}/calibrationPlot.png",
            bbox_inches="tight",
        )
        if config["model"]["hyperparameterOptimization"]:
            if convergencePlot is not None:
                convergencePlot.savefig(
                    f"projects/{config['tracking']['project']}/convergencePlot.svg",
                    bbox_inches="tight",
                )
                convergencePlot.savefig(
                    f"projects/{config['tracking']['project']}/convergencePlot.png",
                    bbox_inches="tight",
                )

        if bootstrapHoldoutCount > 0:
            holdoutAccuracyHistogram.write_html(
                f"projects/{config['tracking']['project']}/holdoutAccuracyPlot.html"
            )
            holdoutProbabilityHistogram.write_html(
                f"projects/{config['tracking']['project']}/holdoutProbabilityPlot.html"
            )
            holdoutAucPlot.savefig(
                f"projects/{config['tracking']['project']}/holdoutAucPlot.svg",
                bbox_inches="tight",
            )
            holdoutAucPlot.savefig(
                f"projects/{config['tracking']['project']}/holdoutAucPlot.png",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"projects/{config['tracking']['project']}/holdoutCalibrationPlot.svg",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"projects/{config['tracking']['project']}/holdoutCalibrationPlot.png",
                bbox_inches="tight",
            )
            confusionMatrixPath = (
                f"projects/{config['tracking']['project']}/confusionMatrix/holdout"
            )
            os.makedirs(confusionMatrixPath, exist_ok=True)
            for name, confusionMatrix in zip(
                list(testLabelsProbabilitiesByModelName.keys()),
                holdoutConfusionMatrixInstanceList,
            ):
                confusionMatrix.savefig(
                    f"{confusionMatrixPath}/{name}.svg", bbox_inches="tight"
                )
            averageHoldoutConfusionMatrix.savefig(
                f"projects/{config['tracking']['project']}/averageConfusionMatrixHoldout.svg",
                bbox_inches="tight",
            )
            averageHoldoutConfusionMatrix.savefig(
                f"projects/{config['tracking']['project']}/averageConfusionMatrixHoldout.png",
                bbox_inches="tight",
            )

    return (
        results,
        clinicalData,
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
    )


async def remove_all_flows():
    from prefect.client import get_client

    orion_client = get_client()
    flows = await orion_client.read_flows()
    for flow in flows:
        flow_id = flow.id
        print(f"Deleting flow: {flow.name}, {flow_id}")
        await orion_client._client.delete(f"/flows/{flow_id}")
        print(f"Flow with UUID {flow_id} deleted")


if __name__ == "__main__":
    ray.shutdown()

    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())

    results = asyncio.run(main())
    pickle.dump(
        results,
        open(
            f"projects/{config['tracking']['project']}/results_{config['tracking']['project']}.pkl",
            "wb",
        ),
    )
