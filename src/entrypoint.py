import asyncio
from inspect import isclass
import pickle
import os
import traceback
import numpy as np
import neptune
import pandas as pd
import ray
import plotly.express as px
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import shap

from prefect_ray.task_runners import RayTaskRunner
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.metrics import roc_auc_score
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold
from input import processInputFiles
from predict import (
    beginTracking,
    getFeatureImportances,
    optimizeHyperparameters,
    plotAUC,
    plotCalibration,
    plotOptimizer,
    prepareDatasets,
    serializeDataFrame,
    trackResults,
)
from neptune.types import File
from config import config

import gc


# limit subflow concurrency
async def limited_gather(tasks, limit):
    if limit == "unlimited":
        return await asyncio.gather(*tasks)
    semaphore = asyncio.Semaphore(limit)

    async def bound_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(bound_task(task) for task in tasks))


# parallel task runner patch https://github.com/PrefectHQ/prefect/issues/7319
# TODO build task runners only
async def build_subflow(name, args):
    if name == "classify":

        @flow(task_runner=RayTaskRunner())
        async def classify(
            caseGenotypes,
            controlGenotypes,
            clinicalData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            outerCvIterator,
        ):
            gc.collect()
            results = {}
            results["model"] = model.__class__.__name__

            bootstrap_tasks = [
                build_subflow(
                    "bootstrap",
                    (
                        model,
                        hyperParameterSpace,
                        caseGenotypes,
                        controlGenotypes,
                        clinicalData,
                        outerCvIterator,
                        innerCvIterator,
                        runNumber,
                    ),
                )
                for runNumber in range(
                    config["sampling"]["lastIteration"],
                    config["sampling"]["bootstrapIterations"],
                )
            ]

            results["bootstraps"] = await limited_gather(
                bootstrap_tasks, config["sampling"]["concurrencyLimit"]
            )

            results["samples"] = {}
            results["labels"] = {}
            for runNumber in range(
                config["sampling"]["lastIteration"],
                config["sampling"]["bootstrapIterations"],
            ):
                # record sample metrics
                for fold in range(config["sampling"]["crossValIterations"]):
                    for j, sampleID in enumerate(
                        results["bootstraps"][runNumber]["testIDs"][fold]
                    ):
                        try:
                            results["samples"][sampleID].append(
                                results["bootstraps"][runNumber]["probabilities"][fold][
                                    j
                                ]
                            )
                        except KeyError:
                            results["samples"][sampleID] = [
                                results["bootstraps"][runNumber]["probabilities"][fold][
                                    j
                                ]
                            ]
                        finally:
                            results["labels"][sampleID] = results["bootstraps"][
                                runNumber
                            ]["testLabels"][fold][j]
            return results

        return await classify(*args)

    if name == "bootstrap":

        @flow(task_runner=RayTaskRunner())
        async def bootstrap(
            model,
            hyperParameterSpace,
            caseGenotypes,
            controlGenotypes,
            clinicalData,
            outerCvIterator,
            innerCvIterator,
            runNumber,
        ):
            trainIDs = set()
            testIDs = set()
            results = {}
            embedding = prepareDatasets(
                caseGenotypes,
                controlGenotypes,
                verbose=(True if runNumber == 0 else False),
            )
            deserializedIDs = list()
            for id in embedding["sampleIndex"]:
                deserializedIDs.extend(id.split("__"))
            totalSampleCount = len(embedding["samples"])
            caseCount = np.count_nonzero(embedding["labels"])
            print(f"{totalSampleCount} samples\n")
            print(f"{caseCount} cases\n")
            print(f"{totalSampleCount - caseCount} controls\n")
            current = {}
            # check if model is initialized
            if isclass(model):
                if model.__name__ == "TabNetClassifier":
                    #  model = model(verbose=False, optimizer_fn=Lion)
                    pass
            print(f"Iteration {runNumber+1} with model {model.__class__.__name__}")
            runID = await beginTracking.submit(
                model, runNumber, embedding, clinicalData, deserializedIDs
            )
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
            outerCrossValResults = zip(
                *await asyncio.gather(
                    *[
                        build_subflow(
                            "evaluate",
                            (
                                trainIndices,
                                testIndices,
                                model,
                                embedding["labels"],
                                embedding["samples"],
                                embedding["variantIndex"],
                                embedding["sampleIndex"],
                                hyperParameterSpace,
                                innerCvIterator,
                            ),
                        )
                        for trainIndices, testIndices in zip(
                            current["trainIndices"], current["testIndices"]
                        )
                    ]
                )
            )
            resultNames = [
                "globalExplanations",
                "localExplanations",
                "probabilities",
                "predictions",
                "testLabels",
                "trainLabels",
                "trainIDs",
                "testIDs",
                "fittedOptimizers",
                "shapExplainers",
                "shapMaskers",
            ]
            current = {
                **current,
                **{
                    name: result
                    for name, result in zip(resultNames, outerCrossValResults)
                },
            }
            current["testAUC"] = [
                roc_auc_score(
                    labels,
                    (
                        probabilities[:, 1]
                        if len(probabilities.shape) > 1
                        else probabilities
                    ),
                )
                for labels, probabilities in zip(
                    current["testLabels"], current["probabilities"]
                )
            ]

            if config["model"]["calculateShapelyExplanations"]:
                current["averageShapelyValues"] = pd.DataFrame.from_dict(
                    {
                        "feature_name": [
                            name
                            for name in current["localExplanations"][0].feature_names
                        ],
                        "value": [
                            np.mean(
                                np.hstack(
                                    [
                                        np.mean(
                                            localExplanations.values[:, featureIndex]
                                        )
                                        for localExplanations in current[
                                            "localExplanations"
                                        ]
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

            if current["globalExplanations"][0] is not None:
                current["averageGlobalExplanations"] = (
                    pd.concat(current["globalExplanations"])
                    .reset_index()
                    .groupby("features")
                    .mean()
                )

            caseAccuracy = np.mean(
                [
                    np.divide(np.count_nonzero(labels == predictions), len(labels))
                    for predictions, labels in zip(
                        current["predictions"], current["testLabels"]
                    )
                ]
            )
            controlAccuracy = 1 - caseAccuracy
            await trackResults.submit(runID, current)

            # plot AUC & hyperparameter convergence
            plotSubtitle = f"""
                {config["tracking"]["name"]}, {embedding["samples"].shape[1]} variants
                Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
                
                {np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(caseAccuracy)} accuracy, {len(embedding['labels']) - np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(controlAccuracy)} accuracy
                {int(np.around(np.mean([len(indices) for indices in current["trainIndices"]])))}±1 train, {int(np.around(np.mean([len(indices) for indices in current["testIndices"]])))}±1 test samples per x-val fold"""
            results = current

            await build_subflow(
                "trackVisualizations",
                (runID, plotSubtitle, model.__class__.__name__, current),
            )

            results["testCount"] = len(trainIDs)
            results["trainCount"] = len(testIDs)
            return results

        return await bootstrap(*args)

    elif name == "evaluate":

        @flow()
        async def evaluate(
            trainIndices,
            testIndices,
            model,
            labels,
            samples,
            variantIndex,
            sampleIndex,
            hyperParameterSpace,
            cvIterator,
        ):
            if config["model"]["hyperparameterOptimization"]:
                fittedOptimizer = optimizeHyperparameters(
                    samples[trainIndices],
                    labels[trainIndices],
                    model,
                    hyperParameterSpace,
                    cvIterator,
                    "neg_mean_squared_error",
                )
                model.set_params(**fittedOptimizer.best_params_)
            else:
                fittedOptimizer = None
            model.fit(samples[trainIndices], labels[trainIndices])
            try:
                probabilities = model.predict_proba(samples[testIndices])
            except AttributeError:
                probabilities = model.predict(samples[testIndices])
                if len(probabilities.shape) <= 1:
                    probabilities = np.array([[1 - p, p] for p in probabilities])
            predictions = np.argmax(probabilities, axis=1)
            modelValues, shapValues, shapExplainer, shapMasker = getFeatureImportances(
                model, samples[testIndices], variantIndex
            )
            globalExplanations = modelValues
            localExplanations = shapValues
            trainLabels = np.array(labels[trainIndices])
            testLabels = np.array(labels[testIndices])
            trainIDs = np.array([sampleIndex[i] for i in trainIndices])
            testIDs = np.array([sampleIndex[i] for i in testIndices])
            return (
                globalExplanations,
                localExplanations,
                probabilities,
                predictions,
                testLabels,
                trainLabels,
                trainIDs,
                testIDs,
                fittedOptimizer,
                shapExplainer,
                shapMasker,
            )

        return await evaluate(*args)

    elif name == "trackVisualizations":

        @flow(task_runner=ConcurrentTaskRunner())
        async def trackVisualizations(runID, plotSubtitle, modelName, current):
            aucPlot = await plotAUC(
                f"""
                    Receiver Operating Characteristic (ROC) Curve
                    {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                    {plotSubtitle}
                    """,
                {
                    f"Fold {k+1}": (
                        current["testLabels"][k],
                        np.array(current["probabilities"][k])[:, 1],
                    )
                    if len(current["probabilities"][k][0].shape) >= 1
                    else (current["testLabels"][k], current["probabilities"][k])
                    for k in range(config["sampling"]["crossValIterations"])
                },
            )
            if config["model"]["hyperparameterOptimization"]:
                optimizerPlot = await plotOptimizer(
                    f"""
                    Hyperparameter convergence, mean squared error
                    {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                    {plotSubtitle}
                    """,
                    {
                        f"Fold {k+1}": [
                            result
                            for result in current["fittedOptimizers"][
                                k
                            ].optimizer_results_
                        ]
                        for k in range(config["sampling"]["crossValIterations"])
                    },
                )

            if config["model"]["calculateShapelyExplanations"]:
                heatmapList = []
                waterfallList = []
                stdDeviation = np.std(
                    (labelsProbabilities[1] - labelsProbabilities[0]) ** 2
                )
                for j in range(config["sampling"]["crossValIterations"]):
                    localExplanations = current["localExplanations"][j]
                    caseExplanations = localExplanations
                    caseExplanations.values = (
                        caseExplanations.values[:, :, 1]
                        if len(caseExplanations.values.shape) > 2
                        else caseExplanations.values
                    )
                    heatmap = plt.figure()
                    plt.title(
                        f"""
                        Shapely explanations from {modelName}
                        Fold {j+1}
                        {plotSubtitle}
                        """
                    )
                    shap.plots.heatmap(caseExplanations, show=False)
                    heatmapList.append(heatmap)
                    plt.close(heatmap)
                    labelsProbabilities = (
                        (
                            current["testLabels"][j],
                            np.array(current["probabilities"][j])[:, 1],
                        )
                        if len(current["probabilities"][j][0].shape) >= 1
                        else (current["testLabels"][j], current["probabilities"][j])
                    )
                    waterfallList.append([])
                    for k in range(len(current["testIDs"][j])):
                        probability = (
                            labelsProbabilities[1][k]
                            if isinstance(labelsProbabilities[1][k], np.ndarray)
                            else labelsProbabilities[1][k]
                        )
                        label = (
                            labelsProbabilities[0][k]
                            if isinstance(labelsProbabilities[0][k], np.ndarray)
                            else labelsProbabilities[0][k]
                        )
                        if (
                            config["tracking"]["plotAllSampleImportances"]
                            or np.absolute((probability - label) ** 2) <= stdDeviation
                        ):
                            sampleID = current["testIDs"][j][k]
                            waterfallPlot = plt.figure()
                            plt.title(
                                f"""
                                {sampleID}
                                Shapely explanations from {modelName}
                                Fold {j+1}
                                {plotSubtitle}
                                """
                            )
                            # patch parameter bug: https://github.com/slundberg/shap/issues/2362
                            to_pass = SimpleNamespace(
                                **{
                                    "values": localExplanations[k].values,
                                    "data": localExplanations[k].data,
                                    "display_data": None,
                                    "feature_names": localExplanations.feature_names,
                                    "base_values": localExplanations[k].base_values[
                                        current["testLabels"][j][k]
                                    ]
                                    if len(localExplanations[k].base_values.shape) == 1
                                    else localExplanations[k].base_values,
                                }
                            )
                            shap.plots.waterfall(to_pass, show=False)
                            waterfallList[j].append(waterfallPlot)
                            plt.close(waterfallPlot)
            plt.close("all")
            if config["tracking"]["remote"]:
                runTracker = neptune.init_run(
                    project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
                    with_id=runID,
                    api_token=config["tracking"]["token"],
                    capture_stdout=False,
                )
                runTracker["plots/aucPlot"] = aucPlot
                if config["model"]["hyperparameterOptimization"]:
                    runTracker["plots/convergencePlot"] = optimizerPlot

                # plot shapely feature importance
                if config["model"]["calculateShapelyExplanations"]:
                    for j in range(config["sampling"]["crossValIterations"]):
                        runTracker[f"plots/featureHeatmap/{j+1}"] = heatmapList[j]
                        if waterfallList[0]:
                            for k in range(len(current["testIDs"][j])):
                                try:
                                    runTracker[
                                        f"plots/samples/{j+1}/{sampleID}"
                                    ] = waterfallList[j][k]
                                except Exception:
                                    runTracker[
                                        f"plots/samples/{j+1}/{sampleID}"
                                    ] = f"""failed to plot: {traceback.format_exc()}"""
                runTracker.stop()
            else:  # store plots locally
                runPath = runID
                aucPlot.savefig(f"{runPath}/aucPlot.svg")
                if config["model"]["hyperparameterOptimization"]:
                    optimizerPlot.savefig(f"{runPath}/optimizerPlot.svg")

                if config["model"]["calculateShapelyExplanations"]:
                    shapelyPath = f"{runPath}/featureImportances/shapelyExplanations"
                    for j in range(config["sampling"]["crossValIterations"]):
                        heatmapList[j].savefig(f"{shapelyPath}/{j+1}_heatmap.svg")
                        if waterfallList[j]:
                            samplePlotPath = f"{runPath}/featureImportances/shapelyExplanations/samples/{j+1}"
                            os.makedirs(samplePlotPath, exist_ok=True)
                            for k in range(len(current["testIDs"][j])):
                                try:
                                    waterfallList[j][k].savefig(
                                        f"{samplePlotPath}/{sampleID}.svg"
                                    )
                                except Exception:
                                    print(
                                        f"""failed to plot: {traceback.format_exc()}"""
                                    )
            plt.close("all")

    await trackVisualizations(*args)


@task()
def download_file(run_id, field="sampleResults", extension="csv"):
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


@flow(task_runner=RayTaskRunner())
async def main():
    (
        caseGenotypes,
        caseIDs,
        controlGenotypes,
        controlIDs,
        clinicalData,
    ) = await processInputFiles()
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator
    if config["tracking"]["remote"]:
        projectTracker = neptune.init_project(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            api_token=config["tracking"]["token"],
        )

    classification_tasks = [
        build_subflow(
            "classify",
            (
                caseGenotypes,
                controlGenotypes,
                clinicalData,
                model,
                hyperParameterSpace,
                innerCvIterator,
                outerCvIterator,
            ),
        )
        for (model, hyperParameterSpace) in list(config["model"]["stack"].items())
    ]

    results = await limited_gather(
        classification_tasks, config["model"]["concurrencyLimit"]
    )

    labelsProbabilitiesByModelName = dict()
    variantCount = 0
    lastVariantCount = 0
    sampleResults = {}

    if (
        config["sampling"]["lastIteration"] > 0
        and j < config["sampling"]["lastIteration"]
        and config["tracking"]["remote"]
    ):
        runsTable = projectTracker.fetch_runs_table().to_pandas()
        pastRuns = runsTable[
            runsTable["bootstrapIteration"] < config["sampling"]["lastIteration"]
        ]
        label_args_list = [
            ("testLabels", runID) for runID in pastRuns["sys/id"].unique()
        ]
        for runID in pastRuns["sys/id"].unique():
            download_file(runID, "testLabels", "csv")
            download_file(runID, "sampleResults", "csv")
            download_file(runID, "featureImportance/modelCoefficients", "csv")
            download_file(runID, "featureImportance/shapelyExplanations/average", "csv")

    # asyncio.gather preserves order
    for i, model in enumerate(config["model"]["stack"]):
        modelName = model.__class__.__name__

        if (
            config["sampling"]["lastIteration"] > 0
            and j < config["sampling"]["lastIteration"]
        ):
            if config["tracking"]["remote"]:
                # if run was interrupted, and bootstrapping began after the first iteration (and incomplete runs deleted)
                for j in range(0, config["sampling"]["lastIteration"]):
                    results["bootstraps"][i][j] = {}
                    # get bootstrap runs for model
                    currentRuns = pastRuns.loc[
                        (pastRuns["bootstrapIteration"] == j)
                        & (pastRuns["model"] == modelName)
                    ]
                    results["bootstraps"][i][j]["trainCount"] = np.around(
                        currentRuns["nTrain"].unique()[0]
                    )
                    results["bootstraps"][i][j]["testCount"] = np.around(
                        currentRuns["nTest"].unique()[0]
                    )
                    samplesResultsByFold = [
                        load_fold_dataframe(("sampleResults", runID))
                        for runID in currentRuns["sys/id"].unique()
                    ]
                    loadedSamples = pd.concat(samplesResultsByFold).set_index(
                        "id", drop=True
                    )
                    # unique run ID ordering matches label_args_list
                    currentRunIDIndices = np.where(
                        pastRuns["sys/id"].unique() == currentRuns.loc["sys/id"]
                    )
                    loadedLabels = [
                        load_fold_dataframe(args)
                        for k in currentRunIDIndices
                        for args in label_args_list[k]
                    ]
                    for sampleID in loadedSamples.index:
                        if sampleID not in results["samples"]:
                            results["samples"][sampleID] = loadedSamples.loc[sampleID][
                                "probability"
                            ].to_numpy()
                        else:
                            results["samples"][sampleID] = np.append(
                                loadedSamples.loc[sampleID]["probability"].to_numpy(),
                                results["samples"][sampleID],
                            )
                        if sampleID not in results["labels"]:
                            results["labels"][sampleID] = loadedSamples.loc[sampleID][
                                "label"
                            ].unique()[
                                0
                            ]  # all labels should be same for sample ID
                    results["bootstraps"][i][j]["testLabels"] = loadedLabels
                    results["bootstraps"][i][j]["probabilities"] = samplesResultsByFold
                    # TODO use conditional to check if run has feature explanations
                    try:
                        results["bootstraps"][i][j]["globalExplanations"] = [
                            load_fold_dataframe(
                                ("featureImportance/modelCoefficients", runID)
                            )
                            for runID in currentRuns["sys/id"].unique()
                        ]
                    except:
                        pass
                    try:
                        results[i][j][
                            "averageShapelyExplanations"
                        ] = load_fold_dataframe(
                            ("featureImportance/shapelyExplanations/average", runID)
                        )
                    except:
                        pass

            else:
                bootstrapFolders = os.listdir(
                    f"{config['tracking']['project']}/bootstraps"
                )
                # convert to int
                bootstrapFolders = [int(folder) for folder in bootstrapFolders]
                bootstrapFolders.sort()
                assert (
                    max(bootstrapFolders) == config["sampling"]["lastIteration"]
                )  # TODO automatically determine last iteration by max
                for j in range(0, config["sampling"]["lastIteration"]):
                    results["bootstraps"][i][j] = {}
                    currentBootstrap = bootstrapFolders[j]
                    modelFolders = os.listdir(
                        f"{config['tracking']['project']}/bootstraps/{currentBootstrap}"
                    )
                    currentFolder = modelFolders.index(modelName)
                    currentFiles = os.listdir(
                        f"{config['tracking']['project']}/bootstraps/{currentBootstrap}/{currentFolder}"
                    )
                    for fileName in currentFiles:
                        if "testCount" in fileName:
                            results["bootstraps"][i][j]["testCount"] = fileName.split(
                                "_"
                            )[1]
                        elif "trainCount" in fileName:
                            results["bootstraps"][i][j]["trainCount"] = fileName.split(
                                "_"
                            )[1]
                        # TODO handle rest of local files

        modelResult = results["bootstraps"][i]

        for sampleID in modelResult["samples"].keys():
            flattenedProbabilities = np.array(
                [
                    prediction[1] if len(prediction) >= 2 else prediction
                    for prediction in modelResult["samples"][sampleID]
                ]
            )
            if sampleID not in sampleResults:
                # label, probability, accuracy
                sampleResults[sampleID] = [
                    modelResult["labels"][sampleID],
                    modelResult["samples"][sampleID],
                    np.mean(
                        [
                            np.around(caseProbability)
                            == modelResult["labels"][sampleID]
                            for caseProbability in flattenedProbabilities
                        ]
                    ),
                ]
            else:
                sampleResults[sampleID][1].append(modelResult["samples"][sampleID])
                sampleResults[sampleID][2] = np.mean(
                    [
                        sampleResults[sampleID][2],
                        np.mean(
                            [
                                np.around(caseProbability)
                                == modelResult["labels"][sampleID]
                                for caseProbability in flattenedProbabilities
                            ]
                        ),
                    ]
                )

        if modelName not in labelsProbabilitiesByModelName:
            labelsProbabilitiesByModelName[modelName] = [[], []]
        globalExplanationsList = []
        for j in range(config["sampling"]["bootstrapIterations"]):
            bootstrapResult = modelResult[j]
            # append labels
            labelsProbabilitiesByModelName[modelName][0] = np.hstack(
                (
                    labelsProbabilitiesByModelName[modelName][0],
                    *bootstrapResult["testLabels"],
                )
            )
            # append probabilities
            labelsProbabilitiesByModelName[modelName][1] = np.hstack(
                [
                    labelsProbabilitiesByModelName[modelName][1],
                    np.concatenate(bootstrapResult["probabilities"])[:, 1]
                    if len(bootstrapResult["probabilities"][0].shape) >= 1
                    else np.concatenate(bootstrapResult["probabilities"]),
                ]
            )

            if not isinstance(bootstrapResult["globalExplanations"][0], pd.DataFrame):
                continue
            variantCount = bootstrapResult["globalExplanations"][0].shape[0]
            assert lastVariantCount == variantCount or lastVariantCount == 0
            lastVariantCount = variantCount

            globalExplanationsList += bootstrapResult["globalExplanations"]

        if globalExplanationsList:
            averageGlobalExplanationsDataFrame = (
                pd.concat(globalExplanationsList)
                .reset_index()
                .groupby("features")
                .mean()
            )
            if config["tracking"]["remote"]:
                projectTracker[f"averageModelCoefficients/{modelName}"].upload(
                    serializeDataFrame(averageGlobalExplanationsDataFrame)
                )
            else:
                os.makedirs(
                    f"{config['tracking']['project']}/averageModelCoefficients/",
                    exist_ok=True,
                )
                averageGlobalExplanationsDataFrame.to_csv(
                    f"{config['tracking']['project']}/averageModelCoefficients/{modelName}.csv"
                )

    sampleResultsDataFrame = pd.DataFrame.from_dict(
        sampleResults, orient="index", columns=["label", "probability", "accuracy"]
    )
    sampleResultsDataFrame.index.name = "id"

    if config["tracking"]["remote"]:
        projectTracker["sampleResults"].upload(
            serializeDataFrame(sampleResultsDataFrame)
        )
        projectTracker["sampleResultsObject"].upload(
            File.as_pickle(sampleResultsDataFrame)
        )
    else:
        sampleResultsDataFrame.to_csv(
            f"{config['tracking']['project']}/sampleResults.csv"
        )

    if config["model"]["calculateShapelyExplanations"]:
        averageShapelyExplanationsDataFrame = (
            pd.concat(
                [
                    modelResult[j]["averageShapelyValues"]
                    for j in range(config["sampling"]["bootstrapIterations"])
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
        else:
            os.makedirs(
                f"{config['tracking']['project']}/averageShapelyExplanations/",
                exist_ok=True,
            )
            averageShapelyExplanationsDataFrame.to_csv(
                f"{config['tracking']['project']}/averageShapelyExplanations.csv"
            )

    caseAccuracy = sampleResultsDataFrame[sampleResultsDataFrame["label"] == 1][
        "accuracy"
    ].mean()
    controlAccuracy = 1 - caseAccuracy

    bootstrapTrainCount = int(
        np.around(
            np.mean(
                [
                    modelResult[j]["trainCount"]
                    for j in range(config["sampling"]["bootstrapIterations"])
                ]
            )
        )
    )
    bootstrapTestCount = int(
        np.around(
            np.mean(
                [
                    modelResult[j]["testCount"]
                    for j in range(config["sampling"]["bootstrapIterations"])
                ]
            )
        )
    )

    plotSubtitle = f"""
    {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
    {config["tracking"]["name"]}, {variantCount} variants
    Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

    {sampleResultsDataFrame['label'].value_counts()[1]} cases @ {'{:.1%}'.format(caseAccuracy)} accuracy, {sampleResultsDataFrame['label'].value_counts()[0]} controls @ {'{:.1%}'.format(controlAccuracy)} accuracy
    {bootstrapTrainCount}±1 train, {bootstrapTestCount}±1 test samples per bootstrap iteration"""

    accuracyHistogram = px.histogram(
        sampleResultsDataFrame,
        x="accuracy",
        color="label",
        pattern_shape="label",
        range_x=[0, 1],
        title=f"""Mean sample accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
    )
    aucPlot = await plotAUC(
        f"""
            Receiver Operating Characteristic (ROC) Curve
            {plotSubtitle}
            """,
        labelsProbabilitiesByModelName,
    )
    calibrationPlot = await plotCalibration(
        f"""
            Calibration Curve
            {plotSubtitle}
            """,
        labelsProbabilitiesByModelName,
    )
    if config["model"]["hyperparameterOptimization"]:
        convergencePlot = plotOptimizer(
            f"""
            Convergence Plot
            {plotSubtitle}
            """,
            {
                modelName: [
                    result
                    for j in range(config["sampling"]["bootstrapIterations"])
                    for foldOptimizer in results["bootstraps"][i][j]["fittedOptimizers"]
                    for result in foldOptimizer.optimizer_results_
                ]
                for i, modelName in enumerate(config["model"]["stack"])
            },
        )

    if config["tracking"]["remote"]:
        projectTracker["sampleAccuracyPlot"].upload(accuracyHistogram)
        projectTracker["aucPlot"].upload(aucPlot)

        if config["model"]["hyperparameterOptimization"]:
            projectTracker["calibrationPlot"].upload(File.as_image(calibrationPlot))

        if config["model"]["hyperparameterOptimization"]:
            projectTracker["convergencePlot"].upload(File.as_image(convergencePlot))

        projectTracker.stop()
    else:
        accuracyHistogram.save_html(
            f"{config['tracking']['project']}/accuracyPlot.html"
        )
        aucPlot.savefig(f"{config['tracking']['project']}/aucPlot.svg")
        calibrationPlot.savefig(f"{config['tracking']['project']}/calibrationPlot.svg")
        if config["model"]["hyperparameterOptimization"]:
            convergencePlot.savefig(
                f"{config['tracking']['project']}/convergencePlot.svg"
            )

    return results


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
    # TODO replace notebook code with src imports
    ray.shutdown()
    parallelRunner = ray.init()

    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())

    results = asyncio.run(main())
    pickle.dump(results, open(f"results_{config['tracking']['project']}.pkl", "wb"))
