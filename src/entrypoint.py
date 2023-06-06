import asyncio
from inspect import isclass
import pickle
import traceback
import numpy as np
import neptune
import pandas as pd
import ray
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from prefect_ray.task_runners import RayTaskRunner
from prefect import flow
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
            parameterSpace,
            innerCvIterator,
            outerCvIterator,
        ):
            results = {}
            results["samples"] = {}
            results["labels"] = {}
            results["model"] = model.__class__.__name__
            for runNumber in range(config["sampling"]["bootstrapIterations"]):
                trainIDs = set()
                testIDs = set()
                results[runNumber] = {}
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
                        pass
                        # model = model(verbose=False, optimizer_fn=Lion)
                print(f"Iteration {runNumber+1} with model {model.__class__.__name__}")
                runID = await beginTracking.submit(
                    model, runNumber, embedding, clinicalData, deserializedIDs
                )
                # outer cross-validation
                crossValIndices = np.array(
                    [
                        (cvTrainIndices, cvTestIndices)
                        for (cvTrainIndices, cvTestIndices) in outerCvIterator.split(
                            embedding["samples"], embedding["labels"]
                        )
                    ]
                )
                current["trainIndices"] = crossValIndices[:, 0]
                current["testIndices"] = crossValIndices[:, 1]
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
                                    parameterSpace,
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
                                for name in current["localExplanations"][
                                    0
                                ].feature_names
                            ],
                            "value": [
                                np.mean(
                                    np.hstack(
                                        [
                                            np.mean(
                                                localExplanations.values[
                                                    :, featureIndex
                                                ]
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
                results[runNumber] = current

                # record sample metrics
                for fold in range(config["sampling"]["crossValIterations"]):
                    for j, sampleID in enumerate(current["testIDs"][fold]):
                        try:
                            results["samples"][sampleID].append(
                                current["probabilities"][fold][j]
                            )
                        except KeyError:
                            results["samples"][sampleID] = [
                                current["probabilities"][fold][j]
                            ]
                        finally:
                            results["labels"][sampleID] = current["testLabels"][fold][j]

                await build_subflow(
                    "trackVisualizations",
                    (runID, plotSubtitle, model.__class__.__name__, current),
                )

                results[runNumber]["testCount"] = len(trainIDs)
                results[runNumber]["trainCount"] = len(testIDs)
            return results

        return await classify(*args)

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
            parameterSpace,
            cvIterator,
        ):
            if config["model"]["hyperparameterOptimization"]:
                fittedOptimizer = optimizeHyperparameters(
                    samples[trainIndices],
                    labels[trainIndices],
                    model,
                    parameterSpace,
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
            runTracker = neptune.init_run(
                project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
                with_id=runID,
                api_token=config["tracking"]["token"],
            )
            runTracker["plots/aucPlot"] = await plotAUC.submit(
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
                runTracker["plots/convergencePlot"] = await plotOptimizer.submit(
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

            # plot shapely feature importance
            if config["model"]["calculateShapelyExplanations"]:
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
                    runTracker[f"plots/featureHeatmap/{j+1}"] = heatmap
                    plt.close(heatmap)
                    labelsProbabilities = (
                        (
                            current["testLabels"][j],
                            np.array(current["probabilities"][j])[:, 1],
                        )
                        if len(current["probabilities"][j][0].shape) >= 1
                        else (current["testLabels"][j], current["probabilities"][j])
                    )
                    stdDeviation = np.std(
                        (labelsProbabilities[1] - labelsProbabilities[0]) ** 2
                    )
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
                            try:
                                runTracker[
                                    f"plots/samples/{j+1}/{sampleID}"
                                ] = waterfallPlot
                            except Exception:
                                runTracker[
                                    f"plots/samples/{j+1}/{sampleID}"
                                ] = f"""failed to plot: {traceback.format_exc()}"""
                            plt.close(waterfallPlot)
            plt.close("all")
            runTracker.stop()

        await trackVisualizations(*args)


@flow(task_runner=RayTaskRunner())
async def bootstrapSampling():
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
    projectTracker = neptune.init_project(
        project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
        api_token=config["tracking"]["token"],
    )

    results = await asyncio.gather(
        *[
            build_subflow(
                "classify",
                (
                    caseGenotypes,
                    controlGenotypes,
                    clinicalData,
                    model,
                    hyperparameterSpace,
                    innerCvIterator,
                    outerCvIterator,
                ),
            )
            for (model, hyperparameterSpace) in list(config["model"]["stack"].items())
        ]
    )

    labelsProbabilitiesByModelName = dict()
    variantCount = 0
    lastVariantCount = 0
    sampleResults = {}

    # asyncio.gather preserves order
    for i, model in enumerate(config["model"]["stack"]):
        modelName = model.__class__.__name__
        modelResult = results[i]

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
            projectTracker[f"averageModelCoefficients/{modelName}"].upload(
                serializeDataFrame(averageGlobalExplanationsDataFrame)
            )

    sampleResultsDataFrame = pd.DataFrame.from_dict(
        sampleResults, orient="index", columns=["label", "probability", "accuracy"]
    )
    sampleResultsDataFrame.index.name = "id"

    projectTracker["sampleResults"].upload(serializeDataFrame(sampleResultsDataFrame))
    projectTracker["sampleResultsObject"].upload(File.as_pickle(sampleResultsDataFrame))

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
        projectTracker["averageShapelyExplanations"].upload(
            serializeDataFrame(averageShapelyExplanationsDataFrame)
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

    projectTracker["sampleAccuracyPlot"].upload(
        px.histogram(
            sampleResultsDataFrame,
            x="accuracy",
            color="label",
            pattern_shape="label",
            range_x=[0, 1],
            title=f"""Mean sample accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
        )
    )

    projectTracker["aucPlot"].upload(
        plotAUC(
            f"""
    Receiver Operating Characteristic (ROC) Curve
    {plotSubtitle}
    """,
            labelsProbabilitiesByModelName,
        )
    )

    projectTracker["calibrationPlot"].upload(
        File.as_image(
            plotCalibration(
                f"""
    Calibration Curve
    {plotSubtitle}
    """,
                labelsProbabilitiesByModelName,
            )
        )
    )

    if config["model"]["hyperparameterOptimization"]:
        projectTracker["convergencePlot"].upload(
            File.as_image(
                plotOptimizer(
                    f"""
      Convergence Plot
      {plotSubtitle}
      """,
                    {
                        modelName: [
                            result
                            for j in range(config["sampling"]["bootstrapIterations"])
                            for foldOptimizer in results[i][j]["fittedOptimizers"]
                            for result in foldOptimizer.optimizer_results_
                        ]
                        for i, modelName in enumerate(config["model"]["stack"])
                    },
                )
            )
        )

    projectTracker.stop()
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
    ray.shutdown()
    parallelRunner = ray.init()

    results = asyncio.run(bootstrapSampling())
    pickle.dump(results, open("results.pkl", "wb"))

    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())
