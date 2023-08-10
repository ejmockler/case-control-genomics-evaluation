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

from tasks.data import recoverPastRuns

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
    bootstrap,
    evaluate,
    serializeDataFrame,
    trackResults,
)
from tasks.visualize import (
    trackBootstrapVisualizations,
)
from tasks.data import processSampleResults
from neptune.types import File
from config import config
from models import stack as modelStack

import gc
from joblib import Parallel, delayed


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
    genotypeData=None,
    clinicalData=None,
):
    if genotypeData is None:
        (
            genotypeData,
            clinicalData,
        ) = processInputFiles(config)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator

    bootstrap_args = [
        (
            genotypeData,
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
        results = recoverPastRuns(modelStack, results)

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

    s

    return (
        results,
        clinicalData,
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
