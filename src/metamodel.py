import asyncio
import gc
import os
from joblib import Parallel, delayed
import pandas as pd
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import numpy as np
from numpy import array, float32  # to eval strings as arrays

from sklearn.metrics import log_loss

import ray
from sklearn.model_selection import StratifiedKFold

from mlStack import (
    bootstrap,
    remove_all_flows,
    main as runMLstack,
    serializeBootstrapResults,
    serializeResultsDataframe,
)
from metaconfig import metaconfig
from config import config
from models import stack as modelStack
from tasks.input import processInputFiles


def relativePerplexity(y_true, y_pred, y_true_baseline, y_pred_baseline, epsilon=1e-15):
    samplePerplexity = perplexity(y_true, y_pred)
    baselineSamplePerplexity = perplexity(y_true_baseline, y_pred_baseline)

    return pd.Series(
        [
            samplePerplexity,
            baselineSamplePerplexity,
            np.divide(samplePerplexity, baselineSamplePerplexity + epsilon),
        ]
    )  # relative perplexity = perplexity / perplexity of model with single-most case correlated feature


def getBaselineFeatureResults(
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    clinicalData,
    config,
):
    selectedFeature = findBaselineFeature(caseGenotypes, controlGenotypes)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator

    bootstrap_args = [
        (
            caseGenotypes.loc[[selectedFeature]],
            controlGenotypes.loc[[selectedFeature]],
            holdoutCaseGenotypes.loc[[selectedFeature]],
            holdoutControlGenotypes.loc[[selectedFeature]],
            clinicalData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            outerCvIterator,
            config,
            False,  # disable tracking
        )
        for model, hyperParameterSpace in list(modelStack.items())
    ]
    results = Parallel(n_jobs=-1)(delayed(bootstrap)(*args) for args in bootstrap_args)

    baselineFeatureResults = {}
    for i in range(len(modelStack)):
        modelResults = results[i]
        baselineFeatureResults = serializeBootstrapResults(
            modelResults, baselineFeatureResults
        )
    baselineFeatureResultsDataframe = serializeResultsDataframe(baselineFeatureResults)
    baselineFeatureResultsDataframe.index.name = "id"

    return baselineFeatureResultsDataframe, selectedFeature


def perplexity(y_true, y_pred):
    crossEntropy = log_loss(
        y_true, y_pred, labels=[0, 1], eps=1e-15
    )  # linear predictions (exactly 0 or 1) depend on offset of 1e-15 when log is applied to avoid NaN

    # perplexity = 2 ^ crossEntropy
    return np.power(2, crossEntropy)


def findBaselineFeature(caseGenotypes, controlGenotypes):
    # calculate the mean of each feature for cases and controls
    mean_cases = caseGenotypes.mean(axis=1)
    mean_controls = controlGenotypes.mean(axis=1)

    # calculate the absolute difference in means for each feature
    diff_means = abs(mean_cases - mean_controls)

    # get the feature with the largest difference in means
    selected_feature = diff_means.idxmax()

    print("Selected feature for baseline perplexity: ", selected_feature)
    return selected_feature


@task(retries=10)
def measureIterations(
    result,
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    clinicalData,
):
    baselineFeatureResults, selectedFeature = getBaselineFeatureResults(
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
        clinicalData,
        config,
    )
    # serialize probability arrays from string
    result["probability"] = result["probability"].apply(lambda x: np.array(eval(x)))
    # take intersection of bootstrapped samples
    result = result.loc[baselineFeatureResults.index.intersection(result.index)]
    baselineFeatureResults = baselineFeatureResults.loc[result.index]
    result["baselineProbability"] = baselineFeatureResults["probability"]

    relativePerplexities = pd.DataFrame(index=result.index)
    new_cols = result.apply(
        lambda row: relativePerplexity(
            [row["label"]] * len(row["probability"]),
            row["probability"],
            [row["label"]] * len(row["baselineProbability"]),
            row["baselineProbability"],
        ),
        axis=1,
        result_type="expand",
    )

    (
        relativePerplexities["all features"],
        relativePerplexities[f"{selectedFeature}"],
        relativePerplexities["relative"],
    ) = (new_cols[0], new_cols[1], new_cols[2])

    # separate discordant & well-classified
    discordantSamples = result.loc[
        result["accuracy"] <= metaconfig["samples"]["discordantThreshold"],
    ]
    accurateSamples = result.loc[
        result["accuracy"] >= metaconfig["samples"]["accurateThreshold"],
    ]

    return (
        relativePerplexities,
        discordantSamples,
        accurateSamples,
        baselineFeatureResults,
    )


def readSampleIDs(table, label):
    sampleIDs = table.loc[table["label"] == label]
    return sampleIDs.index.tolist()


def sequesterOutlierSamples():
    samples = {
        "accurate": pd.read_csv(
            f"projects/{config['tracking']['project']}/accurateSamples.csv",
            index_col="id",
        ),
        "discordant": pd.read_csv(
            f"projects/{config['tracking']['project']}/discordantSamples.csv",
            index_col="id",
        ),
    }

    for sampleType, dataframe in samples.items():
        for labelType, label in {"case": 1, "control": 0}.items():
            if metaconfig["samples"]["sequester"][sampleType][labelType]:
                ids = readSampleIDs(dataframe, label=label)
                config["sampling"]["sequesteredIDs"].extend(ids)


@flow()
def main():
    newWellClassified = True
    countSuffix = 1

    baseProjectPath = config["tracking"]["project"]
    while newWellClassified:
        gc.collect()
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        if countSuffix <= metaconfig["tracking"]["lastIteration"]:
            sequesterOutlierSamples()
            countSuffix += 1
            continue
        else:
            (
                results,
                genotypeData,
                clinicalData,
            ) = runMLstack(config)
            config["sampling"]["lastIteration"] = 0
        currentResults = pd.read_csv(
            f"projects/{config['tracking']['project']}/sampleResults.csv",
            index_col="id",
        )
        (
            samplePerplexities,
            discordantSamples,
            accurateSamples,
            baselineFeatureResults,
        ) = measureIterations(
            currentResults,
            genotypeData.case.genotype,
            genotypeData.control.genotype,
            genotypeData.holdout_case.genotype,
            genotypeData.holdout_control.genotype,
            clinicalData,
        )

        samplePerplexities["accuracy"] = currentResults.loc[
            currentResults.index.intersection(samplePerplexities.index)
        ]["accuracy"]
        samplePerplexities["baselineAccuracy"] = baselineFeatureResults.loc[
            baselineFeatureResults.index.intersection(samplePerplexities.index)
        ]["accuracy"]
        samplePerplexities["label"] = currentResults.loc[
            currentResults.index.intersection(samplePerplexities.index)
        ]["label"]

        np.set_printoptions(threshold=np.inf)
        samplePerplexities.to_csv(
            f"projects/{config['tracking']['project']}/samplePerplexities.csv"
        )
        accurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/accurateSamples.csv"
        )
        discordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/discordantSamples.csv"
        )
        baselineFeatureResults["probability"] = baselineFeatureResults[
            "probability"
        ].map(lambda x: np.array2string(np.array(x), separator=","))
        baselineFeatureResults.to_csv(
            f"projects/{config['tracking']['project']}/baselineFeatureSampleResults.csv"
        )
        if (
            len(
                currentResults.loc[
                    (currentResults["label"] == 1)
                    & (
                        currentResults["accuracy"]
                        >= metaconfig["samples"]["accurateThreshold"]
                    )
                ]
            )
            == 0
        ):
            newWellClassified = False
            break
        sequesterOutlierSamples()
        countSuffix += 1


if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()
    main()
    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())
