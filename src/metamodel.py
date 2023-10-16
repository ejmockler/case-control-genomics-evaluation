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
)
from metaconfig import metaconfig
from config import config
from models import stack as modelStack
from tasks.input import processInputFiles
from tasks.visualize import poolSampleResults


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

    pooledBaselineFeatureResults = {}
    baselineFeatureResultsByModel = {}
    for i, model in enumerate(modelStack.keys()):
        modelResults = results[i]
        baselineFeatureResultsByModel[model.__class__.__name__] = modelResults.sample_results_dataframe
    pooledBaselineFeatureResults = poolSampleResults(pd.concat([df for df in baselineFeatureResultsByModel.values()]))
        
    return pooledBaselineFeatureResults, baselineFeatureResultsByModel, selectedFeature


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
    results,
    pooledResults,
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    clinicalData,
):
    pooledBaselineFeatureResults, baselineFeatureResultsByModel, selectedFeature = getBaselineFeatureResults(
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
        clinicalData,
        config,
    )
    resultsByModel = {}
    for modelResult in results:
        sampleResults = modelResult.sample_results_dataframe
        # serialize probability arrays from string
        modelResult["probability_mean"] = modelResult["probability_mean"].apply(lambda x: np.array(eval(x)))
        # take intersection of bootstrapped samples
        resultsByModel[modelResult.model_name] = modelResult.loc[baselineFeatureResultsByModel[modelResult.model_name].index.intersection(sampleResults.index)]
        baselineFeatureResultsByModel[modelResult.model_name] = baselineFeatureResultsByModel[modelResult.model_name].loc[sampleResults.index]
        modelResult["baseline_probability"] = baselineFeatureResultsByModel[modelResult.model_name]["probability_mean"]
    
    # same for pooled results
    pooledResults = modelResult.loc[pooledBaselineFeatureResults.index.intersection(pooledResults.index)]
    pooledBaselineFeatureResults = pooledBaselineFeatureResults.loc[pooledResults.index]
    
    pooledRelativePerplexities = pd.DataFrame(index=pooledResults.index)
    new_cols = pooledResults.apply(
        lambda row: relativePerplexity(
            [row["label"]] * len(row["probability_mean"]),
            row["probability_mean"],
            [row["label"]] * len(row["baseline_probability"]),
            row["baseline_probability"],
        ),
        axis=1,
        result_type="expand",
    )

    (
        pooledRelativePerplexities["all features"],
        pooledRelativePerplexities[f"{selectedFeature}"],
        pooledRelativePerplexities["relative"],
    ) = (new_cols[0], new_cols[1], new_cols[2])

    # separate discordant & well-classified
    pooledDiscordantSamples = pooledResults.loc[
        pooledResults["accuracy_mean"] <= metaconfig["samples"]["discordantThreshold"],
    ]
    pooledAccurateSamples = pooledResults.loc[
        pooledResults["accuracy_mean"] >= metaconfig["samples"]["accurateThreshold"],
    ]
    
    relativePerplexitiesByModel = {}

    for model_name, modelResult in resultsByModel.items():
        relativePerplexitiesByModel[model_name] = pd.DataFrame(index=modelResult.index)
        new_cols = modelResult.apply(
            lambda row: relativePerplexity(
                [row["label"]] * len(row["probability_mean"]),
                row["probability_mean"],
                [row["label"]] * len(row["baseline_probability"]),
                row["baseline_probability"],
            ),
            axis=1,
            result_type="expand",
        )

        (
            relativePerplexitiesByModel[model_name]["all features"],
            relativePerplexitiesByModel[model_name][f"{selectedFeature}"],
            relativePerplexitiesByModel[model_name]["relative"],
        ) = (new_cols[0], new_cols[1], new_cols[2])


    return (
        pooledRelativePerplexities,
        relativePerplexitiesByModel,
        pooledDiscordantSamples,
        pooledAccurateSamples,
        pooledBaselineFeatureResults,
        baselineFeatureResultsByModel
    )


def readSampleIDs(table, label):
    sampleIDs = table.loc[table["label"] == label]
    return sampleIDs.index.tolist()


def sequesterOutlierSamples(sampleResultsByModel = None, pooledSampleResults = None):
    if os.path.exists(f"projects/{config['tracking']['project']}/pooledAccurateSamples.csv") and os.path.exists(f"projects/{config['tracking']['project']}/pooledDiscordantSamples.csv"):
        thresholdedSamplesByModel = {
            "accurate": { 
                { model.__class__.__name__: 
                    pd.read_csv(f"projects/{config['tracking']['project']}/{model.__class__.__name__}/accurateSamples.csv", index_col="id")
                } for model in modelStack.keys()
            },
           "discordant": {
                { model.__class__.__name__: 
                    pd.read_csv(f"projects/{config['tracking']['project']}/{model.__class__.__name__}/discordantSamples.csv", index_col="id"),
                } for model in modelStack.keys() 
           }
        }
        thresholdedPooledSamples = {
            "accurate": pd.read_csv(
                f"projects/{config['tracking']['project']}/pooledAccurateSamples.csv",
                index_col="id",
            ),
            "discordant": pd.read_csv(
                f"projects/{config['tracking']['project']}/pooledDiscordantSamples.csv",
                index_col="id",
            ),
        }
    else:
        thresholdedSamplesByModel = {}
        thresholdedSamplesByModel["accurate"] = {}
        thresholdedSamplesByModel["discordant"] = {}
        
        for modelName, sampleResults in sampleResultsByModel.items():
            thresholdedSamplesByModel["accurate"][modelName] = sampleResults.loc[sampleResults["probability_mean"] >= metaconfig["samples"]["accurateThreshold"]]
            thresholdedSamplesByModel["discordant"][modelName] = sampleResults.loc[sampleResults["probability_mean"] <= metaconfig["samples"]["discordantThreshold"]]
        
        thresholdedPooledSamples = {
            "accurate": pooledSampleResults.loc[pooledSampleResults["probability_mean"] >= metaconfig["samples"]["accurateThreshold"]],
            "discordant": pooledSampleResults.loc[pooledSampleResults["probability_mean"] <= metaconfig["samples"]["discordantThreshold"]],
        }

    for sampleType, dataframe in thresholdedPooledSamples.items():
        for labelType, label in {config["clinicalTable"]["caseAlias"]: 1, config["clinicalTable"]["controlAlias"]: 0}.items():
            if metaconfig["samples"]["sequester"][sampleType][labelType]:
                ids = readSampleIDs(dataframe, label=label)
                config["sampling"]["sequesteredIDs"].extend(ids)
                
    for sampleType, modelSet in thresholdedSamplesByModel.items():
        for modelName, dataframe in modelSet.items():
            for labelType, label in {config["clinicalTable"]["caseAlias"]: 1, config["clinicalTable"]["controlAlias"]: 0}.items():
                if metaconfig["samples"]["sequester"][sampleType][labelType]:
                    ids = readSampleIDs(dataframe, label=label)
                    config["sampling"]["sequesteredIDs"].extend(ids)
                    
    config["sampling"]["sequesteredIDs"] = list(set(config["sampling"]["sequesteredIDs"]))
    return config
                

@flow()
def main():
    newWellClassified = True
    countSuffix = 1
    baseProjectPath = config["tracking"]["project"]

    while newWellClassified:
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        
        if countSuffix <= metaconfig["tracking"]["lastIteration"]:
            sampleResultsByModel = {
                model.__class__.__name__: pd.read_csv(f"projects/{baseProjectPath}/{model.__class__.__name__}/sampleResults.csv", index_col="id") for model in modelStack.keys()
            }
            pooledSampleResults = pd.read_csv(f"projects/{baseProjectPath}/pooledSampleResults.csv", index_col="id")
            
            config = sequesterOutlierSamples(sampleResultsByModel, pooledSampleResults)
            countSuffix += 1
            continue
        else:
            results, genotypeData, clinicalData = runMLstack(config)
            config["sampling"]["lastIteration"] = 0
        
        pooledResults = poolSampleResults(pd.concat([modelResult.sample_result_dataframe for modelResult in results]))
        
        (
            pooledSamplePerplexities,
            relativePerplexitiesByModel,
            pooledDiscordantSamples,
            pooledAccurateSamples,
            pooledBaselineFeatureResults,
            baselineFeatureResultsByModel
        ) = measureIterations(
            results,
            pooledResults,
            genotypeData.case.genotype,
            genotypeData.control.genotype,
            genotypeData.holdout_case.genotype,
            genotypeData.holdout_control.genotype,
            clinicalData,
        )
        
        np.set_printoptions(threshold=np.inf)
        
        pooledBaselineFeatureResults.to_csv(
            f"projects/{config['tracking']['project']}/pooledBaselineFeatureResults.csv"
        )
        
        pooledSamplePerplexities.to_csv(
            f"projects/{config['tracking']['project']}/pooledSamplePerplexities.csv"
        )

        pooledAccurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledAccurateSamples.csv"
        )
        
        pooledDiscordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledDiscordantSamples.csv"
        )
        
        for modelName, modelResults in baselineFeatureResultsByModel.items():
            modelResults.to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/baselineFeatureResults.csv"
            )
            
        for modelName, modelResults in relativePerplexitiesByModel.items():
            modelResults.to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/relativePerplexities.csv"
            )
            
        for modelName, modelResults in 
        
        # Check for new well-classified samples. If not present, stop iterations.
        newWellClassified = not pooledAccurateSamples.empty
        countSuffix += 1



if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()
    main()
    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())
