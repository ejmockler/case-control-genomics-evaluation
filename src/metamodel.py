import asyncio
import gc
import os
from copy import deepcopy
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
    genotypeData,
    clinicalData,
    config,
):
    selectedFeature = findBaselineFeature(genotypeData.case.genotype, genotypeData.control.genotype)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator
    
    genotypeData.case.genotype = genotypeData.case.genotype.loc[[selectedFeature]]
    genotypeData.control.genotype = genotypeData.control.genotype.loc[[selectedFeature]]
    genotypeData.holdout_case.genotype = genotypeData.holdout_case.genotype.loc[[selectedFeature]]
    genotypeData.holdout_control.genotype = genotypeData.holdout_control.genotype.loc[[selectedFeature]]

    bootstrap_args = [
        (
            genotypeData,
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
    classificationResults,
    pooledResults,
    genotypeData,
    clinicalData,
):
    pooledBaselineFeatureResults, baselineFeatureResultsByModel, selectedFeature = getBaselineFeatureResults(
        deepcopy(genotypeData),  # reduce dataframe attributes to baseline feature
        clinicalData,
        config,
    )

    def processResults(modelResult, baselineFeatureResult):
        # Find the common indices between the two dataframes
        common_indices = modelResult.index.intersection(baselineFeatureResult.index)

        # Filter the dataframes to only the common indices
        modelResult = modelResult.loc[common_indices]
        baselineFeatureResult = baselineFeatureResult.loc[common_indices]

        # modelResult["probabilities_list"] = modelResult["probabilities_list"].apply(lambda x: np.array(eval(x)))
        processedResults = modelResult

        relativePerplexities = pd.DataFrame(index=common_indices)
        new_cols = processedResults.apply(
            lambda row: relativePerplexity(
                [row["label"]] * len(row["probabilities_list"]),
                row["probabilities_list"],
                [row["label"]] * len(baselineFeatureResult.loc[row.name, "probabilities_list"]),
                baselineFeatureResult.loc[row.name, "probabilities_list"],
            ),
            axis=1,
            result_type="expand",
        )

        relativePerplexities["all features"], relativePerplexities[f"{selectedFeature}"], relativePerplexities["relative"] = (new_cols[0], new_cols[1], new_cols[2])
        
        discordantSamples = processedResults.loc[
            processedResults["accuracy_mean"] <= metaconfig["samples"]["discordantThreshold"],
        ]
        accurateSamples = processedResults.loc[
            processedResults["accuracy_mean"] >= metaconfig["samples"]["accurateThreshold"],
        ]

        return relativePerplexities, discordantSamples, accurateSamples


    pooledRelativePerplexities, pooledDiscordantSamples, pooledAccurateSamples = processResults(pooledResults, pooledBaselineFeatureResults)
    
    relativePerplexitiesByModel = {}
    discordantSamplesByModel = {}
    accurateSamplesByModel = {}

    for model in modelStack:
        modelName = model.__class__.__name__
        baselineFeatureResult = baselineFeatureResultsByModel[modelName]
        completeFeatureResult = list(filter(lambda result: result.model_name == modelName, classificationResults.modelResults))[0].sample_results_dataframe
        relativePerplexitiesByModel[modelName], discordantSamplesByModel[modelName], accurateSamplesByModel[modelName] = processResults(completeFeatureResult, baselineFeatureResult)

    return (
        pooledRelativePerplexities,
        relativePerplexitiesByModel,
        pooledDiscordantSamples,
        pooledAccurateSamples,
        discordantSamplesByModel,
        accurateSamplesByModel,
        pooledBaselineFeatureResults,
        baselineFeatureResultsByModel
    )



def readSampleIDs(table, label):
    sampleIDs = table.loc[table["label"] == label]
    return sampleIDs.index.tolist()


def sequesterOutlierSamples(sampleResultsByModel = None, pooledSampleResults = None):
    thresholdedSamplesByModel = {}
    thresholdedSamplesByModel["accurate"] = {}
    thresholdedSamplesByModel["discordant"] = {}
    
    for modelName, sampleResults in sampleResultsByModel.items():
        thresholdedSamplesByModel["accurate"][modelName] = sampleResults.loc[sampleResults["accuracy_mean"] >= metaconfig["samples"]["accurateThreshold"]]
        thresholdedSamplesByModel["discordant"][modelName] = sampleResults.loc[sampleResults["accuracy_mean"] <= metaconfig["samples"]["discordantThreshold"]]
    
    thresholdedPooledSamples = {
        "accurate": pooledSampleResults.loc[pooledSampleResults["accuracy_mean"] >= metaconfig["samples"]["accurateThreshold"]],
        "discordant": pooledSampleResults.loc[pooledSampleResults["accuracy_mean"] <= metaconfig["samples"]["discordantThreshold"]],
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
def main(config):
    newWellClassified = True
    countSuffix = 1
    baseProjectPath = config["tracking"]["project"]
    genotypeData, clinicalData = None, None
    while newWellClassified:
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        
        if countSuffix <= metaconfig["tracking"]["lastIteration"]:
            sampleResultsByModel = {
                model.__class__.__name__: pd.read_csv(f"projects/{config['tracking']['project']}/{model.__class__.__name__}/sampleResults.csv", index_col="id") for model in modelStack.keys()
            }
            pooledSampleResults = pd.read_csv(f"projects/{config['tracking']['project']}/pooledSampleResults.csv", index_col="id")
            
            config = sequesterOutlierSamples(sampleResultsByModel, pooledSampleResults)
            countSuffix += 1
            continue
        else:
            classificationResults, genotypeData, clinicalData = runMLstack(config, genotypeData, clinicalData)
            config["sampling"]["lastIteration"] = 0
        
        pooledResults = poolSampleResults(pd.concat([modelResult.sample_results_dataframe for modelResult in classificationResults.modelResults]))
        
        (
            pooledSamplePerplexities,
            relativePerplexitiesByModel,
            pooledDiscordantSamples,
            pooledAccurateSamples,
            discordantSamplesByModel,
            accurateSamplesByModel,
            pooledBaselineFeatureResults,
            baselineFeatureResultsByModel
        ) = measureIterations(
            classificationResults,
            pooledResults,
            genotypeData,
            clinicalData,
        )
        
        np.set_printoptions(threshold=np.inf)
        
        # Store pooled results
        pooledBaselineFeatureResults.to_csv(
            f"projects/{config['tracking']['project']}/pooledBaselineFeatureResults.csv"
        )
        pd.concat([pooledSamplePerplexities, pooledResults[['label', 'accuracy_mean', 'accuracy_std', 'draw_count']]], axis=1).to_csv(
            f"projects/{config['tracking']['project']}/pooledSamplePerplexities.csv"
        )
        pooledAccurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledAccurateSamples.csv"
        )
        pooledDiscordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledDiscordantSamples.csv"
        )
        
        sampleResultsByModel = {
                model.__class__.__name__: pd.read_csv(f"projects/{config['tracking']['project']}/{model.__class__.__name__}/sampleResults.csv", index_col="id") for model in modelStack.keys()
            }
        
        # Store results for each model
        for modelName in relativePerplexitiesByModel.keys():
            baselineFeatureResultsByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/baselineFeatureResults.csv"
            )
            pd.concat([relativePerplexitiesByModel[modelName], sampleResultsByModel[modelName][['label', 'accuracy_mean', 'accuracy_std', 'draw_count']]], axis=1).to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/relativePerplexities.csv"
            )
            accurateSamplesByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/accurateSamples.csv"
            )
            discordantSamplesByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/discordantSamples.csv"
            )
            
        # Check for new well-classified samples. If not present, stop iterations.
        newWellClassified = not pooledAccurateSamples.empty
        
        config = sequesterOutlierSamples(sampleResultsByModel, pooledResults)
        countSuffix += 1



if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()
    main(config)
    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())
