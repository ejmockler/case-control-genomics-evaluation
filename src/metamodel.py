import asyncio
from copy import deepcopy
from traceback import print_exception
from joblib import Parallel, delayed
import pandas as pd
from prefect import flow, task
import numpy as np
from numpy import array, float32  # to eval strings as arrays
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random

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
from src_rebase.models import stack as modelStack
from tasks.vcf.input import processInputFiles
from tasks.vcf.visualize import poolSampleResults


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
    freqReferenceGenotypeData,
    clinicalData,
    config,
):
    selectedFeatures = findBaselineFeatures(
        genotypeData.case.genotype,
        genotypeData.control.genotype,
        genotypeData.holdout_case.genotype,
        genotypeData.holdout_control.genotype,
    )
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator

    genotypeData.case.genotype = genotypeData.case.genotype.loc[selectedFeatures]
    genotypeData.control.genotype = genotypeData.control.genotype.loc[selectedFeatures]
    genotypeData.holdout_case.genotype = genotypeData.holdout_case.genotype.loc[
        selectedFeatures
    ]
    genotypeData.holdout_control.genotype = genotypeData.holdout_control.genotype.loc[
        selectedFeatures
    ]

    bootstrap_args = [
        (
            genotypeData,
            freqReferenceGenotypeData,
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
    try:
        # TODO: represent in workflow engine
        results = Parallel(n_jobs=-1)(
            delayed(bootstrap)(*args) for args in bootstrap_args
        )
    except Exception as e:
        print(f"Error during bootstrap: {e}")
        print_exception(e)

    pooledBaselineFeatureResults = {}
    baselineFeatureResultsByModel = {}
    for i, model in enumerate(modelStack.keys()):
        modelResults = results[i]
        baselineFeatureResultsByModel[model.__class__.__name__] = (
            modelResults.test_results_dataframe
        )
    pooledBaselineFeatureResults = poolSampleResults(
        pd.concat([df for df in baselineFeatureResultsByModel.values()])
    )

    return pooledBaselineFeatureResults, baselineFeatureResultsByModel, selectedFeatures


def perplexity(y_true, y_pred):
    crossEntropy = log_loss(
        y_true, y_pred, labels=[0, 1], eps=1e-15
    )  # linear predictions (exactly 0 or 1) depend on offset of 1e-15 when log is applied to avoid NaN

    # perplexity = 2 ^ crossEntropy
    return np.power(2, crossEntropy)


def findBaselineFeatures(
    caseGenotypes, controlGenotypes, holdoutCaseGenotypes, holdoutControlGenotypes
):
    # Drop variant rows where any element is NaN in each DataFrame
    caseGenotypes = caseGenotypes.dropna()
    controlGenotypes = controlGenotypes.dropna()
    holdoutCaseGenotypes = holdoutCaseGenotypes.dropna()
    holdoutControlGenotypes = holdoutControlGenotypes.dropna()

    # Ensure that features (rows) are consistent across all DataFrames after dropping NaNs
    common_features = caseGenotypes.index.intersection(controlGenotypes.index)
    common_features = common_features.intersection(holdoutCaseGenotypes.index)
    common_features = common_features.intersection(holdoutControlGenotypes.index)

    # Select only the rows corresponding to the common features in each DataFrame
    caseGenotypes = caseGenotypes.loc[common_features]
    controlGenotypes = controlGenotypes.loc[common_features]
    holdoutCaseGenotypes = holdoutCaseGenotypes.loc[common_features]
    holdoutControlGenotypes = holdoutControlGenotypes.loc[common_features]

    # Combine case and control genotypes into a single dataset with labels
    allGenotypes = pd.concat([caseGenotypes, controlGenotypes], axis=1)
    labels = ["case"] * caseGenotypes.shape[1] + ["control"] * controlGenotypes.shape[
        1
    ]  # Labels for LDA

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=1)  # Assuming binary classification
    lda.fit(allGenotypes.T, labels)  # Transpose to have features as columns

    # Identify the most important features
    # LDA doesn't explicitly rank features, but you can look at the absolute coefficients
    coefficients = pd.Series(lda.coef_[0], index=allGenotypes.index)
    important_features = coefficients.abs().sort_values(ascending=False)

    print("Most important features for distinguishing between cases and controls:")
    print(important_features.head())  # Print or return the top 5% features
    return important_features.iloc[
        : int(np.ceil(0.05 * len(important_features)))
    ].index.tolist()


@task(retries=10)
def measureIterations(
    sampleResultsByModel,
    pooledResults,
    genotypeData,
    freqReferenceGenotypeData,
    clinicalData,
):
    # reduce dataframe attributes to baseline feature & run models
    pooledBaselineFeatureResults, baselineFeatureResultsByModel, selectedFeatures = (
        getBaselineFeatureResults(
            deepcopy(genotypeData),
            freqReferenceGenotypeData,
            clinicalData,
            config,
        )
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
                [row["label"]]
                * len(baselineFeatureResult.loc[row.name, "probabilities_list"]),
                baselineFeatureResult.loc[row.name, "probabilities_list"],
            ),
            axis=1,
            result_type="expand",
        )

        (
            relativePerplexities["all features"],
            relativePerplexities[
                f"{' | '.join([str(feature) for feature in selectedFeatures])}"
            ],
            relativePerplexities["relative"],
        ) = (new_cols[0], new_cols[1], new_cols[2])

        discordantSamples = processedResults.loc[
            processedResults["accuracy_mean"]
            <= metaconfig["samples"]["discordantThreshold"],
        ]
        accurateSamples = processedResults.loc[
            processedResults["accuracy_mean"]
            >= metaconfig["samples"]["accurateThreshold"],
        ]

        return relativePerplexities, discordantSamples, accurateSamples

    pooledRelativePerplexities, pooledDiscordantSamples, pooledAccurateSamples = (
        processResults(pooledResults, pooledBaselineFeatureResults)
    )

    relativePerplexitiesByModel = {}
    discordantSamplesByModel = {}
    accurateSamplesByModel = {}

    for model in modelStack:
        modelName = model.__class__.__name__
        baselineFeatureResult = baselineFeatureResultsByModel[modelName]
        completeFeatureResult = sampleResultsByModel[modelName]
        (
            relativePerplexitiesByModel[modelName],
            discordantSamplesByModel[modelName],
            accurateSamplesByModel[modelName],
        ) = processResults(completeFeatureResult, baselineFeatureResult)

    return (
        pooledRelativePerplexities,
        relativePerplexitiesByModel,
        pooledDiscordantSamples,
        pooledAccurateSamples,
        discordantSamplesByModel,
        accurateSamplesByModel,
        pooledBaselineFeatureResults,
        baselineFeatureResultsByModel,
    )


def readSampleIDs(table, label):
    sampleIDs = table.loc[table["label"] == label]
    return sampleIDs.index.tolist()


def sequesterOutlierSamples(
    sampleResultsByModel=None, pooledSampleResults=None, config=None, metaconfig=None
):
    thresholdedSamplesByModel = {"accurate": {}, "discordant": {}}

    # Initialize sequesteredIDs as a dictionary by modelName
    if not all(
        modelName in config["sampling"]["sequesteredIDs"]
        for modelName in list(sampleResultsByModel.keys())
    ):
        config["sampling"]["sequesteredIDs"] = {
            modelName: [] for modelName in sampleResultsByModel.keys()
        }

    for modelName, sampleResults in sampleResultsByModel.items():
        thresholdedSamplesByModel["accurate"][modelName] = sampleResults.loc[
            sampleResults["accuracy_mean"] >= metaconfig["samples"]["accurateThreshold"]
        ]
        thresholdedSamplesByModel["discordant"][modelName] = sampleResults.loc[
            sampleResults["accuracy_mean"]
            <= metaconfig["samples"]["discordantThreshold"]
        ]

    # Handle individual model samples
    for sampleType, modelSet in thresholdedSamplesByModel.items():
        for modelName, dataframe in modelSet.items():
            for labelType, label in {
                config["clinicalTable"]["caseAlias"]: 1,
                config["clinicalTable"]["controlAlias"]: 0,
            }.items():
                if metaconfig["samples"]["sequester"][sampleType][labelType]:
                    ids_to_sequester = readSampleIDs(dataframe, label=label)
                    if (
                        labelType
                        not in metaconfig["samples"]["sequester"][sampleType]["random"]
                    ):
                        config["sampling"]["sequesteredIDs"][modelName].extend(
                            ids_to_sequester
                        )

                    # Handle random sequestering
                    else:
                        random_label = (
                            1
                            if labelType == config["clinicalTable"]["caseAlias"]
                            else 0
                        )
                        ids_to_sample = readSampleIDs(
                            sampleResultsByModel[modelName], random_label
                        )
                        # Randomly select IDs matching the length of sequestered ids
                        random_sample_ids = random.sample(
                            ids_to_sample, len(ids_to_sequester)
                        )
                        config["sampling"]["sequesteredIDs"][modelName].extend(
                            random_sample_ids
                        )
                        print(
                            f"Randomly sequestered {len(random_sample_ids)} {labelType} samples from {modelName}"
                        )

    # Remove duplicates by converting to a set and back to a list for each model
    for modelName in config["sampling"]["sequesteredIDs"]:
        config["sampling"]["sequesteredIDs"][modelName] = list(
            set(config["sampling"]["sequesteredIDs"][modelName])
        )

    return config


@flow()
def main(config):
    newWellClassified = True
    countSuffix = 1
    baseProjectPath = config["tracking"]["project"]
    genotypeData, freqReferenceGenotypeData, clinicalData = processInputFiles(config)
    originalTrackingName = config["tracking"]["name"]
    while newWellClassified:
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        if countSuffix > 1:
            config["tracking"]["name"] = (
                f">={'{:.1%}'.format(metaconfig['samples']['accurateThreshold'])} accurate cases removed per model, pass {countSuffix}\n"
                + originalTrackingName
            )

        if countSuffix <= metaconfig["tracking"]["lastIteration"]:
            sampleResultsByModel = {
                model.__class__.__name__: pd.read_csv(
                    f"projects/{config['tracking']['project']}/{model.__class__.__name__}/testResults_{model.__class__.__name__}_{config['tracking']['project']}.csv",
                    index_col="id",
                )
                for model in modelStack.keys()
            }

            for dataframe in sampleResultsByModel.values():
                dataframe["probabilities_list"] = dataframe["probabilities_list"].apply(
                    lambda x: np.array(eval(x))
                )  # convert string representation of array to numpy array

            pooledTestResults = pd.read_csv(
                f"projects/{config['tracking']['project']}/pooledTestResults_{config['tracking']['project']}.csv",
                index_col="id",
            )
            config = sequesterOutlierSamples(
                sampleResultsByModel,
                pooledTestResults,
                config=config,
                metaconfig=metaconfig,
            )

            countSuffix += 1
            continue

        else:
            classificationResults, genotypeData, clinicalData = runMLstack(
                config, genotypeData, freqReferenceGenotypeData, clinicalData
            )
            sampleResultsByModel = {
                modelResult.model_name: modelResult.test_results_dataframe
                for modelResult in classificationResults.modelResults
            }
            config["sampling"]["lastIteration"] = 0

            pooledTestResults = poolSampleResults(
                pd.concat(
                    [
                        modelResult.test_results_dataframe
                        for modelResult in classificationResults.modelResults
                    ]
                )
            )
            pooledHoldoutResults = poolSampleResults(
                pd.concat(
                    [
                        modelResult.holdout_results_dataframe
                        for modelResult in classificationResults.modelResults
                    ]
                )
            )
            pooledExcessResults = poolSampleResults(
                pd.concat(
                    [
                        modelResult.excess_results_dataframe
                        for modelResult in classificationResults.modelResults
                    ]
                )
            )

        (
            pooledSamplePerplexities,
            relativePerplexitiesByModel,
            pooledDiscordantSamples,
            pooledAccurateSamples,
            discordantSamplesByModel,
            accurateSamplesByModel,
            pooledBaselineFeatureResults,
            baselineFeatureResultsByModel,
        ) = measureIterations(
            sampleResultsByModel,
            pooledTestResults,
            genotypeData,
            freqReferenceGenotypeData,
            clinicalData,
        )

        np.set_printoptions(threshold=np.inf)

        # Store pooled results
        pooledBaselineFeatureResults.to_csv(
            f"projects/{config['tracking']['project']}/pooledBaselineFeatureResults_{config['tracking']['project']}.csv"
        )
        pd.concat(
            [
                pooledSamplePerplexities,
                pooledTestResults[
                    ["label", "accuracy_mean", "accuracy_std", "draw_count"]
                ],
            ],
            axis=1,
        ).to_csv(
            f"projects/{config['tracking']['project']}/pooledSamplePerplexities_{config['tracking']['project']}.csv"
        )
        pooledAccurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledAccurateSamples_{config['tracking']['project']}.csv"
        )
        pooledDiscordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/pooledDiscordantSamples_{config['tracking']['project']}.csv"
        )

        sampleResultsByModel = {
            model.__class__.__name__: pd.read_csv(
                f"projects/{config['tracking']['project']}/{model.__class__.__name__}/testResults_{model.__class__.__name__}_{config['tracking']['project']}.csv",
                index_col="id",
            )
            for model in modelStack.keys()
        }

        # Store results for each model
        for modelName in relativePerplexitiesByModel.keys():
            baselineFeatureResultsByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/baselineFeatureResults_{modelName}_{config['tracking']['project']}.csv"
            )
            pd.concat(
                [
                    relativePerplexitiesByModel[modelName],
                    sampleResultsByModel[modelName][
                        ["label", "accuracy_mean", "accuracy_std", "draw_count"]
                    ],
                ],
                axis=1,
            ).to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/relativePerplexities_{modelName}_{config['tracking']['project']}.csv"
            )
            accurateSamplesByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/accurateSamples_{modelName}_{config['tracking']['project']}.csv"
            )
            discordantSamplesByModel[modelName].to_csv(
                f"projects/{config['tracking']['project']}/{modelName}/discordantSamples_{modelName}_{config['tracking']['project']}.csv"
            )

        # Check for new well-classified samples. If not present, stop iterations.
        newWellClassified = not pooledAccurateSamples.empty

        config = sequesterOutlierSamples(
            sampleResultsByModel,
            pooledTestResults,
            config=config,
            metaconfig=metaconfig,
        )
        countSuffix += 1


if __name__ == "__main__":
    ray.shutdown()
    main(config)
    clearHistory = True
    if clearHistory:
        asyncio.run(remove_all_flows())
