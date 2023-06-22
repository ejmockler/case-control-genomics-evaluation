from io import StringIO
import json
import os
import pickle
from fastnumbers import check_real
import matplotlib

matplotlib.use("agg")
from neptune.types import File
from prefect import task

from skopt import BayesSearchCV

from config import config
from copy import deepcopy

import pandas as pd
import numpy as np
import neptune
import shap

import gc


def getFeatureImportances(model, data, holdoutData, featureLabels):
    """Get feature importances from fitted model and create SHAP explainer"""
    if model.__class__.__name__ == "MultinomialNB":
        modelCoefficientDF = pd.DataFrame()
        for i, c in enumerate(
            model.feature_count_[0]
            if len(model.feature_count_.shape) > 1
            else model.feature_count_
        ):
            modelCoefficientDF.loc[
                i, f"feature_importances_{config['clinicalTable']['controlAlias']}"
            ] = model.feature_log_prob_[0][i]
            modelCoefficientDF.loc[
                i, f"feature_importances_{config['clinicalTable']['caseAlias']}"
            ] = model.feature_log_prob_[1][i]
    elif hasattr(model, "coef_"):
        modelCoefficientDF = pd.DataFrame()
        if len(model.coef_.shape) > 1:
            try:
                modelCoefficientDF[
                    f"feature_importances_{config['clinicalTable']['controlAlias']}"
                ] = model.coef_[0]
                modelCoefficientDF[
                    f"feature_importances_{config['clinicalTable']['caseAlias']}"
                ] = model.coef_[1]
            except IndexError:
                modelCoefficientDF = pd.DataFrame()
                modelCoefficientDF[f"feature_importances"] = model.coef_[0]
        else:
            modelCoefficientDF[f"feature_importances"] = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        modelCoefficientDF = pd.DataFrame()
        modelCoefficientDF[f"feature_importances"] = model.feature_importances_
    else:
        modelCoefficientDF = None

    if type(modelCoefficientDF) == pd.DataFrame:
        modelCoefficientDF.index = featureLabels
        modelCoefficientDF.index.name = "feature_name"

    if config["model"]["calculateShapelyExplanations"]:
        # Cluster correlated and hierarchical features using masker
        masker = shap.maskers.Partition(data, clustering="correlation")
        shapExplainer = shap.explainers.Permutation(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            masker,
            feature_names=["_".join(label) for label in featureLabels],
        )
        shapValues = shapExplainer(data)
        holdoutShapValues = []
        if len(holdoutData) > 0:
            holdoutShapValues = shapExplainer(holdoutData)
    else:
        shapExplainer = None
        shapValues = None
        masker = None
        holdoutShapValues = None
    return modelCoefficientDF, shapValues, holdoutShapValues, shapExplainer, masker


def get_probabilities(model, samples):
    try:
        probabilities = model.predict_proba(samples)
    except AttributeError:
        probabilities = model.predict(samples)
        if len(probabilities.shape) <= 1:
            probabilities = np.array([[1 - p, p] for p in probabilities])
    return probabilities


def optimizeHyperparameters(
    samples, labels, model, parameterSpace, cvIterator, metricFunction, n_jobs=1
):
    # hyperparameter search (inner cross-validation)
    optimizer = BayesSearchCV(
        model,
        parameterSpace,
        cv=cvIterator,
        n_jobs=n_jobs,
        n_points=2,
        return_train_score=True,
        n_iter=50,
        scoring=metricFunction,
    )
    optimizer.fit(samples, labels)
    return optimizer


def serializeDataFrame(dataframe):
    stream = StringIO()
    dataframe.to_csv(stream)
    return File.from_stream(stream, extension="csv")


@task()
def beginTracking(model, runNumber, embedding, clinicalData, clinicalIDs):
    embeddingDF = pd.DataFrame(
        data=embedding["samples"],
        columns=embedding["variantIndex"],
        index=embedding["sampleIndex"],
    )
    if "holdoutSamples" in embedding:
        holdoutEmbeddingDF = pd.DataFrame(
            data=embedding["holdoutSamples"],
            columns=embedding["variantIndex"],
            index=embedding["holdoutSampleIndex"],
        )
    embeddingDF.index.name = "id"
    if config["tracking"]["remote"]:
        runTracker = neptune.init_run(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            api_token=config["tracking"]["token"],
        )
        runTracker["sys/tags"].add(model.__class__.__name__)
        runTracker["bootstrapIteration"] = runNumber + 1
        runTracker["config"] = {
            key: (item if check_real(item) or isinstance(item, str) else str(item))
            for key, item in config.items()
        }

        runTracker["embedding"].upload(serializeDataFrame(embeddingDF))
        if "holdoutSamples" in embedding:
            runTracker["holdoutEmbedding"].upload(
                serializeDataFrame(holdoutEmbeddingDF)
            )
        runTracker["clinicalData"].upload(
            serializeDataFrame(clinicalData.loc[clinicalData.index.isin(clinicalIDs)])
        )

        runTracker["nVariants"] = len(embedding["variantIndex"])
        runID = runTracker["sys/id"].fetch()
        runTracker.stop()
    else:
        runPath = f"projects/{config['tracking']['project']}/bootstraps/{runNumber+1}/{model.__class__.__name__}"
        if not os.path.exists(runPath):
            os.makedirs(runPath, exist_ok=True)
        with open(f"{runPath}/config.pkl", "wb") as file:
            pickle.dump(config, file)
        embeddingDF.to_csv(f"{runPath}/embedding.csv")
        if "holdoutSamples" in embedding:
            holdoutEmbeddingDF.to_csv(f"{runPath}/holdoutEmbedding.csv")
        clinicalData.loc[clinicalData.index.isin(clinicalIDs)].to_csv(
            f"{runPath}/clinicalData.csv"
        )
        # hack to log metrics as filenames
        with open(f"{runPath}/nVariants_{len(embedding['variantIndex'])}", "w") as file:
            pass
        runID = runPath
    return runID


@task()
def trackResults(runID, current):
    sampleResultsDataframe = pd.DataFrame.from_dict(
        {
            "probability": [
                probability[1]
                for foldResults in [
                    *current["probabilities"],
                    *current["holdoutProbabilities"],
                ]
                for probability in foldResults
            ],
            "id": [
                id
                for foldResults in [*current["testIDs"], *current["holdoutIDs"]]
                for id in foldResults
            ],
        },
        dtype=object,
    ).set_index("id")

    # TODO debug inaccurate remote logging
    if config["tracking"]["remote"]:
        pass
        # runTracker = neptune.init_run(
        #     project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
        #     with_id=runID,
        #     api_token=config["tracking"]["token"],
        # )
        # if config["model"]["hyperparameterOptimization"]:
        #     runTracker["modelParams"] = {
        #         k + 1: current["fittedOptimizer"][k].best_params_
        #         for k in range(config["sampling"]["crossValIterations"])
        #     }

        # runTracker["sampleResults"].upload(serializeDataFrame(sampleResultsDataframe))

        # if config["model"]["calculateShapelyExplanations"]:
        #     runTracker["shapExplanationsPerFold"].upload(
        #         File.as_pickle(current["localExplanations"])
        #     )
        #     runTracker["holdout/shapExplanationsPerFold"].upload(
        #         File.as_pickle(current["holdoutLocalExplanations"])
        #     )
        #     runTracker["shapExplainersPerFold"].upload(
        #         File.as_pickle(current["shapExplainer"])
        #     )
        #     runTracker["shapMaskersPerFold"].upload(
        #         File.as_pickle(current["shapMasker"])
        #     )
        #     runTracker["featureImportance/shapelyExplanations/average"].upload(
        #         serializeDataFrame(current["averageShapelyExplanations"])
        #     )

        # if current["globalExplanations"][0] is not None:
        #     runTracker[f"featureImportance/modelCoefficients/average"].upload(
        #         serializeDataFrame(current["averageGlobalExplanations"])
        #     )

        # for k in range(config["sampling"]["crossValIterations"]):
        #     testLabelsSeries = pd.Series(current["testLabels"][k], name="testLabel")
        #     trainLabelsSeries = pd.Series(current["trainLabels"][k], name="trainLabel")
        #     testLabelsSeries.index = current["testIDs"][k]
        #     testLabelsSeries.index.name = "id"
        #     trainLabelsSeries.index = current["trainIDs"][k]
        #     trainLabelsSeries.index.name = "id"
        #     runTracker[f"trainIDs/{k+1}"].upload(
        #         serializeDataFrame(pd.Series(current["trainIDs"][k]))
        #     )
        #     runTracker[f"testIDs/{k+1}"].upload(
        #         serializeDataFrame(pd.Series(current["testIDs"][k]))
        #     )
        #     runTracker[f"testLabels/{k+1}"].upload(
        #         serializeDataFrame(pd.Series(testLabelsSeries))
        #     )
        #     runTracker[f"trainLabels/{k+1}"].upload(
        #         serializeDataFrame(pd.Series(trainLabelsSeries))
        #     )
        #     if current["globalExplanations"][k] is not None:
        #         runTracker[f"featureImportance/modelCoefficients/{k+1}"].upload(
        #             serializeDataFrame(current["globalExplanations"][k])
        #         )
        #     if config["model"]["calculateShapelyExplanations"]:
        #         runTracker[f"featureImportance/shapelyExplanations/{k+1}"].upload(
        #             serializeDataFrame(
        #                 pd.DataFrame.from_dict(
        #                     {
        #                         "feature_name": [
        #                             name
        #                             for name in current["localExplanations"][
        #                                 0
        #                             ].feature_names
        #                         ],
        #                         "value": [
        #                             np.mean(
        #                                 current["localExplanations"][k].values[
        #                                     :, featureIndex
        #                                 ]
        #                             )
        #                             for featureIndex in range(
        #                                 len(
        #                                     current["localExplanations"][
        #                                         0
        #                                     ].feature_names
        #                                 )
        #                             )
        #                         ],
        #                         "standard_deviation": [
        #                             np.std(
        #                                 current["localExplanations"][k].values[
        #                                     :, featureIndex
        #                                 ]
        #                             )
        #                             for featureIndex in range(
        #                                 len(
        #                                     current["localExplanations"][
        #                                         0
        #                                     ].feature_names
        #                                 )
        #                             )
        #                         ],
        #                     },
        #                     dtype=object,
        #                 ).set_index("feature_name")
        #             )
        #         )

        # runTracker["meanAUC"] = np.mean(current["testAUC"])
        # # average sample count across folds
        # runTracker["nTrain"] = np.mean([len(idList) for idList in current["trainIDs"]])
        # runTracker["nTest"] = np.mean([len(idList) for idList in current["testIDs"]])
        # runTracker.stop()
    else:
        runPath = runID
        for k in range(config["sampling"]["crossValIterations"]):
            if config["model"]["hyperparameterOptimization"]:
                hyperparameterDir = f"{runPath}/hyperparameters/{k+1}"
                os.makedirs(hyperparameterDir, exist_ok=True)
                with open(f"{hyperparameterDir}/{k+1}.json", "w") as file:
                    json.dump(current["fittedOptimizer"][k].best_params_, file)

            testLabelsSeries = pd.Series(current["testLabels"][k], name="testLabel")
            testLabelsSeries.index = current["testIDs"][k]
            testLabelsSeries.index.name = "id"

            testIDsSeries = pd.Series(current["testIDs"][k], name="id")

            trainLabelsSeries = pd.Series(current["trainLabels"][k], name="trainLabel")
            trainLabelsSeries.index = current["trainIDs"][k]
            trainLabelsSeries.index.name = "id"

            trainIDsSeries = pd.Series(current["trainIDs"][k], name="id")

            os.makedirs(f"{runPath}/testLabels", exist_ok=True)
            os.makedirs(f"{runPath}/testIDs", exist_ok=True)
            os.makedirs(f"{runPath}/trainLabels", exist_ok=True)
            os.makedirs(f"{runPath}/trainIDs", exist_ok=True)

            testLabelsSeries.to_csv(f"{runPath}/testLabels/{k+1}.csv")
            testIDsSeries.to_csv(f"{runPath}/testIDs/{k+1}.csv")
            trainLabelsSeries.to_csv(f"{runPath}/trainLabels/{k+1}.csv")
            trainIDsSeries.to_csv(f"{runPath}/trainIDs/{k+1}.csv")

            if len(current["holdoutLabels"][k]) > 0:
                holdoutLabelsSeries = pd.Series(
                    current["holdoutLabels"][k], name="testLabel"
                )
                holdoutLabelsSeries.index = current["holdoutIDs"][k]
                holdoutLabelsSeries.index.name = "id"
                holdoutIDsSeries = pd.Series(current["holdoutIDs"][k], name="id")
                os.makedirs(f"{runPath}/holdoutLabels", exist_ok=True)
                os.makedirs(f"{runPath}/holdoutIDs", exist_ok=True)
                pd.Series(holdoutLabelsSeries).to_csv(
                    f"{runPath}/holdoutLabels/{k+1}.csv"
                )
                pd.Series(holdoutIDsSeries).to_csv(f"{runPath}/holdoutIDs/{k+1}.csv")

            if current["globalExplanations"][k] is not None:
                os.makedirs(
                    f"{runPath}/featureImportance/modelCoefficients", exist_ok=True
                )
                current["globalExplanations"][k].to_csv(
                    f"{runPath}/featureImportance/modelCoefficients/{k+1}.csv"
                )

            if config["model"]["calculateShapelyExplanations"]:
                os.makedirs(
                    f"{runPath}/featureImportance/shapelyExplanations", exist_ok=True
                )
                pd.DataFrame.from_dict(
                    {
                        "feature_name": [
                            name
                            for name in current["localExplanations"][0].feature_names
                        ],
                        "value": [
                            np.mean(
                                current["localExplanations"][k].values[:, featureIndex]
                            )
                            for featureIndex in range(
                                len(current["localExplanations"][0].feature_names)
                            )
                        ],
                        "standard_deviation": [
                            np.std(
                                current["localExplanations"][k].values[:, featureIndex]
                            )
                            for featureIndex in range(
                                len(current["localExplanations"][0].feature_names)
                            )
                        ],
                    },
                    dtype=object,
                ).set_index("feature_name").to_csv(
                    f"{runPath}/featureImportance/shapelyExplanations/{k+1}.csv"
                )
                if len(current["holdoutLabels"][k]) > 0:
                    os.makedirs(
                        f"{runPath}/featureImportance/shapelyExplanations/holdout",
                        exist_ok=True,
                    )
                    pd.DataFrame.from_dict(
                        {
                            "feature_name": [
                                name
                                for name in current["holdoutLocalExplanations"][
                                    0
                                ].feature_names
                            ],
                            "value": [
                                np.mean(
                                    current["holdoutLocalExplanations"][k].values[
                                        :, featureIndex
                                    ]
                                )
                                for featureIndex in range(
                                    len(
                                        current["holdoutLocalExplanations"][
                                            0
                                        ].feature_names
                                    )
                                )
                            ],
                            "standard_deviation": [
                                np.std(
                                    current["holdoutLocalExplanations"][k].values[
                                        :, featureIndex
                                    ]
                                )
                                for featureIndex in range(
                                    len(
                                        current["holdoutLocalExplanations"][
                                            0
                                        ].feature_names
                                    )
                                )
                            ],
                        },
                        dtype=object,
                    ).set_index("feature_name").to_csv(
                        f"{runPath}/featureImportance/shapelyExplanations/holdout/{k+1}.csv"
                    )

            sampleResultsDataframe.to_csv(f"{runPath}/sampleResults.csv")

        if config["model"]["calculateShapelyExplanations"]:
            with open(
                f"{runPath}/featureImportance/shapelyExplanations/shapExplainersPerFold.pkl",
                "wb",
            ) as file:
                pickle.dump(current["shapExplainer"], file)
            with open(
                f"{runPath}/featureImportance/shapelyExplanations/shapMaskersPerFold.pkl",
                "wb",
            ) as file:
                pickle.dump(current["shapMasker"], file)
            current["averageShapelyExplanations"].to_csv(
                f"{runPath}/averageLocalExplanations.csv"
            )
            if len(current["holdoutLabels"][0]) > 0:
                current["averageHoldoutShapelyExplanations"].to_csv(
                    f"{runPath}/averageHoldoutLocalExplanations.csv"
                )

        if current["globalExplanations"][0] is not None:
            current["averageGlobalExplanations"].to_csv(
                f"{runPath}/averageGlobalExplanations.csv"
            )

        with open(
            f"{runPath}/trainCount_{np.mean([len(idList) for idList in current['trainIDs']])}",
            "w",
        ) as file:
            pass
        with open(
            f"{runPath}/testCount_{np.mean([len(idList) for idList in current['testIDs']])}",
            "w",
        ) as file:
            pass
        with open(f"{runPath}/meanAUC_{np.mean(current['testAUC'])}", "w") as file:
            pass
        if "holdoutAUC" in current:
            with open(
                f"{runPath}/meanHoldoutAUC_{np.mean(current['holdoutAUC'])}", "w"
            ) as file:
                pass
            with open(
                f"{runPath}/holdoutCount_{np.mean([len(idList) for idList in current['holdoutIDs']])}",
                "w",
            ) as file:
                pass

    gc.collect()


def clone_model(model):
    ModelClass = type(model)  # Get the type of the model
    params = model.get_params()  # Get the parameters of the model
    cloned_model = ModelClass(
        **params
    )  # Initialize a new object of the same type with the same parameters
    return cloned_model


def evaluate(
    trainSamples,
    trainLabels,
    testSamples,
    testLabels,
    model,
    embedding,
    hyperParameterSpace,
    cvIterator,
    trainIDs,
    testIDs,
    variantIndex,
):
    if config["model"]["hyperparameterOptimization"]:
        fittedOptimizer = optimizeHyperparameters(
            trainSamples,
            trainLabels,
            model,
            hyperParameterSpace,
            cvIterator,
            "neg_mean_squared_error",
        )
        model.set_params(**fittedOptimizer.best_params_)
    else:
        fittedOptimizer = None

    model.fit(trainSamples, trainLabels)

    probabilities = get_probabilities(model, testSamples)
    predictions = np.argmax(probabilities, axis=1)

    holdoutSamples = []
    holdoutProbabilities = []
    holdoutPredictions = []
    holdoutIDs = []
    holdoutLabels = []
    if "holdoutSamples" in embedding:
        holdoutSamples = embedding["holdoutSamples"]
        holdoutIDs = embedding["holdoutSampleIndex"]
        holdoutLabels = embedding["holdoutLabels"]
        holdoutProbabilities = get_probabilities(model, holdoutSamples)
        holdoutPredictions = np.argmax(holdoutProbabilities, axis=1)

    (
        modelValues,
        shapValues,
        holdoutShapValues,
        shapExplainer,
        shapMasker,
    ) = getFeatureImportances(model, testSamples, holdoutSamples, variantIndex)

    globalExplanations = modelValues
    localExplanations = shapValues
    holdoutLocalExplanations = holdoutShapValues

    # TODO implement object to structure these results
    return (
        globalExplanations,
        localExplanations,
        holdoutLocalExplanations,
        probabilities,
        holdoutProbabilities,
        predictions,
        holdoutPredictions,
        testLabels,
        trainLabels,
        holdoutLabels,
        trainIDs,
        testIDs,
        holdoutIDs,
        fittedOptimizer,
        shapExplainer,
        shapMasker,
    )


def processSampleResult(fold, j, sampleID, current, results):
    probability = (
        current["probabilities"][fold][j]
        if j < len(current["testIDs"][fold])
        else current["holdoutProbabilities"][fold][j - len(current["testIDs"][fold])]
    )

    label = (
        current["testLabels"][fold][j]
        if j < len(current["testIDs"][fold])
        else current["holdoutLabels"][fold][j - len(current["testIDs"][fold])]
    )

    try:
        results["samples"][sampleID].append(probability)
    except KeyError:
        results["samples"][sampleID] = [probability]
    finally:
        results["labels"][sampleID] = label
    return results
