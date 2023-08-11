from io import StringIO
import json
import os
import pickle
import matplotlib

matplotlib.use("agg")

from sklearn.metrics import roc_auc_score, roc_curve

from tasks.input import prepareDatasets
from tasks.visualize import trackBootstrapVisualizations
from prefect_ray.task_runners import RayTaskRunner

from neptune.types import File
from prefect import flow, task

from skopt import BayesSearchCV

from config import config

import pandas as pd
import numpy as np
import shap

import gc
import faulthandler


def getFeatureImportances(model, data, holdoutData, featureLabels, config):
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

    faulthandler.enable()
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
    samples, labels, model, parameterSpace, cvIterator, metricFunction, n_jobs=-1
):
    # hyperparameter search (inner cross-validation)
    optimizer = BayesSearchCV(
        model,
        parameterSpace,
        cv=cvIterator,
        n_jobs=n_jobs,
        n_points=2,
        return_train_score=True,
        n_iter=25,
        scoring=metricFunction,
    )
    optimizer.fit(samples, labels)
    return optimizer


def serializeDataFrame(dataframe):
    stream = StringIO()
    dataframe.to_csv(stream)
    return File.from_stream(stream, extension="csv")


def beginTracking(model, runNumber, embedding, clinicalData, clinicalIDs, config):
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
    runPath = f"projects/{config['tracking']['project']}/bootstraps/{runNumber+1}/{model.__class__.__name__}"
    if not os.path.exists(runPath):
        os.makedirs(runPath, exist_ok=True)
    with open(f"{runPath}/config.json", "w") as file:
        json.dump(config, file)
    embeddingDF.to_csv(f"{runPath}/embedding.csv")
    if "holdoutSamples" in embedding:
        holdoutEmbeddingDF.to_csv(f"{runPath}/holdoutEmbedding.csv")
    clinicalData.loc[clinicalData.index.isin(clinicalIDs)].to_csv(
        f"{runPath}/clinicalData.csv"
    )
    # hack to log metric as filename
    with open(f"{runPath}/nVariants_{len(embedding['variantIndex'])}", "w") as file:
        pass
    runID = runPath
    return runID


def trackResults(runID, current, config):
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

    runPath = runID
    for k in range(config["sampling"]["crossValIterations"]):
        if config["model"]["hyperparameterOptimization"]:
            hyperparameterDir = f"{runPath}/hyperparameters"
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
            pd.Series(holdoutLabelsSeries).to_csv(f"{runPath}/holdoutLabels/{k+1}.csv")
            pd.Series(holdoutIDsSeries).to_csv(f"{runPath}/holdoutIDs/{k+1}.csv")

        if current["globalExplanations"][k] is not None:
            os.makedirs(f"{runPath}/featureImportance/modelCoefficients", exist_ok=True)
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
                        name for name in current["localExplanations"][0].feature_names
                    ],
                    "value": [
                        np.mean(current["localExplanations"][k].values[:, featureIndex])
                        for featureIndex in range(
                            len(current["localExplanations"][0].feature_names)
                        )
                    ],
                    "standard_deviation": [
                        np.std(current["localExplanations"][k].values[:, featureIndex])
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
                                    current["holdoutLocalExplanations"][0].feature_names
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
                                    current["holdoutLocalExplanations"][0].feature_names
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

        if config["model"]["hyperparameterOptimization"]:
            with open(f"{runPath}/hyperparameters/fittedOptimizer.pkl", "wb") as file:
                pickle.dump(current["fittedOptimizer"], file)

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
    config,
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
    ) = getFeatureImportances(model, testSamples, holdoutSamples, variantIndex, config)

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
    print(f"Iteration {runNumber+1} with model {model.__class__.__name__}")

    if track:
        runID = beginTracking(
            model, runNumber, embedding, clinicalData, clinicalIDs, config
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
        trackResults(runID, current, config)

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
            trackBootstrapVisualizations(
                runID,
                holdoutPlotSubtitle,
                model.__class__.__name__,
                current,
                holdout=True,
                config=config,
            )
            gc.collect()

        trackBootstrapVisualizations(
            runID,
            plotSubtitle,
            model.__class__.__name__,
            current,
            config=config,
        )

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
