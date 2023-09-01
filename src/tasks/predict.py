from io import StringIO
import json
import os
import pickle
import matplotlib

from tasks.data import BootstrapResult, EvaluationResult, FoldResult, SampleData

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


def getFeatureImportances(model, testData, holdoutData, featureLabels, config):
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

    shapValues = None
    holdoutShapValues = None
    shapExplainer = None
    masker = None

    faulthandler.enable()
    if config["model"]["calculateShapelyExplanations"]:
        # Cluster correlated and hierarchical features using masker
        masker = shap.maskers.Partition(testData.vectors, clustering="correlation")

        # Create explainer
        shapExplainer = shap.explainers.Permutation(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            masker,
            feature_names=["_".join(label) for label in featureLabels],
        )

        # Get SHAP values
        shapValuesArray = shapExplainer(testData.vectors).values

        # Convert to DataFrame
        shapValues = pd.DataFrame(
            data=shapValuesArray, index=testData.ids, columns=featureLabels
        )
        shapValues.index.name = "sample_id"

        # Same for holdout data
        if len(holdoutData.vectors) > 0:
            holdoutShapValuesArray = shapExplainer(holdoutData.vectors).values

            holdoutShapValues = pd.DataFrame(
                data=holdoutShapValuesArray,
                index=holdoutData.ids,
                columns=featureLabels,
            )
            holdoutShapValues.index.name = "sample_id"

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
        n_iter=20,
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


def trackResults(runID: str, evaluationResult: EvaluationResult, config):
    runPath = runID
    sampleResultsDataframe = evaluationResult.create_sample_results_dataframe()
    sampleResultsDataframe.to_csv(f"{runPath}/sampleResults.csv")
    for k in range(config["sampling"]["crossValIterations"]):
        if config["model"]["hyperparameterOptimization"]:
            hyperparameterDir = f"{runPath}/hyperparameters"
            os.makedirs(hyperparameterDir, exist_ok=True)
            with open(f"{hyperparameterDir}/{k+1}.json", "w") as file:
                json.dump(evaluationResult.test[k].fitted_optimizer.best_params_, file)

        os.makedirs(f"{runPath}/testLabels", exist_ok=True)
        os.makedirs(f"{runPath}/trainLabels", exist_ok=True)

        pd.Series(
            evaluationResult.test[k].labels,
            index=evaluationResult.test[k].ids,
            name="testLabels",
        ).to_csv(f"{runPath}/testLabels/{k+1}.csv")
        pd.Series(
            evaluationResult.train[k].labels,
            index=evaluationResult.train[k].ids,
            name="trainLabels",
        ).to_csv(f"{runPath}/trainLabels/{k+1}.csv")

        if len(evaluationResult.holdout[k].labels) > 0:
            os.makedirs(f"{runPath}/holdoutLabels", exist_ok=True)
            pd.Series(
                evaluationResult.holdout[k].labels,
                index=evaluationResult.holdout[k].ids,
                name="holdoutLabels",
            ).to_csv(f"{runPath}/holdoutLabels/{k+1}.csv")

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
            if len(evaluationResult.holdout[k].labels) > 0:
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
    trainData: SampleData,
    testData: SampleData,
    holdoutData: SampleData,
    model,
    hyperParameterSpace,
    cvIterator,
    variantIndex,
    config,
):
    if config["model"]["hyperparameterOptimization"]:
        fittedOptimizer = optimizeHyperparameters(
            trainData.vectors,
            trainData.labels,
            model,
            hyperParameterSpace,
            cvIterator,
            "neg_mean_squared_error",
        )
        model.set_params(**fittedOptimizer.best_params_)
    else:
        fittedOptimizer = None

    model.fit(trainData.vectors, trainData.labels)

    probabilities = get_probabilities(model, testData.vectors)
    holdoutProbabilities = get_probabilities(model, holdoutData.vectors)

    (
        modelValues,
        shapValues,
        holdoutShapValues,
        shapExplainer,
        shapMasker,
    ) = getFeatureImportances(model, testData, holdoutData, variantIndex, config)

    return (
        FoldResult(
            testData.ids,
            testData.labels,
            testData.vectors,
            probabilities,
            modelValues,
            shapValues,
            fittedOptimizer,
            shapExplainer,
            shapMasker,
        ),
        FoldResult(
            holdoutData.ids,
            holdoutData.labels,
            holdoutData.vectors,
            holdoutProbabilities,
            modelValues,
            holdoutShapValues,
        ),
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
    config,
    track=True,
):
    embedding = prepareDatasets(
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
        verbose=(True if runNumber == 0 else False),
    )

    clinicalIDs = list()

    for id in embedding["sampleIndex"] + embedding["holdoutSampleIndex"]:
        clinicalIDs.extend(
            id.split(config["vcfLike"]["compoundSampleIdDelimiter"])[
                config["vcfLike"]["compoundSampleMetaIdStartIndex"]
            ]
        )

    # TODO get counts via instance functions of embedding
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

    print(f"Iteration {runNumber+1} with model {model.__class__.__name__}")

    if track:
        runID = beginTracking(
            model, runNumber, embedding, clinicalData, clinicalIDs, config
        )

    # outer cross-validation
    crossValIndices = list(
        outerCvIterator.split(embedding["samples"], embedding["labels"])
    )

    holdoutData = SampleData(
        set="holdout",
        ids=embedding["holdoutSamples"],
        labels=embedding["holdoutLabels"],
    )

    evaluate_args = [
        (
            SampleData(
                set="train",
                ids=embedding["sampleIndex"][trainIndices],
                labels=embedding["labels"][trainIndices],
            ),
            SampleData(
                set="test",
                ids=embedding["sampleIndex"][testIndices],
                labels=embedding["labels"][testIndices],
            ),
            holdoutData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            embedding["variantIndex"],
            config,
        )
        for trainIndices, testIndices in crossValIndices
    ]

    modelResults = EvaluationResult()
    # run sequentially since models are not concurrency-safe
    for args in evaluate_args:
        # inner cross-validation is hyperparameter optimization
        testResult, holdoutResult = evaluate(*args)
        modelResults.train.append(args[0])
        modelResults.test.append(testResult)
        modelResults.holdout.append(holdoutResult)
        gc.collect()
    modelResults.average()

    if track:
        trackResults(runID, modelResults, config)

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

    return modelResults


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
    bootstrap = BootstrapResult(model.__class__.__name__)

    # parallelize with workflow engine in cluster environment
    for runNumber in range(
        config["sampling"]["lastIteration"],
        config["sampling"]["bootstrapIterations"],
    ):
        # update results for every bootstrap iteration
        bootstrap.iteration_results.append(
            classify(
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
                config,
                track,
            )
        )
        gc.collect()

    return bootstrap
