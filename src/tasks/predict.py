from io import StringIO
import json
import os
import pickle
import matplotlib

from tasks.data import BootstrapResult, EvaluationResult, FoldResult, SampleData

matplotlib.use("agg")

from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, roc_curve

from tasks.input import prepareDatasets
from tasks.visualize import trackBootstrapVisualizations
from prefect_ray.task_runners import RayTaskRunner
from prefect.task_runners import ConcurrentTaskRunner

from neptune.types import File
from prefect import flow, task

from skopt import BayesSearchCV

from config import config

import pandas as pd
import numpy as np
import shap

import gc
import faulthandler


def getFeatureImportances(model, testData, holdoutData, excessData, featureLabels, config):
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
    excessShapValues = None
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
            max_evals=2 * len(featureLabels) + 1,
        )

        # Get SHAP values
        shapValues = shapExplainer(testData.vectors)
        
        # Same for excess data
        if len(excessData.vectors) > 0:
            excessShapValues = shapExplainer(excessData.vectors)

        # Same for holdout data
        if len(holdoutData.vectors) > 0:
            holdoutShapValues = shapExplainer(holdoutData.vectors)
            
        
    return modelCoefficientDF, shapValues, excessShapValues, holdoutShapValues, shapExplainer, masker


def get_probabilities(model, samples):
    try:
        probabilities = model.predict_proba(samples)
    except AttributeError:
        probabilities = model.predict(samples)
        if len(probabilities.shape) <= 1:
            probabilities = np.array([[1 - p, p] for p in probabilities])
    return probabilities

def control_group_scorer(y, X, control_label=0):
    """
    Custom scoring function that calculates the negative mean squared error for the control group.
    
    Parameters:
    - estimator: The model being evaluated.
    - X: The input features for the test set.
    - y: The true labels for the test set.
    
    Returns:
    - score: The negative mean squared error of the predictions.
    """
    # Assuming '0' represents controls and '1' represents ALS patients in 'y'
    # Modify the condition below if your labeling differs.
    control_indices = np.where(y == control_label)  # Identify control group samples
    
    # Calculate MSE for the control group
    mse = mean_squared_error(y[control_indices], X[control_indices])
    
    # Return the negative MSE because BayesSearchCV maximizes the score
    return -mse


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
        n_iter=15,
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
    clinicalData.loc[clinicalData.index.intersection(clinicalIDs)].to_csv(
        f"{runPath}/clinicalData.csv"
    )
    # hack to log metric as filename
    with open(f"{runPath}/nFeatures_{len(embedding['variantIndex'])}", "w") as file:
        pass
    runID = runPath
    return runID


def trackResults(runID: str, evaluationResult: EvaluationResult, config):
    runPath = runID

    testResultsDataframe = evaluationResult.test_results_dataframe
    testResultsDataframe.to_csv(f"{runPath}/testResults.csv")
    holdoutResultsDataframe = evaluationResult.holdout_results_dataframe
    holdoutResultsDataframe.to_csv(f"{runPath}/holdoutResults.csv")
    excessResultsDataframe = evaluationResult.excess_results_dataframe
    excessResultsDataframe.to_csv(f"{runPath}/excessResults.csv")
    
    evaluationResult.average()
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
            name="labels",
        ).to_csv(f"{runPath}/testLabels/{k+1}.csv")
        pd.Series(
            evaluationResult.train[k].labels,
            index=evaluationResult.train[k].ids,
            name="labels",
        ).to_csv(f"{runPath}/trainLabels/{k+1}.csv")

        if len(evaluationResult.holdout[k].labels) > 0:
            os.makedirs(f"{runPath}/holdoutLabels", exist_ok=True)
            pd.Series(
                evaluationResult.holdout[k].labels,
                index=evaluationResult.holdout[k].ids,
                name="labels",
            ).to_csv(f"{runPath}/holdoutLabels/{k+1}.csv")
     
    if evaluationResult.average_global_feature_explanations is not None:
        evaluationResult.average_global_feature_explanations.to_csv(
            f"{runPath}/averageGlobalExplanations.csv"
        )

    if config["model"]["calculateShapelyExplanations"]:
        evaluationResult.average_test_local_explanations.to_csv(
            f"{runPath}/averageLocalExplanations.csv"
        )
        if evaluationResult.holdout:
            evaluationResult.average_holdout_local_explanations.to_csv(
                f"{runPath}/averageHoldoutLocalExplanations.csv"
            )

    with open(
        f"{runPath}/trainCount_{np.mean([len(foldResult.labels) for foldResult in evaluationResult.train])}",
        "w",
    ) as file:
        pass
    with open(
        f"{runPath}/testCount_{np.mean([len(foldResult.labels) for foldResult in evaluationResult.test])}",
        "w",
    ) as file:
        pass
    with open(f"{runPath}/meanTestAUC_{evaluationResult.average_test_auc}", "w") as file:
        pass
    with open(
        f"{runPath}/testMeanAccuracy_{evaluationResult.average_test_accuracy}",
        "w",
    ) as file:
        pass
    if evaluationResult.holdout:
        with open(
            f"{runPath}/meanHoldoutAUC_{evaluationResult.average_holdout_auc}", "w"
        ) as file:
            pass
        with open(
            f"{runPath}/holdoutCount_{np.mean([len(foldResult.labels) for foldResult in evaluationResult.holdout])}",
            "w",
        ) as file:
            pass
        with open(
            f"{runPath}/holdoutMeanAccuracy_{evaluationResult.average_holdout_accuracy}",
            "w",
        ) as file:
            pass
    if evaluationResult.excess:
        with open(
            f"{runPath}/{evaluationResult.excess[0].set}Count_{np.mean([len(foldResult.labels) for foldResult in evaluationResult.excess])}",
            "w",
        ) as file:
            pass
        with open(
            f"{runPath}/{evaluationResult.excess[0].set}MeanAccuracy_{evaluationResult.average_excess_accuracy}",
            "w",
        ) as file:
            pass

    gc.collect()


def evaluate(
    trainData: SampleData,
    testData: SampleData,
    excessData: SampleData,
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
    excessProbabilities = get_probabilities(model, excessData.vectors)
    holdoutProbabilities = get_probabilities(model, holdoutData.vectors)

    (
        modelValues,
        shapValues,
        excessShapValues,
        holdoutShapValues,
        shapExplainer,
        shapMasker,
    ) = getFeatureImportances(model, testData, holdoutData, excessData, variantIndex, config)

    return (
        FoldResult(
            "test",
            testData.ids,
            testData.labels,
            testData.vectors,
            testData.geneCount,
            probabilities,
            modelValues,
            shapValues,
            shapExplainer,
            shapMasker,
            fittedOptimizer,
        ),
        FoldResult(
            "holdout",
            holdoutData.ids,
            holdoutData.labels,
            holdoutData.vectors,
            holdoutData.geneCount,
            holdoutProbabilities,
            modelValues,
            holdoutShapValues,
        ),
        FoldResult(
            excessData.set,
            excessData.ids,
            excessData.labels,
            excessData.vectors,
            excessData.geneCount,
            excessProbabilities,
            modelValues,
            excessShapValues
        )
    )

def classify(
    runNumber,
    model,
    hyperParameterSpace,
    genotypeData,
    freqReferenceGenotypeData,
    clinicalData,
    innerCvIterator,
    outerCvIterator,
    sample_frequencies,
    config,
    track=True,
):
    caseGenotypes = genotypeData.case.genotype
    controlGenotypes = genotypeData.control.genotype
    holdoutCaseGenotypes = genotypeData.holdout_case.genotype
    holdoutControlGenotypes = genotypeData.holdout_control.genotype
    
    sample_frequencies, embedding = prepareDatasets(
        caseGenotypes,
        controlGenotypes,
        holdoutCaseGenotypes,
        holdoutControlGenotypes,
        sample_frequencies,
        verbose=(True if runNumber == 0 else False),
        config=config,
        freqReferenceGenotypeData=freqReferenceGenotypeData
    )
    if embedding is None: raise Exception("No samples found in embedding.")

    clinicalIDs = list()

    for id in np.hstack([embedding["sampleIndex"], embedding["holdoutSampleIndex"]]):
        if config["vcfLike"]["compoundSampleIdDelimiter"] in id:
            clinicalIDs.append(
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
        ids=embedding["holdoutSampleIndex"],
        labels=embedding["holdoutLabels"],
        vectors=embedding["holdoutSamples"],
        geneCount=len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique()) if config['vcfLike']['aggregateGenesBy'] == None else len(embedding["variantIndex"])
    )

    evaluate_args = [
        (
            SampleData(
                set="train",
                ids=embedding["sampleIndex"][trainIndices],
                labels=embedding["labels"][trainIndices],
                vectors=embedding["samples"][trainIndices],
                geneCount=len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique()) if config['vcfLike']['aggregateGenesBy'] == None else len(embedding["variantIndex"])
            ),
            SampleData(
                set="test",
                ids=embedding["sampleIndex"][testIndices],
                labels=embedding["labels"][testIndices],
                vectors=embedding["samples"][testIndices],
                geneCount=len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique()) if config['vcfLike']['aggregateGenesBy'] == None else len(embedding["variantIndex"])
            ),
            SampleData(
                set=embedding["excessMajorSetName"],
                ids=embedding["excessMajorIndex"],
                labels=embedding["excessMajorLabels"],
                vectors=embedding["excessMajorSamples"],
                geneCount=len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique()) if config['vcfLike']['aggregateGenesBy'] == None else len(embedding["variantIndex"])
                ),
            holdoutData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            list(embedding["variantIndex"]),
            config,
        )
        for trainIndices, testIndices in crossValIndices
    ]

    modelResults = EvaluationResult()
    # run sequentially since models are not concurrency-safe
    for args in evaluate_args:
        # inner cross-validation is hyperparameter optimization
        testResult, holdoutResult, excessResult = evaluate(*args)
        testResult.append_allele_frequencies(genotypeData)
        holdoutResult.append_allele_frequencies(genotypeData)
        modelResults.train.append(args[0])
        modelResults.test.append(testResult)
        modelResults.holdout.append(holdoutResult)
        modelResults.excess.append(excessResult)
        gc.collect()

    if track:
        trackResults(runID, modelResults, config)

        # plot AUC & hyperparameter convergence
        plotSubtitle = f"""
            {config["tracking"]["name"]}, {embedding["samples"].shape[1]} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else ("variants (" + str(len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique())) +' genes)')}
            Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
            
            {np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(modelResults.average_test_case_accuracy)} accuracy, {len(embedding['labels']) - np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(modelResults.average_test_control_accuracy)} accuracy
            {int(np.around(np.mean([len(foldResult.labels) for foldResult in modelResults.train])))}±1 train, {int(np.around(np.mean([len(foldResult.labels) for foldResult in modelResults.test])))}±1 test samples per x-val fold"""
        if modelResults.holdout:
            holdoutPlotSubtitle = f"""
                {config["tracking"]["name"]}, {embedding["samples"].shape[1]} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else ("variants (" + str(len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique())) +' genes)')}
                Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
                
                Ethnically variable holdout
                {np.count_nonzero(embedding['holdoutLabels'])} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(modelResults.average_holdout_case_accuracy)} accuracy, {len(embedding['holdoutLabels']) - np.count_nonzero(embedding['holdoutLabels'])} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(modelResults.average_holdout_control_accuracy)} accuracy
                {int(np.around(np.mean([len(foldResult.labels) for foldResult in modelResults.train])))}±1 train, {int(np.around(np.mean([len(foldResult.labels) for foldResult in modelResults.test])))}±1 test samples per x-val fold"""

            trackBootstrapVisualizations(
                runID,
                holdoutPlotSubtitle,
                model.__class__.__name__,
                modelResults,
                holdout=True,
                config=config,
            )
            gc.collect()
        # if modelResults.excess:
        #     excessPlotSubtitle = f"""
        #         {config["tracking"]["name"]}, {embedding["samples"].shape[1]} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else ("variants (" + str(len(embedding["variantIndex"].get_level_values(config['vcfLike']['indexColumn'][config['vcfLike']['geneMultiIndexLevel']]).unique())) +' genes)')}
        #         Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

        #         {len(embedding['excessMajorLabels'])} {modelResults.excess[0].set} samples @ {'{:.1%}'.format(modelResults.average_excess_accuracy)} accuracy"""
        #     trackBootstrapVisualizations(
        #         runID,
        #         excessPlotSubtitle,
        #         model.__class__.__name__,
        #         modelResults,
        #         excess=True,
        #         config=config,
        #     )

        trackBootstrapVisualizations(
            runID,
            plotSubtitle,
            model.__class__.__name__,
            modelResults,
            config=config,
        )

    return sample_frequencies, modelResults


def bootstrap(
    genotypeData,
    frequencyReferenceGenotypeData,
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
    sampleFrequencies = {id: 0 for id in genotypeData.case.genotype.columns.tolist() + genotypeData.control.genotype.columns.tolist() + genotypeData.holdout_case.genotype.columns.tolist() + genotypeData.holdout_control.genotype.columns.tolist()}
    
    for attr in ["case", "control", "holdout_case", "holdout_control"]:
        idsToDrop = config['sampling']['sequesteredIDs'] if isinstance(config['sampling']['sequesteredIDs'],list) else config['sampling']['sequesteredIDs'][model.__class__.__name__]
        dataset = getattr(genotypeData, attr)
        allIDs = dataset.ids
        try:
            subjectIDs = clinicalData.loc[list(allIDs), config['clinicalTable']['subjectIdColumn']]
        except:
            subjectIDs = pd.Series(index=allIDs)
        preSequesterCount = len(allIDs)
        dataset.ids = [id for id in allIDs 
                       if not any([id in idToDrop or idToDrop in id or (str(subjectIDs[id]) in idToDrop or idToDrop in str(subjectIDs[id])) for idToDrop in idsToDrop])]
        dataset.genotype = dataset.genotype.drop([id for id in idsToDrop if id in dataset.genotype.columns], axis=1)
        print(f"Sequestered {preSequesterCount - len(dataset.ids)} {attr} samples")
        setattr(genotypeData, attr, dataset)

    # parallelize with workflow engine in cluster environment
    for runNumber in range(
        config["sampling"]["lastIteration"],
        config["sampling"]["bootstrapIterations"],
    ):
        try:
            # update results for every bootstrap iteration
            sampleFrequencies, iterationResults = classify(
                    runNumber,
                    model,
                    hyperParameterSpace,
                    genotypeData,
                    frequencyReferenceGenotypeData,
                    clinicalData,
                    innerCvIterator,
                    outerCvIterator,
                    sampleFrequencies,
                    config,
                    track,
                )
            bootstrap.iteration_results.append(
                iterationResults
            )
        except Exception as e:
            print("Error in bootstrap iteration " + str(runNumber) + f"{e}")
            raise
        gc.collect()

    return bootstrap
