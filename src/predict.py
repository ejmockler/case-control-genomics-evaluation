from io import StringIO
from fastnumbers import check_real
from matplotlib.gridspec import GridSpec

from neptune.types import File
from prefect import task
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.preprocessing import MinMaxScaler
from skopt.plots import plot_convergence
from skopt import BayesSearchCV
from tqdm import tqdm

from config import config

import pandas as pd
import numpy as np
import neptune
import shap
import matplotlib.pyplot as plt


@task()
def getFeatureImportances(model, data, featureLabels):
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
        modelCoefficientDF.index.name = "features"

    if config["model"]["calculateShapelyExplanations"]:
        # Cluster correlated and hierarchical features using masker
        masker = shap.maskers.Partition(data, clustering="correlation")
        shapExplainer = shap.explainers.Permutation(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            masker,
            feature_names=["_".join(label) for label in featureLabels],
        )
        shapValues = shapExplainer(data)
    else:
        shapExplainer = None
        shapValues = None
        masker = None
    return modelCoefficientDF, shapValues, shapExplainer, masker


@task()
async def plotCalibration(title, labelsPredictionsByInstance):
    # code from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig, ax_calibration_curve = plt.subplots(figsize=(10, 10))
    colors = plt.cm.get_cmap("Dark2")

    calibration_displays = {}
    for i, (name, (labels, predictions)) in enumerate(
        labelsPredictionsByInstance.items()
    ):
        display = CalibrationDisplay.from_predictions(
            [
                config["clinicalTable"]["caseAlias"] if label == 1 else label
                for label in labels
            ],
            predictions,
            pos_label=config["clinicalTable"]["caseAlias"],
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(title)

    # Add histogram
    # grid_positions = [(i+2,j) for i in range(len(predictionsByModelName.keys())//2) for j in range(2)]
    # for i, modelName in enumerate(predictionsByModelName.keys()):
    #     row, col = grid_positions[i]
    #     ax = fig.add_subplot(gs[row, col])
    #     ax.hist(
    #         calibration_displays[modelName].y_prob,
    #         range=(0, 1),
    #         bins=10,
    #         label=modelName,
    #         color=colors(i),
    #     )
    #     ax.set(title=modelName, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    return fig


@task()
async def plotAUC(title, labelsPredictionsByInstance):
    # trace AUC for each set of predictions
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    for name, (labels, predictions) in labelsPredictionsByInstance.items():
        # plot ROC curve for this fold
        viz = RocCurveDisplay.from_predictions(
            [
                config["clinicalTable"]["caseAlias"] if label == 1 else label
                for label in labels
            ],
            predictions,
            name=name,
            pos_label=config["clinicalTable"]["caseAlias"],
            alpha=0.6,
            lw=2,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # summarize ROCs per fold and plot standard deviation
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=4,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    ax.set(title=title)
    return fig


@task()
async def plotConfusionMatrix():
    pass


@task()
async def plotSampleAccuracy():
    pass


@task()
async def plotOptimizer(title, resultsByInstance):
    # code from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2)
    colors = plt.cm.get_cmap("Dark2")
    ax_convergence = fig.add_subplot(gs[:2, :2])
    plot_convergence(
        *[(modelName, result) for modelName, result in resultsByInstance.items()],
        ax=ax_convergence,
        color=colors,
    )
    ax_convergence.set(title=title)
    plt.tight_layout()
    return fig


@task()
def prepareDatasets(caseGenotypes, controlGenotypes, verbose=True):
    caseIDs = caseGenotypes.columns
    controlIDs = controlGenotypes.columns
    # store number of cases & controls
    caseControlCounts = [len(caseIDs), len(controlIDs)]
    # determine which has more samples
    labeledIDs = [caseIDs, controlIDs]
    majorIDs = labeledIDs[np.argmax(caseControlCounts)]
    minorIDs = labeledIDs[np.argmin(caseControlCounts)]
    # downsample larger group to match smaller group
    majorIndex = np.random.choice(
        np.arange(len(majorIDs)), min(caseControlCounts), replace=False
    )

    excessMajorIDs, balancedMajorIDs = [], []
    for index, id in enumerate(majorIDs):
        if index in majorIndex:
            balancedMajorIDs.append(id)
        else:
            excessMajorIDs.append(id)

    allGenotypes = pd.concat([caseGenotypes, controlGenotypes], axis=1)

    genotypeExcessIDs, crossValGenotypeIDs = [], []
    # match IDs between genotype and clinical data; dataframe labels have label suffixes
    unmatchedTrainIDs = balancedMajorIDs + minorIDs
    for label in tqdm(allGenotypes.columns, desc="Matching IDs", unit="ID"):
        for setType in ["excess", "train"]:
            idSet = excessMajorIDs if setType == "excess" else unmatchedTrainIDs
            for i, id in enumerate(idSet):
                if (id in label) or (label in id):
                    if setType == "train":
                        if label not in crossValGenotypeIDs:
                            crossValGenotypeIDs.append(label)
                    elif setType == "excess":
                        if label not in genotypeExcessIDs:
                            genotypeExcessIDs.append(label)
                    idSet = np.delete(idSet, i)
                    break

    if verbose:
        print(f"\n{len(crossValGenotypeIDs)} for training:\n{crossValGenotypeIDs}")
        print(f"\n{len(genotypeExcessIDs)} are excess:\n{genotypeExcessIDs}")
        print(f"\nVariant count: {len(allGenotypes.index)}")

    samples = allGenotypes.loc[:, crossValGenotypeIDs].dropna(
        how="any"
    )  # drop variants with missing values
    excessMajorSamples = allGenotypes.loc[:, genotypeExcessIDs]

    variantIndex = list(samples.index)
    pass
    scaler = MinMaxScaler()
    embedding = {
        "sampleIndex": crossValGenotypeIDs,
        "labels": np.array([1 if id in caseIDs else 0 for id in crossValGenotypeIDs]),
        "samples": scaler.fit_transform(
            samples
        ).transpose(),  # samples are now rows (samples, variants)
        "excessMajorIndex": genotypeExcessIDs,
        "excessMajorLabels": [1 if id in caseIDs else 0 for id in genotypeExcessIDs],
        "excessMajorSamples": scaler.fit_transform(excessMajorSamples).transpose(),
        "variantIndex": variantIndex,
    }
    return embedding


@task()
def optimizeHyperparameters(
    samples, labels, model, parameterSpace, cvIterator, metricFunction, n_jobs=1
):
    # hyperparameter search (inner cross-validation)
    optimizer = BayesSearchCV(
        model,
        parameterSpace,
        cv=cvIterator,
        n_jobs=n_jobs,
        n_points=4,
        return_train_score=True,
        n_iter=100,
        scoring=metricFunction,
    )
    # train / optimize parameters
    optimizer.fit(samples, labels)
    return optimizer


def serializeDataFrame(dataframe):
    stream = StringIO()
    dataframe.to_csv(stream)
    return File.from_stream(stream, extension="csv")


@task()
async def beginTracking(model, runNumber, embedding, clinicalData, deserializedIDs):
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

    embeddingDF = pd.DataFrame(
        data=embedding["samples"],
        columns=embedding["variantIndex"],
        index=embedding["sampleIndex"],
    )
    embeddingDF.index.name = "id"
    runTracker["embedding"].upload(serializeDataFrame(embeddingDF))
    runTracker["clinicalData"].upload(
        serializeDataFrame(clinicalData.loc[clinicalData.index.isin(deserializedIDs)])
    )

    runTracker["nVariants"] = len(embedding["variantIndex"])
    runID = runTracker["sys/id"].fetch()
    runTracker.stop()
    return runID


@task()
async def trackResults(runID, current):
    runTracker = neptune.init_run(
        project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
        with_id=runID,
        api_token=config["tracking"]["token"],
    )
    if config["model"]["hyperparameterOptimization"]:
        runTracker["modelParams"] = {
            k + 1: current["fittedOptimizers"][k].best_params_
            for k in range(config["sampling"]["crossValIterations"])
        }

    runTracker["sampleResults"].upload(
        serializeDataFrame(
            pd.DataFrame.from_dict(
                {
                    "probability": [
                        probability[1]
                        for foldResults in current["probabilities"]
                        for probability in foldResults
                    ],
                    "id": [
                        id for foldResults in current["testIDs"] for id in foldResults
                    ],
                },
                dtype=object,
            ).set_index("id")
        )
    )

    if config["model"]["calculateShapelyExplanations"]:
        runTracker["shapExplanationsPerFold"].upload(
            File.as_pickle(current["localExplanations"])
        )
        runTracker["shapExplainersPerFold"].upload(
            File.as_pickle(current["shapExplainers"])
        )
        runTracker["shapMaskersPerFold"].upload(File.as_pickle(current["shapMaskers"]))
        runTracker["featureImportance/shapelyExplanations/average"].upload(
            serializeDataFrame(current["averageShapelyValues"])
        )

    if current["globalExplanations"][0] is not None:
        runTracker[f"featureImportance/modelCoefficients/average"].upload(
            serializeDataFrame(current["averageGlobalExplanations"])
        )

    for k in range(config["sampling"]["crossValIterations"]):
        testLabelsSeries = pd.Series(current["testLabels"][k], name="testLabel")
        trainLabelsSeries = pd.Series(current["trainLabels"][k], name="trainLabel")
        testLabelsSeries.index = current["testIDs"][k]
        testLabelsSeries.index.name = "id"
        trainLabelsSeries.index = current["trainIDs"][k]
        trainLabelsSeries.index.name = "id"
        runTracker[f"trainIDs/{k+1}"].upload(
            serializeDataFrame(pd.Series(current["trainIDs"][k]))
        )
        runTracker[f"testIDs/{k+1}"].upload(
            serializeDataFrame(pd.Series(current["testIDs"][k]))
        )
        runTracker[f"testLabels/{k+1}"].upload(
            serializeDataFrame(pd.Series(testLabelsSeries))
        )
        runTracker[f"trainLabels/{k+1}"].upload(
            serializeDataFrame(pd.Series(trainLabelsSeries))
        )
        if current["globalExplanations"][k] is not None:
            runTracker[f"featureImportance/modelCoefficients/{k+1}"].upload(
                serializeDataFrame(current["globalExplanations"][k])
            )
        if config["model"]["calculateShapelyExplanations"]:
            runTracker[f"featureImportance/shapelyExplanations/{k+1}"].upload(
                serializeDataFrame(
                    pd.DataFrame.from_dict(
                        {
                            "feature_name": [
                                name
                                for name in current["localExplanations"][
                                    0
                                ].feature_names
                            ],
                            "value": [
                                np.mean(
                                    current["localExplanations"][k].values[featureIndex]
                                )
                                for featureIndex in range(
                                    len(current["localExplanations"][0].feature_names)
                                )
                            ],
                        },
                        dtype=object,
                    ).set_index("feature_name")
                )
            )

    runTracker["meanAUC"] = np.mean(current["testAUC"])
    # average sample count across folds
    runTracker["nTrain"] = np.mean([len(idList) for idList in current["trainIDs"]])
    runTracker["nTest"] = np.mean([len(idList) for idList in current["testIDs"]])
    runTracker.stop()
