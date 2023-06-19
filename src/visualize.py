import os
import traceback
from types import SimpleNamespace
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import neptune
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, auc
from skopt.plots import plot_convergence
import shap

from config import config


def plotCalibration(title, labelsPredictionsByInstance):
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


def plotAUC(title, labelsPredictionsByInstance):
    # trace AUC for each set of predictions
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    for name, (labels, predictions) in labelsPredictionsByInstance.items():
        # plot ROC curve for this fold
        viz = RocCurveDisplay.from_predictions(
            [
                config["clinicalTable"]["caseAlias"]
                if label == 1
                else config["clinicalTable"]["controlAlias"]
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
    plt.tight_layout()
    return fig


def plotConfusionMatrix():
    pass


def plotSampleAccuracy():
    pass


def plotOptimizer(title, resultsByInstance):
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


def trackVisualizations(runID, plotSubtitle, modelName, current, holdout=False):
    aucName = "aucPlot" if not holdout else "aucPlotHoldout"
    probabilities = (
        current["probabilities"] if not holdout else current["holdoutProbabilities"]
    )
    labels = current["testLabels"] if not holdout else current["holdoutLabels"]
    aucPlot = plotAUC(
        f"""
                    Receiver Operating Characteristic (ROC) Curve
                    {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                    {plotSubtitle}
                    """,
        {
            f"Fold {k+1}": (
                labels[k],
                np.array(probabilities[k])[:, 1],
            )
            if len(probabilities[k][0].shape) >= 1
            else (labels[k], probabilities[k])
            for k in range(config["sampling"]["crossValIterations"])
        },
    )
    if config["model"]["hyperparameterOptimization"]:
        optimizerPlotName = (
            "convergencePlot" if not holdout else "convergencePlotHoldout"
        )
        optimizerPlot = plotOptimizer(
            f"""
                    Hyperparameter convergence, mean squared error
                    {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                    {plotSubtitle}
                    """,
            {
                f"Fold {k+1}": [
                    result
                    for result in current["fittedOptimizer"][k].optimizer_results_
                ]
                for k in range(config["sampling"]["crossValIterations"])
            },
        )

    if (
        config["model"]["calculateShapelyExplanations"]
        and config["tracking"]["plotAllSampleImportances"]
    ):
        waterfallList = []
        holdoutWaterfallList = []
        for j in range(config["sampling"]["crossValIterations"]):
            waterfallList.append([])
            holdoutWaterfallList.append([])
            for k in range(len(current["testIDs"][j]) + len(current["holdoutIDs"][j])):
                currentIndex = (
                    k
                    if k < len(current["testIDs"][j])
                    else k - len(current["testIDs"][j])
                )
                currentLabel = (
                    current["testLabels"][j][k]
                    if k < len(current["testIDs"][j])
                    else current["holdoutLabels"][j][currentIndex]
                )
                sampleID = (
                    current["testIDs"][j][k]
                    if k < len(current["testIDs"][j])
                    else current["holdoutIDs"][j][currentIndex]
                )
                localExplanations = (
                    current["localExplanations"][j]
                    if k < len(current["testIDs"][j])
                    else current["holdoutLocalExplanations"][j]
                )
                waterfallPlot = plt.figure()
                plt.title(
                    f"""
                            {sampleID}
                            Shapely explanations from {modelName}
                            Fold {j+1}
                            {plotSubtitle}
                            """
                )
                # patch parameter bug: https://github.com/slundberg/shap/issues/2362
                to_pass = SimpleNamespace(
                    **{
                        "values": localExplanations[currentIndex].values[:, 1]
                        if len(localExplanations[currentIndex].values.shape) > 1
                        else localExplanations[currentIndex].values,
                        "data": localExplanations[currentIndex].data,
                        "display_data": None,
                        "feature_names": localExplanations.feature_names,
                        "base_values": localExplanations[currentIndex].base_values[
                            currentLabel
                        ]
                        if len(localExplanations[currentIndex].base_values.shape) == 1
                        else localExplanations[currentIndex].base_values,
                    }
                )
                shap.plots.waterfall(to_pass, show=False)
                plt.tight_layout()
                waterfallList[j].append(waterfallPlot) if k < len(
                    current["testIDs"][j]
                ) else holdoutWaterfallList[j].append(waterfallPlot)
                plt.close(waterfallPlot)
    plt.close("all")

    if config["tracking"]["remote"]:
        runTracker = neptune.init_run(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            with_id=runID,
            api_token=config["tracking"]["token"],
            capture_stdout=False,
        )
        runTracker[f"plots/{aucName}"] = aucPlot
        if config["model"]["hyperparameterOptimization"]:
            runTracker[f"plots/{optimizerPlotName}"] = optimizerPlot

        # plot shapely feature importance
        if (
            config["model"]["calculateShapelyExplanations"]
            and config["tracking"]["plotAllSampleImportances"]
        ):
            for j in range(config["sampling"]["crossValIterations"]):
                for k in range(
                    len(current["testIDs"][j]) + len(current["holdoutIDs"][j])
                ):
                    if k < len(current["testIDs"][j]):
                        logPath = f"plots/samples/{j+1}/{sampleID}"
                        plotList = waterfallList
                        currentIndex = k
                    else:
                        logPath = f"plots/samples/holdout/{j+1}/{sampleID}"
                        plotList = holdoutWaterfallList
                        currentIndex = k - len(current["testIDs"][j])
                    try:
                        runTracker[logPath] = plotList[j][currentIndex]
                    except Exception:
                        runTracker[
                            logPath
                        ] = f"""failed to plot: {traceback.format_exc()}"""
        runTracker.stop()

    else:  # store plots locally
        runPath = runID
        aucPlot.savefig(f"{runPath}/{aucName}.svg")
        aucPlot.savefig(f"{runPath}/{aucName}.png")
        if config["model"]["hyperparameterOptimization"]:
            optimizerPlot.savefig(f"{runPath}/{optimizerPlotName}.svg")
            optimizerPlot.savefig(f"{runPath}/{optimizerPlotName}.png")
        if (
            config["model"]["calculateShapelyExplanations"]
            and config["tracking"]["plotAllSampleImportances"]
        ):
            for j in range(config["sampling"]["crossValIterations"]):
                for k in range(
                    len(current["testIDs"][j]) + len(current["holdoutIDs"][j])
                ):
                    if k < len(current["testIDs"][j]):
                        samplePlotPath = f"{runPath}/featureImportance/shapelyExplanations/samples/{j+1}"
                        os.makedirs(samplePlotPath, exist_ok=True)
                        waterfallList[j][k].savefig(
                            f"{samplePlotPath}/{current['testIDs'][j][k]}.svg"
                        )
                    else:
                        samplePlotPath = f"{runPath}/featureImportance/shapelyExplanations/samples/holdout/{j+1}"
                        os.makedirs(samplePlotPath, exist_ok=True)
                        holdoutWaterfallList[j][k - len(current["testIDs"][j])].savefig(
                            f"{samplePlotPath}/{current['holdoutIDs'][j][k - len(current['testIDs'][j])]}.svg"
                        )

    plt.close("all")
