import os
import sys
import traceback
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import plotly.express as px
import neptune
import numpy as np
from prefect import task
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
)
from skopt.plots import plot_convergence
import gc
from config import config
from tasks.data import BootstrapResult, ClassificationResults, EvaluationResult

matplotlib.use("agg")


def plotCalibration(title, labelsPredictionsByInstance, config):
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
    title = "\n".join(line.strip() for line in title.split("\n"))
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


def plotAUC(
    title, labelsPredictionsByInstance=None, tprFprAucByInstance=None, config=config
):
    # trace AUC for each set of predictions
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    multipleInstance = False

    fig, ax = plt.subplots(figsize=(10, 10))
    if labelsPredictionsByInstance is not None:
        if len(labelsPredictionsByInstance) > 1:
            multipleInstance = True
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
    elif tprFprAucByInstance is not None:
        if len(tprFprAucByInstance) > 1:
            multipleInstance = True
        for name, (tprList, fprList, aucScore) in tprFprAucByInstance.items():
            viz = RocCurveDisplay(
                tpr=tprList,
                fpr=fprList,
                roc_auc=aucScore,
                estimator_name=name,
            )
            viz.plot(alpha=0.6, lw=2, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    if multipleInstance:
        # summarize ROCs per fold and plot standard deviation
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

    title = "\n".join(line.strip() for line in title.split("\n"))
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    ax.set(title=title)
    plt.tight_layout()
    return fig


def plotConfusionMatrix(title, labelsPredictionsByInstance, config):
    all_labels = []
    all_predictions = []
    matrix_figures = []

    for name, (labels, predictions) in labelsPredictionsByInstance.items():
        # Ensure any probabilities become predictions
        predictions = np.around(predictions).astype(int)
        # Compute confusion matrix for this fold
        matrix = confusion_matrix(labels, predictions)

        # Create ConfusionMatrixDisplay for this fold
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                config["clinicalTable"]["controlAlias"],
                config["clinicalTable"]["caseAlias"],
            ],
        )

        fig, ax = plt.subplots()
        disp.plot(
            include_values=True, cmap="viridis", ax=ax, xticks_rotation="horizontal"
        )

        centeredTitle = "\n".join(
            "    " + line.strip()
            for line in f"""
            {title}
            {name}
            """.split(
                "\n"
            )
        )
        ax.set_title(centeredTitle, fontsize=8)

        plt.tight_layout()
        matrix_figures.append(fig)
        plt.close(fig)

        # Collect all labels and predictions for later use
        all_labels.extend(labels)
        all_predictions.extend(predictions)

    avg_matrix = confusion_matrix(all_labels, all_predictions, normalize="true")

    # Create ConfusionMatrixDisplay for the average confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=avg_matrix,
        display_labels=[
            config["clinicalTable"]["controlAlias"],
            config["clinicalTable"]["caseAlias"],
        ],
    )

    avgFig, ax = plt.subplots()
    cm_display = disp.plot(
        include_values=True, cmap="viridis", ax=ax, xticks_rotation="horizontal"
    )

    # List normalized proportions as percentages
    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            percentage = avg_matrix[i, j] * 100
            cm_display.text_[i, j].set_text(f"{percentage:.1f}%")  # 1 decimal place
            cm_display.text_[i, j].set_fontsize(7)

    centeredTitle = "\n".join(
        "    " + line.strip()
        for line in f"""
        {title}
        Average across folds
        """.split(
            "\n"
        )
    )

    ax.set_title(centeredTitle, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    colorbar = cm_display.im_.colorbar
    colorbar.ax.tick_params(labelsize=7)

    # Convert colorbar ticks to percentages
    tick_vals = colorbar.get_ticks()
    tick_labels = ["{:.0f}%".format(val * 100) for val in tick_vals]
    colorbar.set_ticklabels(tick_labels)
    colorbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.close("all")

    return matrix_figures, avgFig


def plotSampleAccuracy():
    pass


def plotOptimizer(title, resultsByInstance):
    # code from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2)
    ax_convergence = fig.add_subplot(gs[:2, :2])
    plot_convergence(
        *[(modelName, result) for modelName, result in resultsByInstance.items()],
        ax=ax_convergence,
    )
    title = "\n".join(line.strip() for line in title.split("\n"))
    ax_convergence.set(title=title)
    plt.tight_layout()
    return fig


def plotSample(
    j, k, runID, modelName, plotSubtitle, modelResults, holdout=False, config=config
):
    import matplotlib.pyplot as plt
    import shap

    currentLabel = (
        modelResults.test[j].labels[k]
        if not holdout
        else modelResults.holdout[j][holdout].labels[k]
    )
    sampleID = (
        modelResults.test[j].ids[k] if not holdout else modelResults.holdout[j][holdout].ids[k]
    )
    localExplanations = (
        modelResults.test[j].shap_explanation
        if not holdout
        else modelResults.holdout[j][holdout].shap_explanation
    )
    waterfallPlot = plt.figure()
    title = "\n".join(
        line.strip()
        for line in f"""
            {sampleID}
            Shapely explanations from {modelName}
            Fold {j+1}
            {plotSubtitle}
            """.split(
            "\n"
        )
    )

    plt.title(title)
    shap.plots.waterfall(localExplanations[k, :, 1], show=False)
    plt.tight_layout()
    plt.close(waterfallPlot)

    if config["tracking"]["remote"]:
        if config["tracking"]["remote"]:
            runTracker = neptune.init_run(
                project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
                with_id=runID,
                api_token=config["tracking"]["token"],
                capture_stdout=False,
            )
        if not holdout:
            logPath = f"plots/samples/{j+1}/{sampleID}"
        else:
            logPath = f"plots/samples/holdout/{holdout}/{j+1}/{sampleID}"
        try:
            runTracker[logPath] = waterfallPlot
        except Exception:
            runTracker[logPath] = f"""failed to plot: {traceback.format_exc()}"""
    else:
        runPath = runID
        if not holdout:
            samplePlotPath = f"{runPath}/plots/samples/{j+1}"
            os.makedirs(samplePlotPath, exist_ok=True)
            waterfallPlot.savefig(
                f"{samplePlotPath}/{sampleID}.svg",
                bbox_inches="tight",
            )
        else:
            samplePlotPath = f"{runPath}/plots/samples/holdout/{holdout}/{j+1}"
            os.makedirs(samplePlotPath, exist_ok=True)
            waterfallPlot.savefig(
                f"{samplePlotPath}/{sampleID}.svg",
                bbox_inches="tight",
            )


def trackBootstrapVisualizations(
    runID: str,
    plotSubtitle: str,
    modelName: str,
    modelResults: EvaluationResult,
    holdout="",
    excess=False,
    auc=False,
    config=config,
):
    aucName = f"{holdout}__aucPlot" if holdout else "aucPlot"
    probabilities = [
        fold[holdout].probabilities if holdout else fold.probabilities
        for fold in (modelResults.holdout if holdout else modelResults.excess if excess else modelResults.test)
    ]

    labels = [
        fold[holdout].labels if holdout else fold.labels
        for fold in (modelResults.holdout if holdout else modelResults.excess if excess else modelResults.test)
    ]
    ids = [
        fold[holdout].ids if holdout else fold.ids
        for fold in (modelResults.holdout if holdout else modelResults.excess if excess else modelResults.test)
    ]

    labelsProbabilitiesByFold = {
        f"Fold {k+1}": (
            labels[k],
            np.array(probabilities[k])[:, 1],
        )
        for k in range(config["sampling"]["crossValIterations"])
    }
    labelsPredictionsByFold = {
        f"Fold {k+1}": (labels[k], np.argmax(probabilities[k], axis=1))
        for k in range(config["sampling"]["crossValIterations"])
    }

    # only plot AUC if both labels exist
    if not excess or (holdout not in modelResults.average_holdout_case_accuracy and holdout not in modelResults.average_holdout_control_accuracy):
        aucPlot = plotAUC(
            f"""
                Receiver Operating Characteristic (ROC) Curve
                {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                {plotSubtitle}
                """,
            labelsProbabilitiesByFold,
            config=config,
        )
        calibrationName = "calibrationPlotHoldout" if holdout else "calibrationPlotExcess" if excess else "calibrationPlot"
        calibrationPlot = plotCalibration(
            f"""
                Calibration Curve
                {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                {plotSubtitle}
                """,
            labelsProbabilitiesByFold,
            config=config,
        )
    confusionMatrixName = f"{holdout}__confusionMatrix" if holdout else "confusionMatrixExcess" if excess else "confusionMatrix"
    confusionMatrixList, avgConfusionMatrix = plotConfusionMatrix(
        f"""
            Confusion Matrix
            {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
            {plotSubtitle}
            """,
        labelsPredictionsByFold,
        config=config,
    )
    if config["model"]["hyperparameterOptimization"] and not holdout:
        convergencePlotName = "convergencePlot"
        convergencePlot = plotOptimizer(
            f"""
                Hyperparameter convergence, mean squared error
                {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                {plotSubtitle}
                """,
            {
                f"Fold {k+1}": [
                    result
                    for result in modelResults.test[
                        k
                    ].optimizer_results
                ]
                for k in range(config["sampling"]["crossValIterations"])
            },
        )

    if (
        config["model"]["calculateShapelyExplanations"]
        and config["tracking"]["plotAllSampleImportances"]
    ):
        args = []
        for j in range(config["sampling"]["crossValIterations"]):
            for k in range(len(ids[j])):
                args.append(
                    (
                        j,
                        k,
                        runID,
                        modelName,
                        plotSubtitle,
                        modelResults,
                        holdout,
                        config,
                    )
                )

        for arg in args:
            plotSample(*arg)

        # with multiprocess.Pool(multiprocess.cpu_count()) as pool:
        #     # Use map to apply the function to each argument set in the args list
        #     pool.starmap(plotSample, args)

        gc.collect()

    if config["tracking"]["remote"]:
        runTracker = neptune.init_run(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            with_id=runID,
            api_token=config["tracking"]["token"],
            capture_stdout=False,
        )
        if not excess: runTracker[f"plots/{aucName}"] = aucPlot
        for i, confusionMatrix in enumerate(confusionMatrixList):
            runTracker[f"{confusionMatrixName}/{i+1}"].upload(confusionMatrix)
        runTracker[
            f"average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}"
        ].upload(avgConfusionMatrix)
        runTracker[f"plots/{calibrationName}"] = calibrationPlot
        if config["model"]["hyperparameterOptimization"] and not holdout:
            runTracker[f"plots/{convergencePlotName}"] = convergencePlot

        runTracker.stop()

    else:  # store plots locally
        runPath = runID if not holdout else f"{runID}/holdout/{holdout}"
        os.makedirs(f"{runPath}/plots", exist_ok=True)
        if not excess:
            aucPlot.savefig(f"{runPath}/plots/{aucName}.svg", bbox_inches="tight")
            aucPlot.savefig(f"{runPath}/plots/{aucName}.png", bbox_inches="tight")
            calibrationPlot.savefig(
            f"{runPath}/plots/{calibrationName}.svg", bbox_inches="tight"
            )
            calibrationPlot.savefig(
                f"{runPath}/plots/{calibrationName}.png", bbox_inches="tight"
            )
        confusionMatrixPath = f"{runPath}/plots/{confusionMatrixName}"
        os.makedirs(confusionMatrixPath, exist_ok=True)
        for i, confusionMatrix in enumerate(confusionMatrixList):
            confusionMatrix.savefig(
                f"{confusionMatrixPath}/{i+1}.svg", bbox_inches="tight"
            )
        avgConfusionMatrix.savefig(
            f"{runPath}/plots/average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}.svg",
            bbox_inches="tight",
        )
        avgConfusionMatrix.savefig(
            f"{runPath}/plots/average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}.png",
            bbox_inches="tight",
        )
        
        if config["model"]["hyperparameterOptimization"] and not holdout and not excess:
            convergencePlot.savefig(
                f"{runPath}/plots/{convergencePlotName}.svg", bbox_inches="tight"
            )
            convergencePlot.savefig(
                f"{runPath}/plots/{convergencePlotName}.png", bbox_inches="tight"
            )

    plt.close("all")


def trackModelVisualizations(modelResults: BootstrapResult, config=config):
    testResultsDataFrame = modelResults.test_results_dataframe
     
    seenTestCases = (
        testResultsDataFrame["label"] == 1
    ).sum()
    
    seenTestControls = (
        testResultsDataFrame["label"] == 0
    ).sum()

    bootstrapTrainCount = modelResults.iteration_results[0].train[0].vectors.shape[0]
    bootstrapTestCount = modelResults.iteration_results[0].test[0].vectors.shape[0]

    featureCounts = [trainData.vectors.shape[1] for iteration_result in modelResults.iteration_results for trainData in iteration_result.train]
    featureCount =  featureCounts[0] if np.min(featureCounts) == np.max(featureCounts) else f"{np.min(featureCounts)}-{np.max(featureCounts)}"
    
    geneCounts = [trainData.geneCount for iteration_result in modelResults.iteration_results for trainData in iteration_result.train]
    geneCount = geneCounts[0] if np.min(geneCounts) == np.max(geneCounts) else f"{np.min(geneCounts)}-{np.max(geneCounts)}"

    
    testLabelsPredictions = {
        modelResults.model_name: (
            [
                label
                for iterationResult in modelResults.iteration_results
                for label in iterationResult.test_results_dataframe["label"].tolist()
            ],
            [
                prediction
                for iterationResult in modelResults.iteration_results
                for prediction in iterationResult.test_results_dataframe["prediction_most_frequent"].tolist()
            ],
        )
    }
    testLabelsProbabilities = {
        modelResults.model_name: (
            [
                label
                for iterationResult in modelResults.iteration_results
                for label in iterationResult.test_results_dataframe["label"].tolist()
            ],
            [
                probability
                for iterationResult in modelResults.iteration_results
                for probability in iterationResult.test_results_dataframe["probability_mean"].tolist()
            ],
        )
    }

    plotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
    
    {config["tracking"]["name"]}, {featureCount} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else "variants (" + str(geneCount) + " genes)"}
    Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

    {seenTestCases} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(modelResults.average_test_case_accuracy)} accuracy, {seenTestControls} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(modelResults.average_test_control_accuracy)} accuracy
    {bootstrapTrainCount}±1 train, {bootstrapTestCount}±1 test samples per bootstrap iteration
    {'Sequestered ' + str(len(config['sampling']['sequesteredIDs'][modelResults.model_name])) + ' test cases' if modelResults.model_name in config['sampling']['sequesteredIDs'] and len(config['sampling']['sequesteredIDs'][modelResults.model_name]) > 0 else ''}"""

    accuracyHistogram = px.histogram(
        testResultsDataFrame,
        x="accuracy_mean",
        color="label",
        pattern_shape="label",
        hover_data={"index": list(modelResults.test_dict.keys())},
        color_discrete_map={0: "red", 1: "blue"},
        barmode="overlay",
        range_x=[0, 1],
        nbins=100,
        title=f"""Mean test sample accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
    )
    accuracyHistogram.update_layout(title=f"Mean {modelResults.model_name} test set probability<br>{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations")
    probabilityHistogram = px.histogram(
        testResultsDataFrame,
        x="probability_mean",
        color="label",
        pattern_shape="label",
        hover_data={"index": list(modelResults.test_dict.keys())},
        color_discrete_map={0: "red", 1: "blue"},
        barmode="overlay",
        nbins=100,
        title=f"""Mean test sample probability, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
    )
    probabilityHistogram.update_layout(title=f"Mean {modelResults.model_name} test set probability<br>{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations")
    
    aucPlot = plotAUC(
        f"""
            Receiver Operating Characteristic (ROC) Curve
            {plotSubtitle}
            """,
        labelsPredictionsByInstance=testLabelsProbabilities,
        config=config,
    )
    calibrationPlot = plotCalibration(
        f"""
            Calibration Curve
            {plotSubtitle}
            """,
        testLabelsPredictions,
        config=config,
    )
    confusionMatrixInstanceList, averageConfusionMatrix = plotConfusionMatrix(
        f"""
            Confusion Matrix
            {plotSubtitle}
            """,
        testLabelsPredictions,
        config=config,
    )
    if config["model"]["hyperparameterOptimization"]:
        try:
            convergencePlot = plotOptimizer(
                f"""
                    Hyperparameter Convergence Plot
                    {plotSubtitle}
                    """,
                {
                    modelResults.model_name: [
                        result
                        for foldResult in modelResults.iteration_results
                        for testFold in foldResult.test
                        for result in testFold.optimizer_results
                    ]
                },
            )
        except Exception as e:
            print(f"Convergence plot data unavailable! {e}", file=sys.stderr)
            convergencePlot = None

    if modelResults.iteration_results[0].holdout:
        for setName in modelResults.holdout_results_dataframe:
            holdoutResultsDataFrame = modelResults.holdout_results_dataframe[setName]
            seenHoldoutCases = (holdoutResultsDataFrame["label"] == 1).sum()
            
            seenHoldoutControls = (holdoutResultsDataFrame["label"] == 0).sum()
            bootstrapHoldoutCount = seenHoldoutCases + seenHoldoutControls

            holdoutLabelsPredictions = {
                modelResults.model_name: (
                    [
                        label
                        for iterationResult in modelResults.iteration_results
                        for label in iterationResult.holdout_results_dataframe[setName]["label"].tolist()
                    ],
                    [
                        prediction
                        for iterationResult in modelResults.iteration_results
                        for prediction in iterationResult.holdout_results_dataframe[setName]["prediction_most_frequent"].tolist()
                    ],
                )
            }
            
            holdoutLabelsProbabilities = {
                modelResults.model_name: (
                    [
                        label
                        for iterationResult in modelResults.iteration_results
                        for label in iterationResult.holdout_results_dataframe[setName]["label"].tolist()
                    ],
                    [
                        probability
                        for iterationResult in modelResults.iteration_results
                        for probability in iterationResult.holdout_results_dataframe[setName]["probability_mean"].tolist()
                    ],
                )
            }

            holdoutPlotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
            
                {config["tracking"]["name"]}, {featureCount} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else "variants (" + str(geneCount) + " genes)"}
                Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

                {setName} holdout"""
            if setName in modelResults.average_holdout_case_accuracy:
                holdoutPlotSubtitle += f"\n{seenHoldoutCases} {config['clinicalTable']['caseAlias']}s @ {'{:.1%}'.format(modelResults.average_holdout_case_accuracy[setName])} accuracy"
            if setName in modelResults.average_holdout_control_accuracy:
                holdoutPlotSubtitle += f"\n{seenHoldoutControls} {config['clinicalTable']['controlAlias']}s @ {'{:.1%}'.format(modelResults.average_holdout_control_accuracy[setName])} accuracy"
            holdoutPlotSubtitle += f"\n{bootstrapHoldoutCount} holdout samples" 
            
            holdoutAccuracyHistogram = px.histogram(
                holdoutResultsDataFrame,
                x="accuracy_mean",
                color="label",
                pattern_shape="label",
                hover_data={"index": list(modelResults.holdout_dict[setName].keys())},
                color_discrete_map={0: "red", 1: "blue"},
                barmode="overlay",
                range_x=[0, 1],
                nbins=100,
                title=f"""{setName} accuracy, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
            )
            holdoutAccuracyHistogram.update_layout(title={'text': f"{setName} holdout {modelResults.model_name} mean accuracy<br>{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations"})
            holdoutProbabilityHistogram = px.histogram(
                holdoutResultsDataFrame,
                x="probability_mean",
                color="label",
                pattern_shape="label",
                hover_data={"index": list(modelResults.holdout_dict[setName].keys())},
                color_discrete_map={0: "red", 1: "blue"},
                barmode="overlay",
                nbins=100,
                title=f"""{setName} mean probability, {config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations""",
            )
            holdoutProbabilityHistogram.update_layout(title=f"{setName} holdout {modelResults.model_name} mean probability<br>{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations")
            holdoutAucPlot = plotAUC(
                f"""
                    Receiver Operating Characteristic (ROC) Curve
                    {holdoutPlotSubtitle}
                    """,
                labelsPredictionsByInstance=holdoutLabelsProbabilities,
                config=config,
            )
            holdoutCalibrationPlot = plotCalibration(
                f"""
                    Calibration Curve
                    {holdoutPlotSubtitle}
                    """,
                holdoutLabelsPredictions,
                config=config,
            )
            (
                holdoutConfusionMatrixInstanceList,
                averageHoldoutConfusionMatrix,
            ) = plotConfusionMatrix(
                f"""
                    Confusion Matrix
                    {holdoutPlotSubtitle}
                    """,
                holdoutLabelsPredictions,
                config=config,
            )
            
            holdoutPlotPath = f"projects/{config['tracking']['project']}/{modelResults.model_name}/holdout/{setName}/plots"
            os.makedirs(holdoutPlotPath, exist_ok=True)
            
            holdoutAccuracyHistogram.write_html(
                f"{holdoutPlotPath}/{setName}__accuracyPlot.html"
            )
            holdoutProbabilityHistogram.write_html(
                f"{holdoutPlotPath}/{setName}__probabilityPlot.html"
            )
            holdoutAucPlot.savefig(
                f"{holdoutPlotPath}/{setName}__aucPlot.svg",
                bbox_inches="tight",
            )
            holdoutAucPlot.savefig(
                f"{holdoutPlotPath}/{setName}__aucPlot.png",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"{holdoutPlotPath}/{setName}__calibrationPlot.svg",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"{holdoutPlotPath}/{setName}__calibrationPlot.png",
                bbox_inches="tight",
            )
            averageHoldoutConfusionMatrix.savefig(
                f"{holdoutPlotPath}/{setName}__confusionMatrix.svg",
                bbox_inches="tight",
            )
            averageHoldoutConfusionMatrix.savefig(
                f"{holdoutPlotPath}/{setName}__confusionMatrix.png",
                bbox_inches="tight",
            )
            
    plotPath = f"projects/{config['tracking']['project']}/{modelResults.model_name}/plots"
    os.makedirs(
        plotPath,
        exist_ok=True,
    )

    accuracyHistogram.write_html(
        f"{plotPath}/accuracyPlot.html"
    )
    probabilityHistogram.write_html(
        f"{plotPath}/probabilityPlot.html"
    )
    aucPlot.savefig(
        f"{plotPath}/aucPlot.svg",
        bbox_inches="tight",
    )
    aucPlot.savefig(
        f"{plotPath}/aucPlot.png",
        bbox_inches="tight",
    )
    averageConfusionMatrix.savefig(
        f"{plotPath}/confusionMatrix.svg",
        bbox_inches="tight",
    )
    averageConfusionMatrix.savefig(
        f"{plotPath}/confusionMatrix.png",
        bbox_inches="tight",
    )

    calibrationPlot.savefig(
        f"{plotPath}/calibrationPlot.svg",
        bbox_inches="tight",
    )
    calibrationPlot.savefig(
        f"{plotPath}/calibrationPlot.png",
        bbox_inches="tight",
    )
    if config["model"]["hyperparameterOptimization"]:
        if convergencePlot is not None:
            convergencePlot.savefig(
                f"{plotPath}/convergencePlot.svg",
                bbox_inches="tight",
            )
            convergencePlot.savefig(
                f"{plotPath}/convergencePlot.png",
                bbox_inches="tight",
            )


def weighted_mean(data, mean_col, count_col):
    return (data[mean_col] * data[count_col]).sum() / data[count_col].sum()

def pooled_std(data, std_col, count_col):
    return ((data[std_col]**2 * (data[count_col] - 1)).sum() / (data[count_col].sum() - len(data)))**0.5

def poolSampleResults(concatenatedResults):
    # Group by 'id'
    grouped = concatenatedResults.groupby('id')

    # Initialize list to store results for each group
    pooled_results = []

    # Iterate over groups and calculate summary stats for each group
    for name, group in grouped:
        probability_mean = weighted_mean(group, 'probability_mean', 'draw_count')
        probability_std = pooled_std(group, 'probability_std', 'draw_count')
        accuracy_mean = weighted_mean(group, 'accuracy_mean', 'draw_count')
        accuracy_std = pooled_std(group, 'accuracy_std', 'draw_count')
        draw_count_sum = group['draw_count'].sum()
        first_label_instance = group['label'].iloc[0]
        
        # Redetermine the most frequent prediction
        mode_prediction =  group['prediction_most_frequent'].mode()[0]

        # Append results for this group to results list
        pooled_results.append({
            'id': name,
            'probability_mean': probability_mean,
            'probability_std': probability_std,
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'draw_count_sum': draw_count_sum,
            'first_label_instance': first_label_instance,
            'mode_prediction': mode_prediction
        })

    # Convert results list to DataFrame
    pooledSampleResults = pd.DataFrame(pooled_results).set_index('id')
    pooledSampleResults.rename(columns={'draw_count_sum': 'draw_count', 'first_label_instance': 'label', 'mode_prediction': 'prediction_most_frequent'}, inplace=True)
    
    return pooledSampleResults

def trackProjectVisualizations(classificationResults: ClassificationResults, config):
    # Concatenate sample results data frames from all model results
    concatenatedTestResults = pd.concat(
        [
            modelResults.test_results_dataframe
            for modelResults in classificationResults.modelResults
        ]
    )
    concatenatedExcessResults = pd.concat(
        [
            modelResults.excess_results_dataframe
            for modelResults in classificationResults.modelResults
        ]
    )
    
    pooledTestResults = poolSampleResults(concatenatedTestResults)

    output_path = f"projects/{config['tracking']['project']}/pooledTestResults_{config['tracking']['project']}.csv"
    np.set_printoptions(threshold=np.inf)
    pooledTestResults.to_csv(output_path)

    seenTestCases = (pooledTestResults["label"] == 1).sum()
    seenTestControls = (pooledTestResults["label"] == 0).sum()

    bootstrapTrainCount = (
        classificationResults.modelResults[0]
        .iteration_results[0]
        .train[0]
        .vectors.shape[0]
    )
    bootstrapTestCount = (
        classificationResults.modelResults[0]
        .iteration_results[0]
        .test[0]
        .vectors.shape[0]
    )

    featureCount = (
        classificationResults.modelResults[0]
        .iteration_results[0]
        .train[0]
        .vectors.shape[1]
    )
    
    featureCounts = [trainData.vectors.shape[1] for modelResults in classificationResults.modelResults for iteration_result in modelResults.iteration_results for trainData in iteration_result.train]
    featureCount =  featureCounts[0] if np.min(featureCounts) == np.max(featureCounts) else f"{np.min(featureCounts)}-{np.max(featureCounts)}"
    
    geneCounts = [trainData.geneCount for modelResults in classificationResults.modelResults for iteration_result in modelResults.iteration_results for trainData in iteration_result.train]
    geneCount = geneCounts[0] if np.min(geneCounts) == np.max(geneCounts) else f"{np.min(geneCounts)}-{np.max(geneCounts)}"

    testLabelsPredictions = {
        modelResults.model_name: (
            [
                label
                for iterationResult in modelResults.iteration_results
                for label in iterationResult.test_results_dataframe["label"].tolist()
            ],
            [
                prediction
                for iterationResult in modelResults.iteration_results
                for prediction in iterationResult.test_results_dataframe["prediction_most_frequent"].tolist()
            ],
        )
        for modelResults in classificationResults.modelResults
    }
    
    testLabelsProbabilities = {
        modelResults.model_name: (
             [
                label
                for iterationResult in modelResults.iteration_results
                for label in iterationResult.test_results_dataframe["label"].tolist()
            ],
             [
                probability
                for iterationResult in modelResults.iteration_results
                for probability in iterationResult.test_results_dataframe["probability_mean"].tolist()
            ],
        )
        for modelResults in classificationResults.modelResults
    }

    plotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations

    {config["tracking"]["name"]}, {featureCount} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else "variants (" + str(geneCount) + " genes)"}
    Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

    {seenTestCases} {config["clinicalTable"]["caseAlias"]}s @ {'{:.1%}'.format(np.mean([modelResults.average_test_case_accuracy for modelResults in classificationResults.modelResults]))} accuracy, {seenTestControls} {config["clinicalTable"]["controlAlias"]}s @ {'{:.1%}'.format(np.mean([modelResults.average_test_control_accuracy for modelResults in classificationResults.modelResults]))} accuracy
    {bootstrapTrainCount}±1 train, {bootstrapTestCount}±1 test samples per bootstrap iteration"""

    aucPlot = plotAUC(
        f"""
            Receiver Operating Characteristic (ROC) Curve
            {plotSubtitle}
            """,
        labelsPredictionsByInstance=testLabelsProbabilities,
        config=config,
    )

    calibrationPlot = plotCalibration(
        f"""
            Calibration Curve
            {plotSubtitle}
            """,
        testLabelsPredictions,
        config=config,
    )

    if classificationResults.modelResults[0].iteration_results[0].holdout:
        # assuming all models use same holdout sets
        for setName in classificationResults.modelResults[0].holdout_results_dataframe:
            concatenatedHoldoutResults = pd.concat(
                [
                    modelResults.holdout_results_dataframe[setName]
                    for modelResults in classificationResults.modelResults
                ]
            )
            # Group by 'id' and aggregate labels to get the most frequent label per id (which should be constant)
            pooledHoldoutResults = concatenatedHoldoutResults.groupby('id')['label'].agg(lambda x: x.value_counts().idxmax())
            
            # Count the number of cases (label 1) and controls (label 0)
            seenHoldoutCases = (pooledHoldoutResults == 1).sum()
            seenHoldoutControls = (pooledHoldoutResults == 0).sum()

            bootstrapHoldoutCount = seenHoldoutCases + seenHoldoutControls

            holdoutLabelsPredictions = {
                modelResults.model_name: (
                    [
                        label
                        for iterationResult in modelResults.iteration_results
                        for label in iterationResult.holdout_results_dataframe[setName]["label"].tolist()
                    ],
                    [
                        prediction
                        for iterationResult in modelResults.iteration_results
                        for prediction in iterationResult.holdout_results_dataframe[setName]["prediction_most_frequent"].tolist()
                    ],
                )
                for modelResults in classificationResults.modelResults
            }
            
            holdoutLabelsProbabilities = {
                modelResults.model_name: (
                    [
                        label
                        for iterationResult in modelResults.iteration_results
                        for label in iterationResult.holdout_results_dataframe[setName]["label"].tolist()
                    ],
                    [
                        probability
                        for iterationResult in modelResults.iteration_results
                        for probability in iterationResult.holdout_results_dataframe[setName]["probability_mean"].tolist()
                    ],
                )
                for modelResults in classificationResults.modelResults
            }

            holdoutPlotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
            
                {config["tracking"]["name"]}, {featureCount} {"genes" if config['vcfLike']['aggregateGenesBy'] != None else "variants (" + str(geneCount) + " genes)"}
                Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}

                {setName} holdout"""
            if setName in classificationResults.modelResults[0].average_holdout_case_accuracy:
                holdoutPlotSubtitle += f"\n{seenHoldoutCases} {config['clinicalTable']['caseAlias']}s @ {'{:.1%}'.format(np.mean([modelResults.average_holdout_case_accuracy[setName] for modelResults in classificationResults.modelResults]))} accuracy"
            if setName in classificationResults.modelResults[0].average_holdout_control_accuracy:
                holdoutPlotSubtitle += f"\n{seenHoldoutControls} {config['clinicalTable']['controlAlias']}s @ {'{:.1%}'.format(np.mean([modelResults.average_holdout_control_accuracy[setName] for modelResults in classificationResults.modelResults]))} accuracy"
            holdoutPlotSubtitle += f"\n{bootstrapHoldoutCount} holdout samples"

            holdoutAucPlot = plotAUC(
                f"""
                    Receiver Operating Characteristic (ROC) Curve
                    {holdoutPlotSubtitle}
                    """,
                labelsPredictionsByInstance=holdoutLabelsProbabilities,
                config=config,
            )
            holdoutCalibrationPlot = plotCalibration(
                f"""
                    Calibration Curve
                    {holdoutPlotSubtitle}
                    """,
                holdoutLabelsPredictions,
                config=config,
            )
            
            holdoutPlotPath = f"projects/{config['tracking']['project']}/holdout/{setName}"
            os.makedirs(holdoutPlotPath, exist_ok=True)
            
            holdoutAucPlot.savefig(
            f"{holdoutPlotPath}/{setName}__aucPlot.svg",
            bbox_inches="tight",
            )
            holdoutAucPlot.savefig(
                f"{holdoutPlotPath}/{setName}__aucPlot.png",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"{holdoutPlotPath}/{setName}__calibrationPlot.svg",
                bbox_inches="tight",
            )
            holdoutCalibrationPlot.savefig(
                f"{holdoutPlotPath}/{setName}__calibrationPlot.png",
                bbox_inches="tight",
            )


    aucPlot.savefig(
        f"projects/{config['tracking']['project']}/aucPlot.svg", bbox_inches="tight"
    )
    aucPlot.savefig(
        f"projects/{config['tracking']['project']}/aucPlot.png", bbox_inches="tight"
    )

    calibrationPlot.savefig(
        f"projects/{config['tracking']['project']}/calibrationPlot.svg",
        bbox_inches="tight",
    )
    calibrationPlot.savefig(
        f"projects/{config['tracking']['project']}/calibrationPlot.png",
        bbox_inches="tight",
    )
