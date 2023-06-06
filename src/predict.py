from prefect import task, flow, unmapped
from prefect_ray.task_runners import RayTaskRunner
import os
import logging

from lion_pytorch import Lion
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, roc_auc_score, auc
from sklearn.preprocessing import MinMaxScaler
from skopt.plots import plot_convergence

from skopt import BayesSearchCV

from fastnumbers import check_real

from types import SimpleNamespace
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
import neptune
from neptune.types import File

import shap

from inspect import isclass
from io import StringIO
import traceback

import numpy as np
import pandas as pd
from config import config

# stop API errors when awaiting results
# !prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=True


def getFeatureImportances(model, data, featureLabels):
    """Get feature importances from fitted model and create SHAP explainer"""
    modelValuesDF = None
    if model.__class__.__name__ == "MultinomialNB":
        modelValuesDF = pd.DataFrame()
        for i, c in enumerate(
            model.feature_count_[0]
            if len(model.feature_count_.shape) > 1
            else model.feature_count_
        ):
            modelValuesDF.loc[
                i, f"log_prob_{config['clinicalTable']['controlAlias']}"
            ] = model.feature_log_prob_[0][i]
            modelValuesDF.loc[
                i, f"log_prob_{config['clinicalTable']['caseAlias']}"
            ] = model.feature_log_prob_[1][i]
    elif hasattr(model, "coef_"):
        modelValuesDF = pd.DataFrame()
        if len(model.coef_.shape) > 1:
            try:
                modelValuesDF[
                    f"feature_importances_{config['clinicalTable']['controlAlias']}"
                ] = model.coef_[0]
                modelValuesDF[
                    f"feature_importances_{config['clinicalTable']['caseAlias']}"
                ] = model.coef_[1]
            except IndexError:
                modelValuesDF[
                    f"feature_importances_{config['clinicalTable']['caseAlias']}"
                ] = model.coef_[0]
        else:
            modelValuesDF["feature_importances"] = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        modelValuesDF = pd.DataFrame()
        modelValuesDF[
            f"feature_importances_{config['clinicalTable']['caseAlias']}"
        ] = model.feature_importances_

    if type(modelValuesDF) == pd.DataFrame:
        modelValuesDF.index = featureLabels

    # Cluster correlated and hierarchical features using masker
    masker = shap.maskers.Partition(data, clustering="correlation")

    shapExplainer = shap.explainers.Permutation(
        model.predict_proba if hasattr(model, "predict_proba") else model.predict,
        masker,
        feature_names=["_".join(label) for label in featureLabels],
    )
    shapValues = shapExplainer(data)
    return modelValuesDF, shapValues, shapExplainer


@task()
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


@task()
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
def plotConfusionMatrix():
    pass


@task()
def plotSampleAccuracy():
    pass


@task()
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

    genotypeExcessIDs, genotypeTrainIDs = [], []
    # match IDs between genotype and clinical data; dataframe labels have label suffixes
    unmatchedTrainIDs = balancedMajorIDs + minorIDs
    for label in tqdm(allGenotypes.columns, desc="Matching IDs", unit="ID"):
        for setType in ["excess", "train"]:
            idSet = excessMajorIDs if setType == "excess" else unmatchedTrainIDs
            for i, id in enumerate(idSet):
                if (id in label) or (label in id):
                    if setType == "train":
                        if label not in genotypeTrainIDs:
                            genotypeTrainIDs.append(label)
                    elif setType == "excess":
                        if label not in genotypeExcessIDs:
                            genotypeExcessIDs.append(label)
                    idSet = np.delete(idSet, i)
                    break

    if verbose:
        print(f"\n{len(genotypeTrainIDs)} for training:\n{genotypeTrainIDs}")
        print(f"\n{len(genotypeExcessIDs)} are excess:\n{genotypeExcessIDs}")
        print(f"\nVariant count: {len(allGenotypes.index)}")

    samples = allGenotypes.loc[:, genotypeTrainIDs].dropna(
        how="any"
    )  # drop variants with missing values
    excessMajorSamples = allGenotypes.loc[:, genotypeExcessIDs]

    variantIndex = list(samples.index)
    pass
    scaler = MinMaxScaler()
    embedding = {
        "sampleIndex": genotypeTrainIDs,
        "labels": np.array([1 if id in caseIDs else 0 for id in genotypeTrainIDs]),
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
def evaluate(
    trainIndices,
    testIndices,
    model,
    labels,
    samples,
    variantIndex,
    sampleIndex,
    parameterSpace,
    cvIterator,
):
    fittedOptimizer = optimizeHyperparameters(
        samples[trainIndices],
        labels[trainIndices],
        model,
        parameterSpace,
        cvIterator,
        "neg_mean_squared_error",
    )
    model.set_params(**fittedOptimizer.best_params_)
    model.fit(samples[trainIndices], labels[trainIndices])
    # get model prediction probabilities
    try:
        probabilities = model.predict_proba(samples[testIndices])
    except AttributeError:
        probabilities = model.predict(samples[testIndices])
        if len(probabilities.shape) <= 1:
            probabilities = np.array([[1 - p, p] for p in probabilities])
    predictions = np.argmax(probabilities, axis=1)
    modelValues, shapValues, shapExplainer = getFeatureImportances(
        model, samples[testIndices], variantIndex
    )
    globalExplanations = modelValues
    localExplanations = shapValues
    trainLabels = np.array(labels[trainIndices])
    testLabels = np.array(labels[testIndices])
    trainIDs = np.array([sampleIndex[i] for i in trainIndices])
    testIDs = np.array([sampleIndex[i] for i in testIndices])
    return (
        globalExplanations,
        localExplanations,
        probabilities,
        predictions,
        testLabels,
        trainLabels,
        trainIDs,
        testIDs,
        fittedOptimizer,
        shapExplainer,
    )


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


@task()
def download_file(run_id, field="sampleResults", extension="csv"):
    path = f"./{field}/{run_id}.{extension}"
    if os.path.isfile(path):
        return
    if not os.path.exists(f"./{field}"):
        os.makedirs(f"./{field}")
    run = neptune.init_run(
        with_id=run_id,
        project=config["entity"] + "/" + config["project"],
        api_token=config["neptuneApiToken"],
    )
    run[field].download(destination=path)
    run.stop()


def serializeDataFrame(dataframe):
    stream = StringIO()
    dataframe.to_csv(stream)
    return File.from_stream(stream, extension="csv")


@flow(
    task_runner=RayTaskRunner(
        init_kwargs={
            "configure_logging": True,
            "logging_level": logging.WARN,
        }
    )
)
async def classify(
    caseGenotypes,
    caseIDs,
    controlGenotypes,
    controlIDs,
    clinicalData,
):
    matplotlib.use("agg")
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator
    results = {}
    projectTracker = neptune.init_project(
        project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
        api_token=config["tracking"]["token"],
    )
    for i in tqdm(
        range(
            config["sampling"]["startFrom"], config["sampling"]["bootstrapIterations"]
        ),
        unit="cohort",
    ):
        embedding = prepareDatasets(
            caseGenotypes, controlGenotypes, verbose=(True if i == 0 else False)
        )
        deserializedIDs = list()
        for id in embedding["sampleIndex"]:
            deserializedIDs.extend(id.split("__"))
        totalSampleCount = len(embedding["samples"])
        caseCount = np.count_nonzero(embedding["labels"])
        print(f"{totalSampleCount} samples\n")
        print(f"{caseCount} cases\n")
        print(f"{totalSampleCount - caseCount} controls\n")
        results[i] = {}
        results["samples"] = {}
        results["labels"] = {}
        for j, (model, parameterSpace) in enumerate(config["modelStack"].items()):
            if model.__class__.__name__ not in results:
                results[i][model.__class__.__name__] = {
                    "predictions": [],
                    "optimizeResults": [],
                }
            current = {}
            # check if model is initialized
            if isclass(model):
                if model.__name__ == "TabNetClassifier":
                    model = model(verbose=False, optimizer_fn=Lion)
            print(f"Iteration {i+1} with model {model.__class__.__name__}")
            runTracker = neptune.init_run(
                project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
                api_token=config["tracking"]["token"],
            )
            runTracker["sys/tags"].add(model.__class__.__name__)
            runTracker["bootstrapIteration"] = i + 1
            runTracker["config"] = {
                key: (item if check_real(item) or isinstance(item, str) else str(item))
                for key, item in config.items()
            }
            runTracker["embedding"].upload(
                serializeDataFrame(
                    pd.DataFrame(
                        data=embedding["samples"],
                        columns=embedding["variantIndex"],
                        index=embedding["sampleIndex"],
                    )
                )
            )
            runTracker["clinicalData"].upload(
                serializeDataFrame(
                    clinicalData.loc[clinicalData.index.isin(deserializedIDs)]
                )
            )
            # outer cross-validation
            crossValIndices = np.array(
                [
                    (cvTrainIndices, cvTestIndices)
                    for (cvTrainIndices, cvTestIndices) in outerCvIterator.split(
                        embedding["samples"], embedding["labels"]
                    )
                ]
            )
            current["trainIndices"] = crossValIndices[:, 0]
            current["testIndices"] = crossValIndices[:, 1]
            # this could be parallelized more efficiently since Shapely explanations are independent
            outerCrossValFutures = evaluate.map(
                trainIndices=crossValIndices[:, 0],
                testIndices=crossValIndices[:, 1],
                model=unmapped(model),
                labels=unmapped(embedding["labels"]),
                samples=unmapped(embedding["samples"]),
                variantIndex=unmapped(embedding["variantIndex"]),
                sampleIndex=unmapped(embedding["sampleIndex"]),
                parameterSpace=unmapped(parameterSpace),
                cvIterator=unmapped(innerCvIterator),
            )
            outerCrossValResults = zip(
                *[fold.result() for fold in outerCrossValFutures]
            )
            resultNames = [
                "globalExplanations",
                "localExplanations",
                "probabilities",
                "predictions",
                "testLabels",
                "trainLabels",
                "trainIDs",
                "testIDs",
                "fittedOptimizers",
                "shapExplainers",
            ]
            current = {
                **current,
                **{
                    name: result
                    for name, result in zip(resultNames, outerCrossValResults)
                },
            }

            # plot AUC & elbow
            runTracker["modelParams"] = {
                k + 1: current["fittedOptimizers"][k].best_params_
                for k in range(config["sampling"]["crossValIterations"])
            }
            plotSubtitle = f"""
            {config["tracking"]["name"]}, {embedding["samples"].shape[1]} variants
            Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
            {np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["caseAlias"]}s, {len(embedding['labels']) - np.count_nonzero(embedding['labels'])} {config["clinicalTable"]["controlAlias"]}s"""
            runTracker["plots/aucPlot"] = plotAUC(
                f"""
        Receiver Operating Characteristic (ROC) Curve
        {model.__class__.__name__} with {config['sampling']['crossValIterations']}-fold cross-validation
        {plotSubtitle}
        """,
                {
                    f"Fold {k+1}": (
                        current["testLabels"][k],
                        np.array(current["probabilities"][k])[:, 1],
                    )
                    if len(current["probabilities"][k][0].shape) >= 1
                    else (current["testLabels"][k], current["probabilities"][k])
                    for k in range(config["sampling"]["crossValIterations"])
                },
            )
            runTracker["plots/convergencePlot"] = plotOptimizer(
                f"""
        Hyperparameter convergence, mean squared error
        {model.__class__.__name__} with {config['sampling']['crossValIterations']}-fold cross-validation
        {plotSubtitle}
        """,
                {
                    f"Fold {k+1}": [
                        result
                        for result in current["fittedOptimizers"][k].optimizer_results_
                    ]
                    for k in range(config["sampling"]["crossValIterations"])
                },
            )

            # update model metrics
            current["testAUC"] = [
                roc_auc_score(
                    labels,
                    (
                        probabilities[:, 1]
                        if len(probabilities.shape) > 1
                        else probabilities
                    ),
                )
                for labels, probabilities in zip(
                    current["testLabels"], current["probabilities"]
                )
            ]
            runTracker["meanAUC"] = np.mean(current["testAUC"])
            # update sample metrics
            sampleResults = {}
            for fold in range(config["sampling"]["crossValIterations"]):
                for j, sampleID in enumerate(current["testIDs"][fold]):
                    try:
                        results["samples"][sampleID] += current["probabilities"][fold][
                            j
                        ]
                    except KeyError:
                        results["samples"][sampleID] = [
                            current["probabilities"][fold][j]
                        ]
                    finally:
                        results["labels"][sampleID] = current["testLabels"][fold][j]
            results[i][model.__class__.__name__] = current

            runTracker["shapExplanationsPerFold"].upload(
                File.as_pickle(current["localExplanations"])
            )
            runTracker["shapExplainersPerFold"].upload(
                File.as_pickle(current["shapExplainers"])
            )
            runTracker["trainIDs"].upload(File.as_pickle(current["trainIDs"]))
            runTracker["testIDs"].upload(File.as_pickle(current["testIDs"]))
            runTracker["testLabels"].upload(File.as_pickle(current["testLabels"]))
            runTracker["trainLabels"].upload(File.as_pickle(current["trainLabels"]))

            # plot feature importance
            for j in range(config["sampling"]["crossValIterations"]):
                if current["globalExplanations"][j] is not None:
                    runTracker[f"globalFeatureImportance/{j+1}"].upload(
                        serializeDataFrame(current["globalExplanations"][j])
                    )
                localExplanations = current["localExplanations"][j]
                caseExplanations = localExplanations
                caseExplanations.values = (
                    caseExplanations.values[:, :, 1]
                    if len(caseExplanations.values.shape) > 2
                    else caseExplanations.values
                )
                heatmap = plt.figure()
                plt.title(
                    f"""
          Shapely explanations from {model.__class__.__name__}
          Fold {j+1}
          {plotSubtitle}
          """
                )
                shap.plots.heatmap(caseExplanations, show=False)
                runTracker[f"plots/featureHeatmap/{j+1}"] = heatmap
                plt.close(heatmap)
                labelsProbabilities = (
                    (
                        current["testLabels"][j],
                        np.array(current["probabilities"][j])[:, 1],
                    )
                    if len(current["probabilities"][j][0].shape) >= 1
                    else (current["testLabels"][j], current["probabilities"][j])
                )
                stdDeviation = np.std(
                    (labelsProbabilities[1] - labelsProbabilities[0]) ** 2
                )
                for k in range(len(current["testIDs"][j])):
                    probability = (
                        labelsProbabilities[1][k]
                        if isinstance(labelsProbabilities[1][k], np.ndarray)
                        else labelsProbabilities[1][k]
                    )
                    label = (
                        labelsProbabilities[0][k]
                        if isinstance(labelsProbabilities[0][k], np.ndarray)
                        else labelsProbabilities[0][k]
                    )
                    if (
                        config["tracking"]["plotAllSampleImportances"]
                        or np.absolute((probability - label) ** 2) <= stdDeviation
                    ):
                        sampleID = current["testIDs"][j][k]
                        waterfallPlot = plt.figure()
                        plt.title(
                            f"""
              {sampleID}
              Shapely explanations from {model.__class__.__name__}
              Fold {j+1}
              {plotSubtitle}
              """
                        )
                        # patch parameter bug: https://github.com/slundberg/shap/issues/2362
                        to_pass = SimpleNamespace(
                            **{
                                "values": localExplanations[k].values,
                                "data": localExplanations[k].data,
                                "display_data": None,
                                "feature_names": localExplanations.feature_names,
                                "base_values": localExplanations[k].base_values[
                                    current["testLabels"][j][k]
                                ]
                                if len(localExplanations[k].base_values.shape) == 1
                                else localExplanations[k].base_values,
                            }
                        )
                        shap.plots.waterfall(to_pass, show=False)
                        try:
                            runTracker[
                                f"plots/samples/{j+1}/{sampleID}"
                            ] = waterfallPlot
                        except Exception as e:
                            runTracker[
                                f"plots/samples/{j+1}/{sampleID}"
                            ] = f"""failed to plot: {traceback.format_exc()}"""
                        plt.close(waterfallPlot)

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
                                id
                                for foldResults in current["testIDs"]
                                for id in foldResults
                            ],
                        },
                        dtype=object,
                    ).set_index("id")
                )
            )
            plt.close("all")
            runTracker.stop()

    labelsProbabilitiesByModelName = dict()
    if len(results == config["sampling"]["bootstrapIterations"]):
        for modelName in results[0].keys():
            labelsProbabilitiesByModelName[modelName] = [[], []]
            for k in range(config["sampling"]["bootstrapIterations"]):
                # append labels
                labelsProbabilitiesByModelName[modelName][0] = np.hstack(
                    [
                        labelsProbabilitiesByModelName[modelName][0],
                        np.concatenate(results[k][modelName]["testLabels"]),
                    ]
                )
                # append probabilities
                labelsProbabilitiesByModelName[modelName][1] = np.hstack(
                    [
                        labelsProbabilitiesByModelName[modelName][1],
                        np.concatenate(results[k][modelName]["probabilities"])[:, 1]
                        if len(results[k][modelName]["probabilities"][0].shape) >= 1
                        else np.concatenate(results[k][modelName]["probabilities"]),
                    ]
                )

        seenCaseCount, seenControlCount = 0, 0
        sampleResults = []

        for sampleID in results["samples"].keys():
            flattenedProbabilities = np.array(
                [
                    result[1] if len(caseExplanations.values.shape) > 2 else result
                    for foldResult in results["samples"][sampleID]
                    for result in foldResult
                ]
            )
            sampleResults += [
                {
                    "probability": results["samples"][sampleID][:1],
                    "accuracy": np.mean(
                        [
                            np.ceil(caseProbability) == results["labels"][sampleID]
                            for caseProbability in flattenedProbabilities
                        ]
                    ),
                    "id": sampleID,
                }
            ]
            if results["labels"][sampleID] == 1:
                seenCaseCount += 1
            else:
                seenControlCount += 1
    else:  # fetch results from Neptune
        import multiprocess as multiprocessing

        runs_table_df = projectTracker.fetch_runs_table().to_pandas()
        # Get the number of available CPUs
        cpu_count = multiprocessing.cpu_count()

        # Create a list of tuples containing the run ID and the corresponding function call
        sample_probability_tasks = [
            (download_file, run["sys/id"]) for _, run in runs_table_df.iterrows()
        ]
        sample_label_tasks = [
            (download_file, run["sys/id"], "testLabels", "pkl")
            for _, run in runs_table_df.iterrows()
        ]

        with multiprocessing.Pool(cpu_count) as pool:
            # Use the multiprocessing Pool to map the tasks to different processes
            pool.starmap(lambda func, *args: func(*args), sample_probability_tasks)
            pool.starmap(lambda func, *args: func(*args), sample_label_tasks)

    plotSubtitle = f"""{config['sampling']['crossValIterations']}x cross-validation over {config['sampling']['bootstrapIterations']} bootstrap iterations
  {config["tracking"]["name"]}, {embedding["samples"].shape[1]} variants
  Minor allele frequency over {'{:.1%}'.format(config['vcfLike']['minAlleleFrequency'])}
  {seenCaseCount} cases, {seenControlCount} controls
  """
    # print(f'preds 1: {results[0]["TabNetClassifier"]["predictions"]}\n')
    # print("labels:")
    # projectTracker["sampleResults"].upload(serializeDataFrame(pd.DataFrame.from_records(sampleResults, index="id")))
    projectTracker["aucPlot"].upload(
        plotAUC(
            f"""
    Receiver Operating Characteristic (ROC) Curve
    {plotSubtitle}
    """,
            labelsProbabilitiesByModelName,
        )
    )

    projectTracker["calibrationPlot"].upload(
        File.as_image(
            plotCalibration(
                f"""
    Calibration Curve
    {plotSubtitle}
    """,
                labelsProbabilitiesByModelName,
            )
        )
    )

    projectTracker["convergencePlot"].upload(
        File.as_image(
            plotOptimizer(
                f"""
    Convergence Plot
    {plotSubtitle}
    """,
                {
                    modelName: [
                        result
                        for k in range(config["sampling"]["bootstrapIterations"])
                        for foldOptimizer in results[k][modelName]["fittedOptimizers"]
                        for result in foldOptimizer.optimizer_results_
                    ]
                    for modelName in results[0].keys()
                    if modelName != "testLabels"
                },
            )
        )
    )
    projectTracker.stop()
    return results
