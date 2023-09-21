import asyncio
import os
import neptune
import ray
import matplotlib

from tasks.data import (
    ClassificationResults,
    recoverPastRuns,
)
from tasks.visualize import trackModelVisualizations, trackProjectVisualizations

matplotlib.use("agg")


from prefect import flow, task
from sklearn.model_selection import StratifiedKFold

from tasks.input import (
    processInputFiles,
)
from tasks.predict import (
    bootstrap,
)
from config import config
from models import stack as modelStack

from joblib import Parallel, delayed


@task()
def download_file(run_id, field="sampleResults", extension="csv", config=config):
    path = f"./{field}/{run_id}.{extension}"
    if not os.path.exists(field):
        os.makedirs(field, exist_ok=True)
    if os.path.isfile(path):
        return
    run = neptune.init_run(
        with_id=run_id,
        project=config["entity"] + "/" + config["project"],
        api_token=config["neptuneApiToken"],
    )
    try:
        if field == "globalFeatureImportance" or field == "testLabels":
            for i in range(11):
                path = f"./{field}/{run_id}_{i}.{extension}"
                run[f"{field}/{i}"].download(destination=path)
            averageResultsPath = f"./{field}/{run_id}_average.{extension}"
            run[f"{field}/average"].download(destination=averageResultsPath)
        else:
            run[field].download(destination=path)
    except:
        pass
    run.stop()


@flow()
def main(
    config=config,
    genotypeData=None,
    clinicalData=None,
):
    if genotypeData is None:
        (
            genotypeData,
            clinicalData,
        ) = processInputFiles(config)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator

    bootstrap_args = [
        (
            genotypeData.case.genotype,
            genotypeData.control.genotype,
            genotypeData.holdout_case.genotype,
            genotypeData.holdout_control.genotype,
            clinicalData,
            model,
            hyperParameterSpace,
            innerCvIterator,
            outerCvIterator,
            config,
        )
        for model, hyperParameterSpace in list(modelStack.items())
    ]

    # results = []
    # for args in bootstrap_args:
    #     results.append(bootstrap(*args))

    results = Parallel(n_jobs=-1)(delayed(bootstrap)(*args) for args in bootstrap_args)
    classificationResults = ClassificationResults(modelResults=results)

    # TODO recover existing runs into results dataclass
    # if config["sampling"]["lastIteration"] > 0:
    #     results = recoverPastRuns(moding(np.array(x), separator=",")
    # )

    if config["model"]["calculateShapelyExplanations"]:
        pass

    for i in range(len(modelStack)):
        modelResult = classificationResults.modelResults[i]
        modelResult.calculate_average_accuracies()
        os.makedirs(
            f"projects/{config['tracking']['project']}/{modelResult.model_name}/",
            exist_ok=True,
        )
        modelResult.sample_results_dataframe.to_csv(
            f"projects/{config['tracking']['project']}/{modelResult.model_name}/sampleResults.csv"
        )
        if modelResult.average_global_feature_explanations is not None:
            modelResult.average_global_feature_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/globalFeatures.csv"
            )
        if modelResult.average_test_local_case_explanations is not None:
            modelResult.average_test_local_case_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/testCaseLocalFeatures.csv"
            )
        if modelResult.average_test_local_control_explanations is not None:
            modelResult.average_test_local_control_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/testControlLocalFeatures.csv"
            )
        if modelResult.average_holdout_local_case_explanations is not None:
            modelResult.average_holdout_local_case_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/holdoutCaseLocalFeatures.csv"
            )
        if modelResult.average_holdout_local_control_explanations is not None:
            modelResult.average_holdout_local_control_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/holdoutControlLocalFeatures.csv"
            )
        trackModelVisualizations(modelResult, config=config)

    trackProjectVisualizations(
        classificationResults,
        config=config,
    )

    return (
        results,
        genotypeData,
        clinicalData,
    )


async def remove_all_flows():
    from prefect.client import get_client

    orion_client = get_client()
    flows = await orion_client.read_flows()
    for flow in flows:
        flow_id = flow.id
        print(f"Deleting flow: {flow.name}, {flow_id}")
        await orion_client._client.delete(f"/flows/{flow_id}")
        print(f"Flow with UUID {flow_id} deleted")


if __name__ == "__main__":
    ray.shutdown()

    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())

    main()
