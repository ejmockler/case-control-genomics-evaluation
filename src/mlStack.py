import asyncio
import os
from traceback import print_exception
import ray
import matplotlib
import numpy as np

from tasks.data import (
    ClassificationResults,
    recoverPastRuns,
)
from tasks.visualize import trackModelVisualizations, trackProjectVisualizations

# multiprocessing-friendly backend
matplotlib.use("agg")

from prefect import flow
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

@flow()
async def main(
    config=config,
    genotypeData=None,
    freqReferenceGenotypeData = None,
    clinicalData=None,
    trackVisualizations=True,
):
    if (genotypeData is None and freqReferenceGenotypeData is None and clinicalData is None):
        (
            genotypeData,
            freqReferenceGenotypeData,
            clinicalData,
        ) = processInputFiles(config)
    outerCvIterator = StratifiedKFold(
        n_splits=config["sampling"]["crossValIterations"], shuffle=False
    )
    innerCvIterator = outerCvIterator

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
        )
        for model, hyperParameterSpace in list(modelStack.items())
    ]
    results = None
    try:
        # TODO: represent in workflow engine
        results = Parallel(n_jobs=-1)(delayed(bootstrap)(*args) for args in bootstrap_args)
    except Exception as e:
        print(f"Error during bootstrap: {e}")
        print_exception(e)
    classificationResults = ClassificationResults(modelResults=results)

    # TODO recover existing runs into results dataclass
    # if config["sampling"]["lastIteration"] > 0:
    #     results = recoverPastRuns(moding(np.array(x), separator=",")
    # )

    if config["model"]["calculateShapelyExplanations"]:
        pass
    
    np.set_printoptions(threshold=np.inf)
    for i in range(len(modelStack)):
        modelResult = classificationResults.modelResults[i]
        modelResult.calculate_average_accuracies()
        os.makedirs(
            f"projects/{config['tracking']['project']}/{modelResult.model_name}/",
            exist_ok=True,
        )
        modelResult.test_results_dataframe.to_csv(
            f"projects/{config['tracking']['project']}/{modelResult.model_name}/testResults_{modelResult.model_name}_{config['tracking']['project']}.csv"
        )
        for setName in modelResult.holdout_results_dataframe:
            os.makedirs(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/holdout/{setName}/",
                exist_ok=True,
            )
            modelResult.holdout_results_dataframe[setName].to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/holdout/{setName}/{setName}_{modelResult.model_name}_{config['tracking']['project']}.csv"
            )
            if config['model']['calculateShapelyExplanations']:
                modelResult.average_holdout_local_explanations[setName].to_csv(
                    f"projects/{config['tracking']['project']}/{modelResult.model_name}/holdout/{setName}/localFeatures__{setName}_{config['tracking']['project']}.csv"
                )
        modelResult.excess_results_dataframe.to_csv(
            f"projects/{config['tracking']['project']}/{modelResult.model_name}/excessResults_{modelResult.model_name}_{config['tracking']['project']}.csv"
        )
        if modelResult.average_global_feature_explanations is not None:
            modelResult.average_global_feature_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/globalFeatures_{modelResult.model_name}_{config['tracking']['project']}.csv"
            )
        if config['model']['calculateShapelyExplanations']:
            modelResult.average_test_local_explanations.to_csv(
                f"projects/{config['tracking']['project']}/{modelResult.model_name}/testLocalFeatures_{modelResult.model_name}_{config['tracking']['project']}.csv"
            )

        if trackVisualizations: trackModelVisualizations(modelResult, config=config)

    if trackVisualizations: trackProjectVisualizations(
        classificationResults,
        config=config,
    )

    return (
        classificationResults,
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

    asyncio.run(main())
