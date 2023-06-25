import asyncio
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import numpy as np

from sklearn.metrics import log_loss

import ray

from mlStack import remove_all_flows, runMLstack
from metaconfig import metaconfig


def viscosity(y_true, y_pred, last_viscosity=0):
    current_calculation = np.divide(
        perplexity(y_true, y_pred) / np.var(y_pred)
    )  # constant area of 1

    # Vn = α * Vn-1 + β * current_calculation
    # where α > β assuming rheopectic viscosity (viscosity increases as easily-predicted samples are sequestered)
    return metaconfig["viscosity"]["pastScalar"] * last_viscosity + (
        metaconfig["viscosity"]["currentScalar"] * current_calculation
    )


def perplexity(y_true, y_pred):
    crossEntropy = log_loss(
        y_true, y_pred
    )  # linear predictions (exactly 0 or 1) depend on default offset of 1e-15 when log is applied to avoid NaN

    # perplexity = 2 ^ crossEntropy
    return np.power(2, crossEntropy)


@task()
def measureIterations(result):
    # find viscosity of entire sample distribution
    totalViscosity = viscosity(
        result["sampleResultsDataframe"]["label"],
        result["sampleResultsDataframe"]["probability"],
    )
    # separate discordant & well-classified
    # find viscosity of discordant & well-classified
    # return values
    pass


@flow(task_runner=RayTaskRunner())
def splash():
    viscosityDict = {"discordant": [], "wellClassified": [], "all": []}
    resultCollection = []
    newWellClassified = True
    while newWellClassified:
        resultCollection.append(runMLstack())
        measureIterations(resultCollection[-1])
        pass


if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()

    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())
