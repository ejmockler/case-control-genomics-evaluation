import asyncio
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner

import ray

from mlStack import remove_all_flows, runMLstack


def viscosity():
    # current_calculation = (Perplexity * Skewness^2) / (Variance + Kurtosis))
    # Vn = α * Vn-1 + β * current_calculation
    # where α > β assuming rheopectic viscosity (viscosity increases as easily-predicted samples are sequestered)
    pass


@task()
def measureIteration():
    # find viscosity of entire sample distribution
    # separate discordant & well-classified
    # find viscosity of discordant & well-classified
    # return values
    pass


@flow(task_runner=RayTaskRunner())
def splash():
    pass


if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()

    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())
