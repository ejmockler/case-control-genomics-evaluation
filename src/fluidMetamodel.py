import asyncio
import os
import pandas as pd
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import numpy as np

from sklearn.metrics import log_loss

import ray

from mlStack import remove_all_flows, runMLstack
from metaconfig import metaconfig
from config import config


def viscosity(y_true, y_pred, last_viscosity=0):
    currentViscosity = np.divide(
        perplexity(y_true, y_pred) / np.var(y_pred)
    )  # constant area of 1

    # Vn = α * Vn-1 + β * current_calculation
    # where α > β assuming rheopectic viscosity (viscosity increases as easily-predicted samples are sequestered)
    nonlinearViscosity = metaconfig["viscosity"]["pastScalar"] * last_viscosity + (
        metaconfig["viscosity"]["currentScalar"] * currentViscosity
    )

    return nonlinearViscosity, currentViscosity


def perplexity(y_true, y_pred):
    crossEntropy = log_loss(
        y_true, y_pred
    )  # linear predictions (exactly 0 or 1) depend on default offset of 1e-15 when log is applied to avoid NaN

    # perplexity = 2 ^ crossEntropy
    return np.power(2, crossEntropy)


@task()
def measureIterations(result, runningViscosities):
    # find viscosity of entire sample distribution
    runningTotalViscosity, instantaneousTotalViscosity = viscosity(
        result["label"],
        result["probability"],
        runningViscosities[0],
    )
    # separate discordant & well-classified
    discordantSamples = result.loc[
        result["accuracy"] <= metaconfig["samples"]["discordantThreshold"],
    ]
    accurateSamples = result.loc[
        result["accuracy"] >= metaconfig["samples"]["accurateThreshold"],
    ]
    # find viscosity of discordant & well-classified
    runningDiscordantViscosity, instantaneousDiscordantViscosity = viscosity(
        discordantSamples["label"],
        discordantSamples["probability"],
        runningViscosities[1],
    )
    runningAccurateViscosity, instantaneousAccurateViscosity = viscosity(
        accurateSamples["label"], accurateSamples["probability"], runningViscosities[2]
    )
    return (
        (runningTotalViscosity, runningDiscordantViscosity, runningAccurateViscosity),
        (
            instantaneousTotalViscosity,
            instantaneousDiscordantViscosity,
            instantaneousAccurateViscosity,
        ),
        discordantSamples,
        accurateSamples,
    )


@flow(task_runner=RayTaskRunner())
def splash():
    runningViscosityDict = {"discordant": [], "wellClassified": [], "all": []}
    instantaneousViscosityDict = {"discordant": [], "wellClassified": [], "all": []}
    runningViscosities = (0, 0, 0)
    newWellClassified = True
    countSuffix = 0
    projectSummaryPath = f"projects/{config['tracking']['project']}__summary"
    baseProjectPath = config["tracking"]["project"]
    while newWellClassified:
        countSuffix += 1
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        runMLstack(config)
        currentResults = pd.read_csv(
            f"{config['tracking']['project']}/sampleResults.csv",
            index_col="id",
        )
        (
            runningViscosities,
            instantaneousViscosities,
            discordantSamples,
            accurateSamples,
        ) = measureIterations(currentResults, runningViscosities)

        if len(
            currentResults.loc[
                currentResults["accuracy"]
                >= metaconfig["samples"]["accurateThreshold"],
            ]
        ) == len(currentResults):
            newWellClassified = False
            break

        runningViscosityDict["all"].append(runningViscosities[0])
        runningViscosityDict["discordant"].append(runningViscosities[1])
        runningViscosityDict["wellClassified"].append(runningViscosities[2])

        instantaneousViscosityDict["all"].append(instantaneousViscosities[0])
        instantaneousViscosityDict["discordant"].append(instantaneousViscosities[1])
        instantaneousViscosityDict["wellClassified"].append(instantaneousViscosities[2])
        accurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/accurateSamples.csv"
        )
        discordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/discordantSamples.csv"
        )
        config["sampling"]["sequesteredIDs"].append(
            accurateSamples.loc[accurateSamples["label"] == 1]["id"].tolist()
            + discordantSamples.loc[discordantSamples["label"] == 1]["id"].tolist()
        )
    os.makedirs(
        projectSummaryPath,
        exist_ok=True,
    )
    pd.DataFrame.from_dict(runningViscosityDict).to_csv(
        f"{projectSummaryPath}/runningViscosity.csv"
    )
    pd.DataFrame.from_dict(instantaneousViscosityDict).to_csv(
        f"{projectSummaryPath}/instantaneousViscosity.csv"
    )


if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()
    splash()
    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())
