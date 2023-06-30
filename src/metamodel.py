import asyncio
import os
import pandas as pd
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import numpy as np
from numpy import array, float32  # to eval strings as arrays

from sklearn.metrics import log_loss

import ray

from mlStack import remove_all_flows, main as runMLstack
from metaconfig import metaconfig
from config import config


def inertia(y_true, y_pred):
    return np.divide(
        perplexity(y_true, y_pred), np.var(y_pred)
    )  # predictive inertia = perplexity / variance


def perplexity(y_true, y_pred):
    crossEntropy = log_loss(
        y_true, y_pred, labels=[0, 1], eps=1e-15
    )  # linear predictions (exactly 0 or 1) depend on offset of 1e-15 when log is applied to avoid NaN

    # perplexity = 2 ^ crossEntropy
    return np.power(2, crossEntropy)


@task()
def measureIterations(result):
    sampleInertia = pd.DataFrame(index=result.index)
    result["probability"] = result["probability"].apply(lambda x: array(eval(x))[:, 1])
    # find inertia of entire sample distribution
    sampleInertia["inertia"] = result.apply(
        lambda row: inertia(
            [row["label"]] * len(row["probability"]), row["probability"]
        ),
        axis=1,
    )
    # separate discordant & well-classified
    discordantSamples = result.loc[
        result["accuracy"] <= metaconfig["samples"]["discordantThreshold"],
    ]
    accurateSamples = result.loc[
        result["accuracy"] >= metaconfig["samples"]["accurateThreshold"],
    ]
    return (
        sampleInertia,
        discordantSamples,
        accurateSamples,
    )


@flow(task_runner=RayTaskRunner())
def main():
    newWellClassified = True
    countSuffix = 0
    projectSummaryPath = f"projects/{config['tracking']['project']}__summary"
    baseProjectPath = config["tracking"]["project"]
    while newWellClassified:
        countSuffix += 1
        if countSuffix >= 2:
            config["sampling"]["lastIteration"] = 0
        config["tracking"]["project"] = f"{baseProjectPath}__{str(countSuffix)}"
        runMLstack(config)
        currentResults = pd.read_csv(
            f"projects/{config['tracking']['project']}/sampleResults.csv",
            index_col="id",
        )
        (
            sampleInertias,
            discordantSamples,
            accurateSamples,
        ) = measureIterations(currentResults)

        if len(
            currentResults.loc[
                (currentResults["label"] == 1)
                & (
                    currentResults["accuracy"]
                    >= metaconfig["samples"]["accurateThreshold"]
                )
            ]
        ) == len(currentResults):
            newWellClassified = False
            break

        sampleInertias.to_csv(
            f"projects/{config['tracking']['project']}/sampleInertias.csv"
        )
        accurateSamples.to_csv(
            f"projects/{config['tracking']['project']}/accurateSamples.csv"
        )
        discordantSamples.to_csv(
            f"projects/{config['tracking']['project']}/discordantSamples.csv"
        )
        # remove well-classified cases before next iteration
        config["sampling"]["sequesteredIDs"].append(
            accurateSamples.loc[accurateSamples["label"] == 1].index.tolist()
            + discordantSamples.loc[discordantSamples["label"] == 1].index.tolist()
        )
    os.makedirs(
        projectSummaryPath,
        exist_ok=True,
    )


if __name__ == "__main__":
    # TODO replace notebook code with src imports
    ray.shutdown()
    main()
    clearHistory = False
    if clearHistory:
        asyncio.run(remove_all_flows())
