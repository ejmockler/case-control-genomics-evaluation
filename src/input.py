from prefect import unmapped, task, flow
from prefect.task_runners import ConcurrentTaskRunner
from tqdm import tqdm
from config import config
import pandas as pd
import numpy as np


@task()
def filterTable(table, filterString):
    if not filterString:
        return table
    print(f"Filtering: {filterString}")
    filteredTable = table.query(filterString)
    return filteredTable


@task()
def applyAlleleModel(values, columns, genotypeIDs):
    # some genotype IDs are subset of column names (or vice versa)
    genotypeDict = dict()
    resolvedGenotypeIDs = set()
    for id in tqdm(genotypeIDs, unit="id"):
        for j, column in enumerate(columns):
            if id in column or column in id:
                # implement allele model
                genotypeDict[f"{column}"] = [
                    (
                        np.sum(
                            [
                                int(allele)
                                for allele in genotype.replace("'", "").split("/")
                            ]
                        )  # split by allele delimiter
                        if not config["vcfLike"]["binarize"]
                        else np.clip(
                            np.sum(
                                [
                                    int(allele)
                                    for allele in genotype.replace("'", "").split("/")
                                ]
                            ),
                            a_max=1,
                            a_min=None,
                        )
                    )
                    if any(char.isdigit() for char in genotype)
                    else np.nan
                    for genotype in values[:, j]
                ]
                columns = np.delete(columns, j)
                values = np.delete(values, j, axis=1)
                resolvedGenotypeIDs.update({id})
                break
    missingGenotypeIDs = (
        set(genotypeIDs) - resolvedGenotypeIDs
    )  # leftover columns are missing
    return genotypeDict, missingGenotypeIDs, resolvedGenotypeIDs


@task()
def load():
    clinicalData = pd.read_excel(
        config["clinicalTable"]["path"], index_col=config["clinicalTable"]["idColumn"]
    ).drop_duplicates(subset=config["clinicalTable"]["uniqueIdColumn"])
    externalSamples = [
        pd.read_csv(path, sep="\t", index_col=idColumn)
        for path, idColumn in zip(
            config["externalTables"]["path"], config["externalTables"]["idColumn"]
        )
    ]
    annotatedVCF = (
        pd.read_csv(
            config["vcfLike"]["path"],
            sep="\t",
            dtype=str,
            index_col=config["vcfLike"]["indexColumn"],
        )
        if "xlsx" not in config["vcfLike"]["path"]
        else pd.read_excel(
            config["vcfLike"]["path"],
            sheet_name=(
                config["vcfLike"]["sheet"] if config["vcfLike"]["sheet"] else None
            ),
            dtype=str,
            na_values=["."],
            keep_default_na=False,
        )
    )
    # remove null chromosome positions
    annotatedVCF[config["vcfLike"]["indexColumn"]] = (
        annotatedVCF[config["vcfLike"]["indexColumn"]].astype(str).replace("", np.nan)
    )
    return (
        clinicalData,
        externalSamples,
        annotatedVCF.dropna(subset=config["vcfLike"]["indexColumn"]).set_index(
            config["vcfLike"]["indexColumn"]
        ),
    )


@flow(task_runner=ConcurrentTaskRunner(), log_prints=True)
async def processInputFiles():
    clinicalData, externalSamples, annotatedVCF = load()

    filteredClinicalData = filterTable(clinicalData, config["clinicalTable"]["filters"])
    print(
        f"filtered {len(clinicalData) - len(filteredClinicalData)} samples from clinical data"
    )
    filteredExternalSamples = [
        filterTable(externalSampleTable, filterString)
        for externalSampleTable, filterString in zip(
            externalSamples, config["externalTables"]["filters"]
        )
    ]
    for i, (externalSampleTable, path) in enumerate(
        zip(filteredExternalSamples, config["externalTables"]["path"])
    ):
        print(
            f"filtered {len(externalSamples[i]) - len(externalSampleTable)} samples from external data {path}"
        )
    filteredVCF = filterTable(annotatedVCF, config["vcfLike"]["filters"])
    print(f"filtered {annotatedVCF.shape[0] - filteredVCF.shape[0]} variants from VCF")

    caseIDsMask, controlIDsMask = [
        filteredClinicalData[config["clinicalTable"]["labelColumn"]]
        .isin(labels)
        .dropna()
        for labels in (
            config["clinicalTable"]["caseLabels"],
            config["clinicalTable"]["controlLabels"],
        )
    ]

    caseIDs = caseIDsMask[caseIDsMask].index.to_numpy()
    controlIDs = controlIDsMask[controlIDsMask].index.to_numpy()
    for i, label in enumerate(config["externalTables"]["label"]):
        if label == config["clinicalTable"]["caseAlias"]:
            caseIDs = np.append(caseIDs, filteredExternalSamples[i].index.to_numpy())
        elif label == config["clinicalTable"]["controlAlias"]:
            controlIDs = np.append(
                controlIDs, filteredExternalSamples[i].index.to_numpy()
            )

    # cast genotypes as numeric, drop chromosome positions with missing values
    caseGenotypeFutures, controlGenotypeFutures = applyAlleleModel.map(
        unmapped(filteredVCF.to_numpy()),
        unmapped(filteredVCF.columns.to_numpy()),
        genotypeIDs=[IDs for IDs in (caseIDs, controlIDs)],
    )
    caseGenotypeDict, missingCaseIDs, resolvedCaseIDs = caseGenotypeFutures.result()
    (
        controlGenotypeDict,
        missingControlIDs,
        resolvedControlIDs,
    ) = controlGenotypeFutures.result()

    if len(missingCaseIDs) > 0 or len(missingControlIDs) > 0:
        for alias, IDs in {
            "caseAlias": missingCaseIDs,
            "controlAlias": missingControlIDs,
        }.items():
            print(f"\nmissing {len(IDs)} {config['clinicalTable'][alias]} IDs:\n {IDs}")

    caseGenotypes = pd.DataFrame.from_dict(caseGenotypeDict)
    caseGenotypes.index.name = filteredVCF.index.name
    caseGenotypes.index = filteredVCF.index
    controlGenotypes = pd.DataFrame.from_dict(controlGenotypeDict)
    controlGenotypes.index.name = filteredVCF.index.name
    controlGenotypes.index = filteredVCF.index

    caseIDs = resolvedCaseIDs
    controlIDs = resolvedControlIDs

    print(f"\n{len(caseIDs)} cases:\n {caseIDs}")
    print(f"\n{len(controlIDs)} controls:\n {controlIDs}")
    # filter allele frequencies
    allGenotypes = pd.concat(
        [
            caseGenotypes.dropna(how="any", axis=0),
            controlGenotypes.dropna(how="any", axis=0),
        ],
        axis=1,
    )
    filteredGenotypes = allGenotypes.loc[
        allGenotypes.gt(0).sum(axis=1).divide(len(allGenotypes.columns))
        >= config["vcfLike"]["minAlleleFrequency"]
    ]
    print(
        f"Filtered {len(filteredVCF) - len(filteredGenotypes)} alleles with frequency below {'{:.3%}'.format(config['vcfLike']['minAlleleFrequency'])}"
    )
    print(f"Kept {len(filteredGenotypes)} alleles")

    caseGenotypes = filteredGenotypes.loc[:, caseGenotypes.columns]
    controlGenotypes = filteredGenotypes.loc[:, controlGenotypes.columns]

    return [caseGenotypes, caseIDs, controlGenotypes, controlIDs, filteredClinicalData]
