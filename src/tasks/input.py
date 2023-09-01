from prefect import unmapped, task, flow
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from multiprocess import Pool, Manager, managers

from tasks.data import Genotype, GenotypeData


@task(retries=1000)
def filterTable(table, filterString):
    if not filterString:
        return table
    print(f"Filtering: {filterString}")
    filteredTable = table.query(filterString)
    return filteredTable


@task(retries=1000)
def applyAlleleModel(values, columns, genotypeIDs, config):
    # some genotype IDs are subset of column names (or vice versa)
    genotypeDict = dict()
    resolvedGenotypeIDs = set()
    for id in tqdm(genotypeIDs, unit="id"):
        if id in config["sampling"]["sequesteredIDs"]:
            continue
        for j, column in enumerate(columns):
            matched = False
            if column == id:
                matched = True
            else:
                # check for compound sample IDs
                if config["vcfLike"]["compoundSampleIdDelimiter"] in column:
                    delimitedColumn = column.split(
                        config["vcfLike"]["compoundSampleIdDelimiter"]
                    )[config["vcfLike"]["compoundSampleIdStartIndex"] :]
                    matched = any(columnValue == id for columnValue in delimitedColumn)
                if config["vcfLike"]["compoundSampleIdDelimiter"] in id:
                    delimitedID = id.split(
                        config["vcfLike"]["compoundSampleIdDelimiter"]
                    )[config["vcfLike"]["compoundSampleIdStartIndex"] :]
                    matched = any(idValue == column for idValue in delimitedID)
                if (
                    config["vcfLike"]["compoundSampleIdDelimiter"] in column
                    and config["vcfLike"]["compoundSampleIdDelimiter"] in id
                ):
                    matched = any(idValue in delimitedColumn for idValue in delimitedID)
            if matched:
                if column in config["sampling"]["sequesteredIDs"]:
                    break
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
                            a_min=0,
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


@task(retries=1000)
def load(config):
    clinicalData = pd.read_excel(
        config["clinicalTable"]["path"], index_col=config["clinicalTable"]["idColumn"]
    ).drop_duplicates(subset=config["clinicalTable"]["subjectIdColumn"])
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


def balanceCaseControlDatasets(caseGenotypes, controlGenotypes):
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

    return minorIDs, balancedMajorIDs, excessMajorIDs


@task(retries=1000)
def prepareDatasets(
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    verbose=True,
):
    minorIDs, balancedMajorIDs, excessMajorIDs = balanceCaseControlDatasets(
        caseGenotypes, controlGenotypes
    )

    (
        holdoutMinorIDs,
        holdoutBalancedMajorIDs,
        holdoutExcessMajorIDs,
    ) = balanceCaseControlDatasets(holdoutCaseGenotypes, holdoutControlGenotypes)
    holdoutCaseIDs = holdoutCaseGenotypes.columns

    allGenotypes = pd.concat(
        [
            caseGenotypes,
            controlGenotypes,
            holdoutCaseGenotypes,
            holdoutControlGenotypes,
        ],
        axis=1,
    )
    caseIDs = caseGenotypes.columns

    excessIDs, crossValGenotypeIDs = [], []
    holdoutExcessIDs, holdoutTestIDs = [], []
    trainIDs = balancedMajorIDs + minorIDs
    holdoutIDs = holdoutBalancedMajorIDs + holdoutMinorIDs
    for label in tqdm(allGenotypes.columns, desc="Matching IDs", unit="ID"):
        for setType in ["holdout", "crossval"]:
            if (
                setType == "holdout"
                and len(holdoutCaseGenotypes) <= 0
                and len(holdoutControlGenotypes) <= 0
            ):
                continue
            for subsetType in ["excess", "toSample"]:
                idSet = (
                    (holdoutExcessMajorIDs if setType == "holdout" else excessMajorIDs)
                    if subsetType == "excess"
                    else holdoutIDs
                    if setType == "holdout" and subsetType == "toSample"
                    else trainIDs
                )
                for i, id in enumerate(idSet):
                    if (id in label) or (label in id):
                        if setType == "crossval" and subsetType == "toSample":
                            if label not in crossValGenotypeIDs:
                                crossValGenotypeIDs.append(label)
                        elif setType == "crossval" and subsetType == "excess":
                            if label not in excessIDs:
                                excessIDs.append(label)
                        elif setType == "holdout" and subsetType == "toSample":
                            if label not in holdoutTestIDs:
                                holdoutTestIDs.append(label)
                        elif setType == "holdout" and subsetType == "excess":
                            if label not in holdoutExcessIDs:
                                holdoutExcessIDs.append(label)
                        idSet = np.delete(idSet, i)
                        break

    if verbose:
        print(f"\n{len(crossValGenotypeIDs)} for training:\n{crossValGenotypeIDs}")
        print(f"\n{len(excessIDs)} are excess:\n{excessIDs}")
        if len(holdoutCaseGenotypes) > 0 and len(holdoutControlGenotypes) > 0:
            print(f"\n{len(holdoutTestIDs)} for holdout:\n{holdoutTestIDs}")
            print(f"\n{len(holdoutExcessIDs)} are excess holdout:\n{holdoutExcessIDs}")
        print(f"\nVariant count: {len(allGenotypes.index)}")

    # drop variants with missing values
    allGenotypes = allGenotypes.dropna(
        how="any",
    )

    samples = allGenotypes.loc[:, crossValGenotypeIDs]
    excessMajorSamples = allGenotypes.loc[:, excessIDs]

    variantIndex = list(samples.index)
    scaler = MinMaxScaler()
    # TODO dataclass
    embedding = {
        "sampleIndex": crossValGenotypeIDs,
        "labels": np.array([1 if id in caseIDs else 0 for id in crossValGenotypeIDs]),
        "samples": scaler.fit_transform(
            samples
        ).transpose(),  # samples are now rows (samples, variants)
        "excessMajorIndex": excessIDs,
        "excessMajorLabels": [1 if id in caseIDs else 0 for id in excessIDs],
        "excessMajorSamples": scaler.fit_transform(excessMajorSamples).transpose(),
        "variantIndex": variantIndex,
    }
    if len(holdoutCaseGenotypes) > 0 and len(holdoutControlGenotypes) > 0:
        holdoutSamples = allGenotypes.loc[:, holdoutTestIDs]
        excessHoldoutSamples = allGenotypes.loc[:, holdoutExcessIDs]
        embedding = {
            **embedding,
            **{
                "holdoutSampleIndex": holdoutTestIDs,
                "holdoutLabels": np.array(
                    [1 if id in holdoutCaseIDs else 0 for id in holdoutTestIDs]
                ),
                "holdoutSamples": scaler.fit_transform(holdoutSamples).transpose(),
                "excessHoldoutMajorIndex": holdoutExcessIDs,
                "excessHoldoutMajorLabels": [
                    1 if id in holdoutCaseIDs else 0 for id in holdoutExcessIDs
                ],
                "excessHoldoutMajorSamples": scaler.fit_transform(
                    excessHoldoutSamples
                ).transpose(),
            },
        }
    return embedding


def createGenotypeDataframe(genotype_dict, filteredVCF):
    df = pd.DataFrame.from_dict(genotype_dict)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index.name = filteredVCF.index.name
    df.index = filteredVCF.index
    return df


@flow(task_runner=ConcurrentTaskRunner(), log_prints=True)
def processInputFiles(config):
    clinicalData, externalSamples, annotatedVCF = load(config)

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

    holdoutCaseIDs = np.array([])
    holdoutControlIDs = np.array([])
    for i, label in enumerate(config["externalTables"]["label"]):
        if config["externalTables"]["setType"][i] == "holdout":
            if label == config["clinicalTable"]["caseAlias"]:
                holdoutCaseIDs = np.append(
                    holdoutCaseIDs,
                    [
                        id
                        for id in filteredExternalSamples[i].index.to_numpy()
                        if id not in config["sampling"]["sequesteredIDs"]
                    ],
                )
            elif label == config["clinicalTable"]["controlAlias"]:
                holdoutControlIDs = np.append(
                    holdoutControlIDs,
                    [
                        id
                        for id in filteredExternalSamples[i].index.to_numpy()
                        if id not in config["sampling"]["sequesteredIDs"]
                    ],
                )

        elif config["externalTables"]["setType"][i] == "crossval":
            if label == config["clinicalTable"]["caseAlias"]:
                caseIDs = np.append(
                    caseIDs,
                    [
                        id
                        for id in filteredExternalSamples[i].index.to_numpy()
                        if id not in config["sampling"]["sequesteredIDs"]
                    ],
                )
            elif label == config["clinicalTable"]["controlAlias"]:
                controlIDs = np.append(
                    controlIDs,
                    [
                        id
                        for id in filteredExternalSamples[i].index.to_numpy()
                        if id not in config["sampling"]["sequesteredIDs"]
                    ],
                )

    # cast genotypes as numeric, drop chromosome positions with missing values
    caseGenotypeFutures, controlGenotypeFutures = applyAlleleModel.map(
        unmapped(filteredVCF.to_numpy()),
        unmapped(filteredVCF.columns.to_numpy()),
        genotypeIDs=[IDs for IDs in (caseIDs, controlIDs)],
        config=unmapped(config),
    )

    caseGenotypeDict, missingCaseIDs, resolvedCaseIDs = caseGenotypeFutures.result()
    (
        controlGenotypeDict,
        missingControlIDs,
        resolvedControlIDs,
    ) = controlGenotypeFutures.result()

    resolvedHoldoutCaseIDs, missingHoldoutCaseIDs = [], []
    resolvedHoldoutControlIDs, missingHoldoutControlIDs = [], []
    if len(holdoutCaseIDs) > 0:
        (
            holdoutCaseGenotypeFutures,
            holdoutControlGenotypeFutures,
        ) = applyAlleleModel.map(
            unmapped(filteredVCF.to_numpy()),
            unmapped(filteredVCF.columns.to_numpy()),
            genotypeIDs=[IDs for IDs in (holdoutCaseIDs, holdoutControlIDs)],
            config=unmapped(config),
        )
        (
            holdoutCaseGenotypeDict,
            missingHoldoutCaseIDs,
            resolvedHoldoutCaseIDs,
        ) = holdoutCaseGenotypeFutures.result()
        (
            holdoutControlGenotypeDict,
            missingHoldoutControlIDs,
            resolvedHoldoutControlIDs,
        ) = holdoutControlGenotypeFutures.result()
    else:
        holdoutCaseGenotypeDict = {}
        holdoutControlGenotypeDict = {}

    for alias, (IDs, genotypeDict) in {
        "caseAlias": (missingCaseIDs, caseGenotypeDict),
        "controlAlias": (missingControlIDs, controlGenotypeDict),
        "holdout cases": (missingHoldoutCaseIDs, holdoutCaseGenotypeDict),
        "holdout controls": (missingHoldoutControlIDs, holdoutControlGenotypeDict),
    }.items():
        sequesteredIDs = set(config["sampling"]["sequesteredIDs"]).intersection(
            set(genotypeDict.keys())
        )
        IDs = set(IDs) - sequesteredIDs
        if len(IDs) > 0:
            if "holdout" not in alias:
                print(
                    f"\nmissing {len(IDs)} {config['clinicalTable'][alias]} IDs:\n {IDs}"
                )
            elif "holdout" in alias:
                print(f"\nmissing {len(IDs)} {alias} IDs:\n {IDs}")
        if len(sequesteredIDs) > 0:
            if "holdout" not in alias:
                print(
                    f"\nsequestered {len(sequesteredIDs)} {config['clinicalTable'][alias]} IDs:\n {sequesteredIDs}"
                )
            elif "holdout" in alias:
                print(f"\nsequestered {len(sequesteredIDs)} {alias} IDs:\n {IDs}")

    caseGenotypesDataframe = createGenotypeDataframe(caseGenotypeDict, filteredVCF)
    controlGenotypesDataframe = createGenotypeDataframe(
        controlGenotypeDict, filteredVCF
    )

    holdoutCaseGenotypesDataframe = (
        createGenotypeDataframe(holdoutCaseGenotypeDict, filteredVCF)
        if resolvedHoldoutCaseIDs
        else pd.DataFrame()
    )

    holdoutControlGenotypesDataframe = (
        createGenotypeDataframe(holdoutControlGenotypeDict, filteredVCF)
        if resolvedHoldoutControlIDs
        else pd.DataFrame()
    )

    caseGenotypes = Genotype(caseGenotypesDataframe, resolvedCaseIDs, "Case")
    controlGenotypes = Genotype(
        controlGenotypesDataframe, resolvedControlIDs, "Control"
    )
    holdoutCaseGenotypes = Genotype(
        holdoutCaseGenotypesDataframe, resolvedHoldoutCaseIDs, "Holdout Case"
    )
    holdoutControlGenotypes = Genotype(
        holdoutControlGenotypesDataframe,
        resolvedHoldoutControlIDs,
        "Holdout Control",
    )

    genotypeData = GenotypeData(
        caseGenotypes,
        holdoutCaseGenotypes,
        controlGenotypes,
        holdoutControlGenotypes,
    )

    print(f"\n{len(resolvedCaseIDs)} cases:\n {resolvedCaseIDs}")
    print(f"\n{len(resolvedControlIDs)} controls:\n {resolvedControlIDs}")
    if resolvedHoldoutCaseIDs:
        print(f"\n{len(resolvedHoldoutCaseIDs)} holdout cases:\n {holdoutCaseIDs}")
    if resolvedHoldoutControlIDs:
        print(
            f"\n{len(resolvedHoldoutControlIDs)} holdout controls:\n {holdoutControlIDs}"
        )

    return genotypeData, filteredClinicalData


def toMultiprocessDict(orig_dict, manager):
    shared_dict = manager.dict()
    for key, value in orig_dict.items():
        if isinstance(value, dict):
            shared_dict[key] = toMultiprocessDict(
                value, manager
            )  # convert inner dict to manager dict
        else:
            shared_dict[key] = value
    return shared_dict


def fromMultiprocessDict(shared_dict):
    orig_dict = {
        k: fromMultiprocessDict(v) if isinstance(v, managers.DictProxy) else v
        for k, v in shared_dict.items()
    }
    return orig_dict
