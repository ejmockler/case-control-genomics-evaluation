from typing import Iterable, Literal, Union
from prefect import unmapped, task, flow
from prefect_ray.task_runners import RayTaskRunner
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from config import config

from multiprocess import Pool, Manager, managers

from tasks.data import Genotype, GenotypeData

@task()
def filterTable(table, filterString):
    if not filterString:
        return table
    print(f"Filtering: {filterString}")
    filteredTable = table.query(filterString)
    return filteredTable

@task()
def applyAlleleModel(values, columns, genotypeIDs, clinicalSampleIDmap, config):
    # some genotype IDs are subset of column names (or vice versa)
    genotypeDict = dict()
    resolvedGenotypeIDs = set()
    seenSampleIDs = set()
    # iterate clinical sample IDs
    for id in tqdm(genotypeIDs, unit="id"):
        # if id in config["sampling"]["sequesteredIDs"]:
        #     continue
        # iterate genotype samples
        for j, column in enumerate(columns):
            matched = False
            subIDs = column.split(config["vcfLike"]["compoundSampleIdDelimiter"])
            if (column in id or id in column) and (len(column) > 3 and len(id) > 3) and (id not in clinicalSampleIDmap or clinicalSampleIDmap[id] not in seenSampleIDs or any([subID in clinicalSampleIDmap.values() for subID in subIDs])):
                if id in clinicalSampleIDmap: seenSampleIDs.update(clinicalSampleIDmap[id])  # prevent duplicate genotypes from clinical data
                else:
                    for subID in subIDs:  # handle case where only external subject ID is available
                        if subID in clinicalSampleIDmap.values():
                            seenSampleIDs.update(subID)
                            break
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
                # if column in config["sampling"]["sequesteredIDs"] or id in config["sampling"]["sequesteredIDs"]:
                #     break
                processed_genotypes = []
                for genotype in values[:, j]:
                    alleles = genotype.replace("'", "").split("/") if isinstance(genotype, str) else []
                    if all([allele.isdigit() for allele in alleles]):
                        if config["vcfLike"]["binarize"]:
                            # Binarize logic
                            result = np.clip(np.sum([int(allele) for allele in alleles]), a_min=0, a_max=1)
                        elif config["vcfLike"]["zygosity"]:
                            # Zygosity logic
                            if all(allele == "0" for allele in alleles):
                                result = 0
                            elif all(allele != "0" for allele in alleles):
                                result = 2
                            elif any(int(allele) > 1 for allele in alleles):
                                result = 2
                            else:
                                result = 1
                        else:
                            # Default case (sum of alleles)
                            if all([allele.isdigit() for allele in alleles]):
                                result = np.sum([int(allele) for allele in alleles])
                    else:
                        result = np.nan

                    processed_genotypes.append(result)

                genotypeDict[f"{column}"] = processed_genotypes
                columns = np.delete(columns, j)
                values = np.delete(values, j, axis=1)
                resolvedGenotypeIDs.update({id})
                break
    missingGenotypeIDs = (
        set(genotypeIDs) - resolvedGenotypeIDs
    )  # leftover columns are missing
    return genotypeDict, missingGenotypeIDs, resolvedGenotypeIDs


def aggregateIntoGenes(
    genotypeDataframe: pd.DataFrame,
    config,
):
    genes = genotypeDataframe.index.get_level_values(
        config["vcfLike"]["geneMultiIndexLevel"]
    )
    aggregatedGenotype = pd.DataFrame(
        index=genes.unique(), columns=genotypeDataframe.columns
    )

    for gene in genes:
        gene_rows = genotypeDataframe.iloc[genes == gene]
        match config["vcfLike"]["aggregateGenesBy"]:
            case "mean":  # similar to mean MAF (mean allele frequency)
                aggregatedGenotype.loc[gene] = gene_rows.mean(axis=0)
            case "sum":
                aggregatedGenotype.loc[gene] = gene_rows.sum(axis=0)
            case _:
                raise NotImplementedError
    return aggregatedGenotype


@task()
def load(config):
    clinicalData = pd.read_excel(
        config["clinicalTable"]["path"], index_col=config["clinicalTable"]["idColumn"]
    )
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
           
            na_values=[".", "NA"],
            keep_default_na=True,
        )
        if "xlsx" not in config["vcfLike"]["path"]
        else pd.read_excel(
            config["vcfLike"]["path"],
            sheet_name=(
                config["vcfLike"]["sheet"] if config["vcfLike"]["sheet"] else None
            ),
            dtype=str,
            na_values=[".", "NA"],
            keep_default_na=True,
        )
    )
    referenceVCF = (
        (pd.read_csv(
            config["vcfLike"]["frequencyMatchReference"],
            sep="\t",
            dtype=str,
           
            na_values=[".", "NA"],
            keep_default_na=True,
        )
        if "xlsx" not in config["vcfLike"]["frequencyMatchReference"]
        else pd.read_excel(
            config["vcfLike"]["frequencyMatchReference"],
            sheet_name=(
                config["vcfLike"]["sheet"] if config["vcfLike"]["sheet"] else None
            ),
            dtype=str,
            na_values=[".", "NA"],
            keep_default_na=True,
        )) if config["vcfLike"]["frequencyMatchReference"] else None
    )
    
    annotatedVCF = annotatedVCF.set_index(config["vcfLike"]["indexColumn"])
    referenceVCF = referenceVCF.set_index(config["vcfLike"]["indexColumn"]) if referenceVCF is not None else None
    
    return (
        clinicalData,
        externalSamples,
        annotatedVCF.loc[annotatedVCF.index.dropna()],
        referenceVCF.loc[referenceVCF.index.dropna()] if referenceVCF is not None else None
    )

def prepareCaseControlSamples(caseGenotypes, controlGenotypes, sample_frequencies, doBalance=True):
    caseIDs = caseGenotypes.columns
    controlIDs = controlGenotypes.columns
    
    # If balance is False, return all IDs without balancing
    if not doBalance:
        for id in list(caseIDs) + list(controlIDs):
            sample_frequencies[id] += 1
        return caseIDs, controlIDs, [], sample_frequencies
    
    # store number of cases & controls
    caseControlCounts = [len(caseIDs), len(controlIDs)]
    # determine which has more samples
    labeledIDs = [caseIDs, controlIDs]
    majorIDs = labeledIDs[np.argmax(caseControlCounts)]
    minorIDs = labeledIDs[np.argmin(caseControlCounts)]
    
    # Calculate weights inversely proportional to frequency
    weights = np.array([1.0 / (sample_frequencies[id] + 1) for id in majorIDs])
    
    # Normalize the weights so that they sum to 1 (required for probability distribution)
    weights /= weights.sum()
    
    # Downsample larger group to match smaller group
    majorIndex = np.random.choice(
        np.arange(len(majorIDs)), min(caseControlCounts), replace=False, p=weights
    )

    excessMajorIDs, balancedMajorIDs = [], []
    for index, id in enumerate(majorIDs):
        if index in majorIndex:
            balancedMajorIDs.append(id)
        else:
            excessMajorIDs.append(id)

    for id in balancedMajorIDs + list(minorIDs):
        sample_frequencies[id] += 1

    return minorIDs, balancedMajorIDs, excessMajorIDs, sample_frequencies

def prepareDatasets(
    caseGenotypes,
    controlGenotypes,
    holdoutCaseGenotypes,
    holdoutControlGenotypes,
    sampleFrequencies,
    verbose=True,
    config=config,
    freqReferenceGenotypeData=None,
):
    minorIDs, balancedMajorIDs, excessMajorIDs, sampleFrequencies = prepareCaseControlSamples(
        caseGenotypes, controlGenotypes, sampleFrequencies, doBalance=True
    )

    # Only balance samples used in training
    (
        holdoutMinorIDs,
        holdoutBalancedMajorIDs,
        holdoutExcessMajorIDs,
        sampleFrequencies
    ) = prepareCaseControlSamples(holdoutCaseGenotypes, holdoutControlGenotypes, sampleFrequencies, doBalance=False)
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
    controlIDs = controlGenotypes.columns

    excessIDs, crossValGenotypeIDs = [], []
    holdoutExcessIDs, holdoutTestIDs = [], []
    trainIDs = np.hstack([balancedMajorIDs, minorIDs])
    holdoutIDs = np.hstack([holdoutBalancedMajorIDs, holdoutMinorIDs])
    
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

    # drop variants with missing values or invariant
    preCleanedVariantCounts = len(allGenotypes.index)
    allGenotypes = allGenotypes.dropna(
            how="any",
        ).loc[allGenotypes[crossValGenotypeIDs].std(axis=1) > 0]
    print(f"Dropped {preCleanedVariantCounts - len(allGenotypes.index)} variants with missing values or invariant")

    samples = allGenotypes.loc[:, crossValGenotypeIDs]
    excessMajorSamples = allGenotypes.loc[:, excessIDs]

    variantIndex = samples.index
    scaler = MinMaxScaler()
    # TODO dataclass
    embedding = {
        "sampleIndex": np.array(crossValGenotypeIDs),
        "labels": np.array([1 if id in caseIDs else 0 for id in crossValGenotypeIDs]),
        "samples": scaler.fit_transform(
            samples
        ).transpose(),  # samples are now rows (samples, variants)
        "excessMajorIndex": np.array(excessIDs),
        "excessMajorLabels": [1 if id in caseIDs else 0 for id in (excessIDs)],
        "excessMajorSamples": scaler.fit_transform(excessMajorSamples).transpose() if not excessMajorSamples.empty else np.array([]),
        "excessMajorSetName": "excess case" if all([id in caseIDs for id in excessIDs]) else "excess control" if all([id in controlIDs for id in controlIDs]) else "mixed excess",
        "variantIndex": variantIndex,
    }
    if len(holdoutCaseGenotypes) > 0 and len(holdoutControlGenotypes) > 0:
        holdoutSamples = allGenotypes.loc[:, holdoutTestIDs]
        excessHoldoutSamples = allGenotypes.loc[:, holdoutExcessIDs]
        try:
            embedding = {
                **embedding,
                **{
                    "holdoutSampleIndex": np.array(holdoutTestIDs),
                    "holdoutLabels": np.array(
                        [1 if id in holdoutCaseIDs else 0 for id in holdoutTestIDs]
                    ),
                    "holdoutSamples": scaler.fit_transform(holdoutSamples).transpose(),
                    "excessHoldoutMajorIndex": np.array(holdoutExcessIDs),
                    "excessHoldoutMajorLabels": [
                        1 if id in holdoutCaseIDs else 0 for id in holdoutExcessIDs
                    ],
                    "excessHoldoutMajorSamples": scaler.fit_transform(
                        excessHoldoutSamples
                        ).transpose() if not excessHoldoutSamples.empty else np.array([]),
                },
            }
        except ValueError as e:
            print(e)
            raise e
    return sampleFrequencies, embedding


def createGenotypeDataframe(genotype_dict, filteredVCF):
    df = pd.DataFrame.from_dict(genotype_dict)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index.name = filteredVCF.index.name
    df.index = filteredVCF.index
    return df


def processGenotypes(filteredVCF, clinicalSampleIDmap, caseIDs, controlIDs, config):
    # cast genotypes as numeric, drop chromosome positions with missing values
    caseGenotypeFutures, controlGenotypeFutures = applyAlleleModel.map(
        unmapped(filteredVCF.to_numpy()),
        unmapped(filteredVCF.columns.to_numpy()),
        genotypeIDs=[IDs for IDs in (caseIDs, controlIDs)],
        clinicalSampleIDmap=unmapped(clinicalSampleIDmap),
        config=unmapped(config),
    )

    caseGenotypeDict, missingCaseIDs, resolvedCaseIDs = caseGenotypeFutures.result()
    controlGenotypeDict, missingControlIDs, resolvedControlIDs = controlGenotypeFutures.result()

    return caseGenotypeDict, controlGenotypeDict, missingCaseIDs, missingControlIDs, resolvedCaseIDs, resolvedControlIDs


@task()
def integrateExternalSampleIDs(filteredExternalSamples, config, caseIDs, controlIDs):
    holdoutCaseIDs = np.array([])
    holdoutControlIDs = np.array([])
    for i, label in enumerate(config["externalTables"]["label"]):
        if config["externalTables"]["setType"][i] == "holdout":
            if label == config["clinicalTable"]["caseAlias"]:
                holdoutCaseIDs = np.append(
                    holdoutCaseIDs, [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
            elif label == config["clinicalTable"]["controlAlias"]:
                holdoutControlIDs = np.append(
                    holdoutControlIDs, [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
        elif config["externalTables"]["setType"][i] == "crossval":
            if label == config["clinicalTable"]["caseAlias"]:
                caseIDs = np.append(
                    caseIDs, [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
            elif label == config["clinicalTable"]["controlAlias"]:
                controlIDs = np.append(
                    controlIDs, [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
    return caseIDs, controlIDs, holdoutCaseIDs, holdoutControlIDs

@task()
def findClosestFrequencyVariant(reference_frequencies, input_frequencies):
    """
    Identifies the closest matching variant in the input dataset for each variant in the reference dataset based on allele frequency.

    Args:
    - reference_frequencies (pd.Series): Allele frequencies of the reference dataset's variants.
    - input_frequencies (pd.Series): Allele frequencies of the input dataset's variants.

    Returns:
    - closest_variants (list): Indices of variants in the input dataset that are closest in frequency to each variant in the reference dataset.
    """
    closest_variants = []
    if not reference_frequencies.empty and not input_frequencies.empty:
        for ref_freq in reference_frequencies:
            differences = (input_frequencies - ref_freq).abs()
            # input variant with the smallest difference in frequency
            closest_variant = differences.idxmin()
            closest_variants.append(closest_variant)
            input_frequencies = input_frequencies.drop(closest_variant)
    else:
        print("One or both of the frequency datasets are empty.")
    return closest_variants


@flow(task_runner=RayTaskRunner(), log_prints=True)
def processInputFiles(config):
    clinicalData, externalSamples, annotatedVCF, referenceVCF = load(config)

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
    if referenceVCF is not None:
        filteredReferenceVCF = filterTable(referenceVCF, config["vcfLike"]["filters"])
        print(
            f"filtered {referenceVCF.shape[0] - filteredReferenceVCF.shape[0]} variants from reference VCF"
        )

    caseIDsMask, controlIDsMask = [
        filteredClinicalData[config["clinicalTable"]["labelColumn"]]
        .isin(labels)
        .dropna()
        for labels in (
            config["clinicalTable"]["caseLabels"],
            config["clinicalTable"]["controlLabels"],
        )
    ]

    clinicalSampleIDmap = {id: filteredClinicalData.loc[id, config["clinicalTable"]["subjectIdColumn"]] for id in filteredClinicalData.index.tolist()}
    caseIDs = caseIDsMask[caseIDsMask].index.to_numpy()
    controlIDs = controlIDsMask[controlIDsMask].index.to_numpy()

    caseIDs, controlIDs, holdoutCaseIDs, holdoutControlIDs = integrateExternalSampleIDs(filteredExternalSamples, config, caseIDs, controlIDs)

    caseGenotypeDict, controlGenotypeDict, missingCaseIDs, missingControlIDs, resolvedCaseIDs, resolvedControlIDs = processGenotypes(
        filteredVCF, clinicalSampleIDmap, caseIDs, controlIDs, config
    )
    
    if len(holdoutCaseIDs) > 0:
        holdoutCaseGenotypeDict, holdoutControlGenotypeDict, missingHoldoutCaseIDs, missingHoldoutControlIDs, resolvedHoldoutCaseIDs, resolvedHoldoutControlIDs = processGenotypes(
            filteredVCF, clinicalSampleIDmap, holdoutCaseIDs, holdoutControlIDs, config
        )
    else:
        holdoutCaseGenotypeDict = {}
        holdoutControlGenotypeDict = {}
        resolvedHoldoutCaseIDs, missingHoldoutCaseIDs = None, None
        resolvedHoldoutControlIDs, missingHoldoutControlIDs = None, None
        
    if referenceVCF is not None:
        freqReferenceCaseGenotypeDict, freqReferenceControlGenotypeDict, missingFreqReferenceCaseIDs, missingFreqReferenceControlIDs, resolvedFreqReferenceCaseIDs, resolvedFreqReferenceControlIDs = processGenotypes(
            filteredReferenceVCF, clinicalSampleIDmap, caseIDs, controlIDs, config
        )
        # ensure reference & input share same resolved IDs
        resolvedCaseIDs = list(set(resolvedCaseIDs).intersection(set(resolvedFreqReferenceCaseIDs)))
        resolvedControlIDs = list(set(resolvedControlIDs).intersection(set(resolvedFreqReferenceControlIDs)))
    else:
        freqReferenceCaseGenotypeDict = {}
        freqReferenceControlGenotypeDict = {}
        missingFreqReferenceCaseIDs, missingFreqReferenceControlIDs = None, None
    
    for alias, (IDs, genotypeDict) in {
        "caseAlias": (missingCaseIDs, caseGenotypeDict),
        "controlAlias": (missingControlIDs, controlGenotypeDict),
        "holdout cases": (missingHoldoutCaseIDs, holdoutCaseGenotypeDict),
        "holdout controls": (missingHoldoutControlIDs, holdoutControlGenotypeDict),
        "caseAlias": (missingFreqReferenceCaseIDs, freqReferenceCaseGenotypeDict),
        "controlAlias": (missingFreqReferenceControlIDs, freqReferenceControlGenotypeDict),
    }.items():
        if len(genotypeDict) == 0: continue
        if len(IDs) > 0:
            if "holdout" not in alias:
                print(
                    f"\nmissing {len(IDs)} {config['clinicalTable'][alias]} IDs:\n {IDs}"
                )
            elif "holdout" in alias:
                print(f"\nmissing {len(IDs)} {alias} IDs:\n {IDs}")

    resolvedIDs = np.hstack([list(caseGenotypeDict.keys()), list(controlGenotypeDict.keys())])
    allGenotypes = createGenotypeDataframe({**caseGenotypeDict, **controlGenotypeDict}, filteredVCF).dropna().astype(np.int8)
    if referenceVCF is not None:
        freqReferenceAllGenotypes = createGenotypeDataframe({**freqReferenceCaseGenotypeDict, **freqReferenceControlGenotypeDict}, filteredReferenceVCF).dropna().astype(np.int8)
    
    if isinstance(allGenotypes.index, pd.MultiIndex):
        # Manually check for nulls in each level of the MultiIndex
        non_null_indices = ~allGenotypes.index.to_frame().isnull().any(axis=1)
        allGenotypes = allGenotypes[non_null_indices]
    else:
        # For a standard index, retain rows with non-null indices
        allGenotypes = allGenotypes[allGenotypes.index.notnull()]
        
    # Calculate the allele frequencies
    allele_frequencies = (
        allGenotypes.gt(0).sum(axis=1) / len(resolvedIDs)
    ).loc[
        lambda x: x.between(config["vcfLike"]["minAlleleFrequency"], config["vcfLike"]["maxAlleleFrequency"])
    ]

    # Filter the genotypes based on frequency criteria
    if referenceVCF is not None:
        reference_allele_frequencies = (
                freqReferenceAllGenotypes.gt(0).sum(axis=1) / len(resolvedIDs)
            ).loc[
            lambda x: x.between(config["vcfLike"]["minAlleleFrequency"], config["vcfLike"]["maxAlleleFrequency"])
        ]
        matched_allele_frequencies = findClosestFrequencyVariant(reference_allele_frequencies, allele_frequencies)
        frequencyFilteredGenotypes = allGenotypes.loc[
            matched_allele_frequencies
        ].sort_index()
        referenceFrequencyFilteredGenotypes = freqReferenceAllGenotypes.loc[reference_allele_frequencies].sort_index()
        print(f"Matched {len(frequencyFilteredGenotypes)} alleles to reference VCF {config['vcfLike']['frequencyMatchReference']}")
    else:
        frequencyFilteredGenotypes = allGenotypes.loc[
            allele_frequencies.index
        ]
        
    print(
        f"Filtered {len(filteredVCF) - len(frequencyFilteredGenotypes)} alleles with frequency below {'{:.3%}'.format(config['vcfLike']['minAlleleFrequency'])} or above {'{:.3%}'.format(config['vcfLike']['maxAlleleFrequency'])}"
    )
    print(f"Kept {len(frequencyFilteredGenotypes)} alleles")
    
    caseGenotypesDataframe = createGenotypeDataframe(caseGenotypeDict, filteredVCF).loc[frequencyFilteredGenotypes.index]
    controlGenotypesDataframe = createGenotypeDataframe(
        controlGenotypeDict, filteredVCF
    ).loc[frequencyFilteredGenotypes.index]

    holdoutCaseGenotypesDataframe = (
        createGenotypeDataframe(holdoutCaseGenotypeDict, filteredVCF).loc[frequencyFilteredGenotypes.index]
        if resolvedHoldoutCaseIDs
        else pd.DataFrame(index=frequencyFilteredGenotypes.index)
    )
    holdoutControlGenotypesDataframe = (
        createGenotypeDataframe(holdoutControlGenotypeDict, filteredVCF).loc[frequencyFilteredGenotypes.index]
        if resolvedHoldoutControlIDs
        else pd.DataFrame(index=frequencyFilteredGenotypes.index)
    )
    
    if referenceVCF is not None:
        referenceCaseGenotypesDataframe = createGenotypeDataframe(freqReferenceCaseGenotypeDict, filteredReferenceVCF).loc[referenceFrequencyFilteredGenotypes.index]
        referenceControlGenotypesDataframe = createGenotypeDataframe(
            freqReferenceControlGenotypeDict, filteredReferenceVCF
        ).loc[referenceFrequencyFilteredGenotypes.index]

    if config["vcfLike"]["aggregateGenesBy"] != None:
        caseGenotypesDataframe = aggregateIntoGenes(caseGenotypesDataframe, config)
        controlGenotypesDataframe = aggregateIntoGenes(
            controlGenotypesDataframe, config
        )
        holdoutCaseGenotypesDataframe = aggregateIntoGenes(
            holdoutCaseGenotypesDataframe, config
        )
        holdoutControlGenotypesDataframe = aggregateIntoGenes(
            holdoutControlGenotypesDataframe, config
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
    if referenceVCF is not None:
        referenceCaseGenotypes = Genotype(referenceCaseGenotypesDataframe, resolvedFreqReferenceCaseIDs, "Reference Case")
        referenceControlGenotypes = Genotype(referenceControlGenotypesDataframe, resolvedFreqReferenceControlIDs, "Reference Control")

    genotypeData = GenotypeData(
        caseGenotypes,
        holdoutCaseGenotypes,
        controlGenotypes,
        holdoutControlGenotypes,
    )
    
    if referenceVCF is not None:
        frequencyReferenceGenotypeData = {
            "case": referenceCaseGenotypes,
            "control": referenceControlGenotypes
        }
    else:
        frequencyReferenceGenotypeData = None

    print(f"\n{len(resolvedCaseIDs)} cases:\n {resolvedCaseIDs}")
    print(f"\n{len(resolvedControlIDs)} controls:\n {resolvedControlIDs}")
    if resolvedHoldoutCaseIDs:
        print(f"\n{len(resolvedHoldoutCaseIDs)} holdout cases:\n {holdoutCaseIDs}")
    if resolvedHoldoutControlIDs:
        print(
            f"\n{len(resolvedHoldoutControlIDs)} holdout controls:\n {holdoutControlIDs}"
        )

    return genotypeData, frequencyReferenceGenotypeData, filteredClinicalData


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
