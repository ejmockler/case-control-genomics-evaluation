import os
from scipy.stats import mode
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
    filteredTable = table.query(filterString, engine="python")
    return filteredTable

@task()
def processAlleles(values, columns, genotypeIDs, clinicalSampleIDmap, config):
    """Resolve genotype IDs & apply configured allele model

    Args:
        values (ndarray): Genotype matrix with variants as rows and samples as columns. 
        columns (list): Genotype sample column names.
        genotypeIDs (list): Selected genotype IDs to resolve.
        clinicalSampleIDmap (dict): Mapping of clinical IDs to genotype IDs (for samples in clinical data)
        config: Set in src/config.py

    Returns:
        tuple: genotypeDict, missingGenotypeIDs (dict), resolvedGenotypeIDs (dict)
    """
    # some genotype IDs are subset of column names (or vice versa)
    resolvedGenotypeIDs = dict()
    # lookup table for allele values to impute missing genotypes with most common value
    alleleModeMap = {}
    # if genotypeIDs is not a dict with holdout set names, convert to dict for crossval IDs
    if not isinstance(genotypeIDs, dict):
        genotypeDict = {"crossval": {}}
        genotypeIDs = {"crossval": genotypeIDs}
        resolvedGenotypeIDs = {"crossval": set()}
        missingGenotypeIDs = {"crossval": set()}
    else:
        genotypeDict = {setName: {} for setName in genotypeIDs}
        resolvedGenotypeIDs = {setName: set() for setName in genotypeIDs}
        missingGenotypeIDs = {setName: set() for setName in genotypeIDs}
    #originalColumns = np.copy(columns)
    #originalValues = np.copy(values)
    # iterate sample IDs
    for setName, idSet in genotypeIDs.items():
        seenSampleIDs = set()
        #columns = np.copy(originalColumns)
        #values = np.copy(originalValues)
        for id in tqdm(idSet, unit="id", desc=f"Processing {setName} IDs"):
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
                    processed_genotypes = []
                    for i, genotype in enumerate(values[:, j]):
                        alleles = genotype.replace("'", "").split("/") if isinstance(genotype, str) else []
                        if all([allele.isdigit() for allele in alleles]):
                            # Compute result based on allele model
                            result = applyAlleleModel(alleles, config)
                        else:
                            # If allele mode not already calculated, compute and store it
                            if i not in alleleModeMap:
                                # Extract alleles for all samples for this variant
                                all_alleles = [genotype.replace("'", "").replace('.', "").split("/") for genotype in values[i] if isinstance(genotype, str)]
                                # Calculate mode for each allele based on allele model
                                allele_mode = computeAlleleMode(all_alleles, config)
                                alleleModeMap[i] = allele_mode
                            result = alleleModeMap[i]

                        processed_genotypes.append(result)

                    genotypeDict[setName][f"{column}"] = processed_genotypes
                    #columns = np.delete(columns, j)
                    #values = np.delete(values, j, axis=1)
                    resolvedGenotypeIDs[setName].update({id})
                    break
        missingGenotypeIDs[setName] = (
            set(idSet) - resolvedGenotypeIDs[setName]
        )  # leftover columns are missing
    return genotypeDict, missingGenotypeIDs, resolvedGenotypeIDs

def applyAlleleModel(alleles, config):
    if config["vcfLike"]["binarize"]:
        return np.clip(np.sum([int(allele) for allele in alleles]), a_min=0, a_max=1)
    elif config["vcfLike"]["zygosity"]:
        if all(allele == "0" for allele in alleles):
            return 0
        elif all(allele != "0" for allele in alleles):
            return 2
        elif any(int(allele) > 1 for allele in alleles):
            return 2
        else:
            return 1
    else:
        return np.sum([int(allele) for allele in alleles])

def computeAlleleMode(all_alleles, config):
    # Process all_alleles to match the allele model (binary, zygosity, sum, etc.)
    processed_alleles = [applyAlleleModel(a, config) for a in all_alleles if all(a)]
    most_common = mode(processed_alleles, keepdims=True)
    return most_common.mode[0] if most_common.count[0] > 0 else np.nan

def processGenotypes(filteredVCF, clinicalSampleIDmap, caseIDs, controlIDs, config):
    """Process genotypes with configured allele model & resolve sample IDs in VCF-like input. Returned ID lists are keyed to set name; either holdout name or "crossval".

    Args:
        filteredVCF (DataFrame): VCF-like data with variants as rows and samples as columns.
        clinicalSampleIDmap (dict): Mapping of genotype IDs to clinical IDs (for genotypes in clinical data).
        caseIDs (np.array || dict): List of crossval case IDs. If a dict, ID lists are keyed by set name.
        controlIDs (np.array || dict): List of crossval control IDs. If a dict, ID lists are keyed by set name.
        config: Set in src/config.py

    Returns:
        tuple: (caseGenotypeDict (dict), controlGenotypeDict (dict), missingCaseIDs (dict), missingControlIDs (dict), resolvedCaseIDs (dict), resolvedControlIDs (dict))
    """
    # cast genotypes as numeric, drop chromosome positions with missing values
    caseGenotypeFutures, controlGenotypeFutures = processAlleles.map(
        unmapped(filteredVCF.to_numpy()),
        unmapped(filteredVCF.columns.to_list()),
        genotypeIDs=(caseIDs, controlIDs),
        clinicalSampleIDmap=unmapped(clinicalSampleIDmap),
        config=unmapped(config),
    )

    caseGenotypeDict, missingCaseIDs, resolvedCaseIDs = caseGenotypeFutures.result()
    controlGenotypeDict, missingControlIDs, resolvedControlIDs = controlGenotypeFutures.result()

    return caseGenotypeDict, controlGenotypeDict, missingCaseIDs, missingControlIDs, resolvedCaseIDs, resolvedControlIDs



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
    """Load VCF-like and clinical datasets from file paths specified in the config.

    Args:
        config: Set in src/config.py

    Returns:
        tuple: (clinica metadata dataframe, list of external sample metadata dataframes, VCF dataframe, optional VCF dataframe with reference allele frequencies)
    """
    clinicalData = pd.read_excel(
        config["clinicalTable"]["path"], index_col=config["clinicalTable"]["idColumn"]
    )
    externalSamples = [
        pd.read_csv(externalTable["path"], sep=("\t" if ".tsv" in externalTable["path"] else ","), index_col=externalTable["idColumn"])
        for externalTable in config["externalTables"]["metadata"]
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
    print(weights)
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
    """Transform genotype data into feature embedding vectors suitable for training and testing a machine learning model. Numerical alleles are normalized to [0, 1] and missing values are imputed with the mean allele frequency. Cases and controls for training are balanced via reservoir sampling.

    Args:
        caseGenotypes (DataFrame): Case genotypes with numeric allele model
        controlGenotypes (DataFrame): Control genotypes with numeric allele model
        holdoutCaseGenotypes (dict[DataFrame]): Dict of holdout case genotypes with numeric allele model, mapped to set name
        holdoutControlGenotypes (dict[DataFrame]): Dict of holdout control genotypes with numeric allele model, mapped to set name
        sampleFrequencies (dict): Sample frequencies for each genotype ID to guide reservoir sampling
        verbose (bool, optional): _description_. Defaults to True.
        config (optional): _description_. Defaults to config.
        freqReferenceGenotypeData (optional): _description_. GenotypeData with desired background frequency profile. Defaults to None.

    Raises:
        e: ValueError if case/control genotypes are empty and/or contain invalid allele values (non-numeric).

    Returns:
        dict: embedding
    """
    minorIDs, balancedMajorIDs, excessMajorIDs, sampleFrequencies = prepareCaseControlSamples(
        caseGenotypes, controlGenotypes, sampleFrequencies, doBalance=True
    )
  
    allCrossValGenotypes = pd.concat(
        [
            caseGenotypes,
            controlGenotypes,
        ],
        axis=1,
    )
    caseIDs = caseGenotypes.columns
    controlIDs = controlGenotypes.columns

    excessIDs, crossValGenotypeIDs = [], []
    trainIDs = np.hstack([balancedMajorIDs, minorIDs])
    
    for label in tqdm(allCrossValGenotypes.columns, desc="Matching IDs", unit="ID"):
        for subsetType in ["excess", "toSample"]:
            idSet = (
                excessMajorIDs
                if subsetType == "excess"
                else trainIDs
            )
            for i, id in enumerate(idSet):
                if (id in label) or (label in id):
                    if subsetType == "toSample":
                        if label not in crossValGenotypeIDs:
                            crossValGenotypeIDs.append(label)
                    elif subsetType == "excess":
                        if label not in excessIDs:
                            excessIDs.append(label)
                    idSet = np.delete(idSet, i)
                    break
    
    allGenotypes = pd.concat([
            allCrossValGenotypes, 
            *[holdoutCaseSet for holdoutCaseSet in holdoutCaseGenotypes.values()], 
            *[holdoutControlSet for holdoutControlSet in holdoutControlGenotypes.values()]
        ], axis=1)
    
    if verbose:
        print(f"\n\n{len(crossValGenotypeIDs)} for training:\n{crossValGenotypeIDs}")
        print(f"\n\n{len(excessIDs)} are excess:\n{excessIDs}")
        if len(holdoutCaseGenotypes) > 0 or len(holdoutControlGenotypes) > 0:
            for setsType, holdoutSets in {"case": holdoutCaseGenotypes, "control": holdoutControlGenotypes}.items():
                for name in holdoutSets:
                    holdoutIDs = holdoutSets[name].columns.tolist()
                    print(f"--\n{len(holdoutIDs)} {setsType} in {name}")
        print(f"\nVariant count: {len(allGenotypes.index)}")

    # drop variants with missing values or invariant
    preCleanedVariantCounts = len(allGenotypes.index)
    allGenotypes = allGenotypes.dropna(
            how="any",
        ).loc[allGenotypes[crossValGenotypeIDs].std(axis=1) > 0.1]
    print(f"Dropped {preCleanedVariantCounts - len(allGenotypes.index)} variants with insufficient variance (stddev < 0.1) or missing values")

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
    if len(holdoutCaseGenotypes) > 0 or len(holdoutControlGenotypes) > 0:
        holdoutSamplesBySetName, holdoutIndexBySetName, holdoutLabelsBySetName = {}, {}, {}
        # first create case embeddings
        for setName, sampleSet in holdoutCaseGenotypes.items():
            # match dropped crossval allele frequencies  
            sampleSet = sampleSet.loc[allGenotypes.index]
            # transpose so that samples are rows, variants are columns
            holdoutSamplesBySetName[setName] = scaler.fit_transform(sampleSet).transpose()
            holdoutIndexBySetName[setName] = np.array(sampleSet.columns.tolist())
            holdoutLabelsBySetName[setName] = np.array([1] * len(sampleSet.columns.tolist()))
            if setName in holdoutControlGenotypes:
                holdoutControlGenotypes[setName] = holdoutControlGenotypes[setName].loc[allGenotypes.index]
                # append control samples to case samples
                holdoutSamplesBySetName[setName] = np.concatenate(
                    [
                        holdoutSamplesBySetName[setName],
                        scaler.fit_transform(holdoutControlGenotypes[setName]).transpose(),
                    ]
                )
                holdoutIndexBySetName[setName] = np.concatenate(
                    [
                        holdoutIndexBySetName[setName],
                        np.array(holdoutControlGenotypes[setName].columns.tolist()),
                    ]
                )
                holdoutLabelsBySetName[setName] = np.concatenate(
                    [
                        holdoutLabelsBySetName[setName],
                        np.array([0] * len(holdoutControlGenotypes[setName].columns.tolist())),
                    ]
                )
        # create control embeddings
        for setName, sampleSet in holdoutControlGenotypes.items():
            sampleSet = sampleSet.loc[allGenotypes.index]
            if setName in holdoutSamplesBySetName:
                continue
            holdoutSamplesBySetName[setName] = scaler.fit_transform(sampleSet).transpose()
            holdoutIndexBySetName[setName] = np.array(sampleSet.columns.tolist())
            holdoutLabelsBySetName[setName] = np.array([0] * len(sampleSet.columns.tolist()))
        try:
            embedding = {
                **embedding,
                **{
                    "holdoutSampleIndex": holdoutIndexBySetName,
                    "holdoutLabels": holdoutLabelsBySetName,
                    "holdoutSamples": holdoutSamplesBySetName,
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


@task()
def integrateExternalSampleIDs(filteredExternalSamples, config, caseIDs, controlIDs):
    """Parse sample IDs from external metadata tables & split into crossval or holdout case/control sets. Holdout sets are matched by set name, defined in config.  

    Args:
        filteredExternalSamples (Dataframe[]): List of filtered external metadata tables
        config: Set in src/config.py 
        caseIDs (np.array): Crossval case IDs to append external IDs to.
        controlIDs (np.array): Crossval control IDs to append external IDs to.

    Returns:
        tuple: (crossvalCaseIDs (np.array), crossvalControlIDs (np.array), holdoutCaseIDs (dict), holdoutControlIDs (dict))
    """
    holdoutCaseIDs = {holdoutSetName: np.array([]) for holdoutSetName in config['externalTables']['holdoutSetNames']}
    holdoutControlIDs = {holdoutSetName: np.array([]) for holdoutSetName in config['externalTables']['holdoutSetNames']}
    for i, externalTableMetadata in enumerate(config["externalTables"]["metadata"]):
        label = externalTableMetadata["label"]
        if externalTableMetadata["setType"] in config["externalTables"]["holdoutSetNames"]:
            if label == config["clinicalTable"]["caseAlias"]:
                holdoutCaseIDs[externalTableMetadata["setType"]] = np.append(
                    holdoutCaseIDs[externalTableMetadata["setType"]], [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
            elif label == config["clinicalTable"]["controlAlias"]:
                holdoutControlIDs[externalTableMetadata["setType"]] = np.append(
                    holdoutControlIDs[externalTableMetadata["setType"]], [id for id in filteredExternalSamples[i].index.to_numpy()]
                )
        elif externalTableMetadata["setType"] == "crossval":
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
        f"dropped {len(clinicalData) - len(filteredClinicalData)} samples from clinical data"
    )
    
    filteredExternalSamples = []
    for i, externalTableMetadata in enumerate(config["externalTables"]["metadata"]):
        filteredExternalSamples.append(
            filterTable(externalSamples[i], externalTableMetadata["filters"])
        )
        print(
            f"dropped {len(externalSamples[i]) - len(filteredExternalSamples[i])} samples from external data {externalTableMetadata['path']}"
        )
        
    filteredVCF = filterTable(annotatedVCF, config["vcfLike"]["filters"])
    print(f"filtered {annotatedVCF.shape[0] - filteredVCF.shape[0]} variants from VCF")
    if referenceVCF is not None:
        filteredReferenceVCF = filterTable(referenceVCF, config["vcfLike"]["filters"])
        print(
            f"dropped {referenceVCF.shape[0] - filteredReferenceVCF.shape[0]} variants from reference VCF"
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

    caseIDs, controlIDs, holdoutDictCaseIDs, holdoutDictControlIDs = integrateExternalSampleIDs(filteredExternalSamples, config, caseIDs, controlIDs)

    caseGenotypeDict, controlGenotypeDict, missingCaseIDs, missingControlIDs, resolvedCaseIDs, resolvedControlIDs = processGenotypes(
        filteredVCF, clinicalSampleIDmap, caseIDs, controlIDs, config
    )
    
    if config['sampling']['shuffleLabels']:
        combinedIDs = list(resolvedCaseIDs['crossval']) + list(resolvedControlIDs['crossval'])
        np.random.shuffle(combinedIDs)
        caseIDs = combinedIDs[:len(resolvedCaseIDs['crossval'])]
        controlIDs = combinedIDs[len(resolvedCaseIDs['crossval']):]
        caseGenotypeDict, controlGenotypeDict, missingCaseIDs, missingControlIDs, resolvedCaseIDs, resolvedControlIDs = processGenotypes(
            filteredVCF, clinicalSampleIDmap, caseIDs, controlIDs, config
        )
        print(f"shuffled case ID len: {len(caseGenotypeDict['crossval'].keys())}")
        print(f"shuffled control ID len: {len(controlGenotypeDict['crossval'].keys())}")
        
            
    # if at least one holdout set exists
    if len(holdoutDictCaseIDs) > 0:
        holdoutCaseGenotypeDict, holdoutControlGenotypeDict, missingHoldoutCaseIDs, missingHoldoutControlIDs, resolvedHoldoutCaseIDs, resolvedHoldoutControlIDs = processGenotypes(
            filteredVCF, clinicalSampleIDmap, holdoutDictCaseIDs, holdoutDictControlIDs, config
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
        # ensure training data for reference & input share same resolved IDs
        resolvedCaseIDs = list(set(resolvedCaseIDs["crossval"]).intersection(set(resolvedFreqReferenceCaseIDs["crossval"])))
        resolvedControlIDs = list(set(resolvedControlIDs["crossval"]).intersection(set(resolvedFreqReferenceControlIDs["crossval"])))
    else:
        freqReferenceCaseGenotypeDict = {}
        freqReferenceControlGenotypeDict = {}
        missingFreqReferenceCaseIDs, missingFreqReferenceControlIDs = None, None
    
    for alias, (IDs, genotypeDict) in {
        "caseAlias": (missingCaseIDs["crossval"], caseGenotypeDict["crossval"]),
        "controlAlias": (missingControlIDs["crossval"], controlGenotypeDict["crossval"]),
        "holdout cases": (missingHoldoutCaseIDs, holdoutCaseGenotypeDict),
        "holdout controls": (missingHoldoutControlIDs, holdoutControlGenotypeDict),
        #"caseAlias": (missingFreqReferenceCaseIDs["crossval"], freqReferenceCaseGenotypeDict),
        #"controlAlias": (missingFreqReferenceControlIDs["crossval"], freqReferenceControlGenotypeDict),
    }.items():
        if len(genotypeDict) == 0: continue
        if len(IDs) > 0:
            if "holdout" not in alias:
                print(
                    f"\nmissing {len(IDs)} {config['clinicalTable'][alias]} IDs:\n {IDs}"
                )
            elif "holdout" in alias:
                for holdoutSetName, idList in IDs.items():
                    print(f"\n{holdoutSetName} missing {len(idList)} {alias} IDs:\n {idList}")

    resolvedIDs = np.hstack([list(caseGenotypeDict["crossval"].keys()), list(controlGenotypeDict["crossval"].keys())])
    allCrossValGenotypes = createGenotypeDataframe({**caseGenotypeDict["crossval"], **controlGenotypeDict["crossval"]}, filteredVCF).dropna().astype(int)
    if referenceVCF is not None:
        freqReferenceAllCrossValGenotypes = createGenotypeDataframe({**freqReferenceCaseGenotypeDict["crossval"], **freqReferenceControlGenotypeDict["crossval"]}, filteredReferenceVCF).dropna().astype(int)
    
    if isinstance(allCrossValGenotypes.index, pd.MultiIndex):
        # Manually check for nulls in each level of the MultiIndex
        non_null_indices = ~allCrossValGenotypes.index.to_frame().isnull().any(axis=1)
        allCrossValGenotypes = allCrossValGenotypes[non_null_indices]
    else:
        # For a standard index, retain rows with non-null indices
        allCrossValGenotypes = allCrossValGenotypes[allCrossValGenotypes.index.notnull()]
        
    # Calculate the allele frequencies
    allele_frequencies = (
        allCrossValGenotypes.gt(0).sum(axis=1) / len(resolvedIDs)
    ).loc[
        lambda x: x.between(config["vcfLike"]["minAlleleFrequency"], config["vcfLike"]["maxAlleleFrequency"])
    ]

    # Filter the genotypes based on frequency criteria
    if referenceVCF is not None:
        reference_allele_frequencies = (
                freqReferenceAllCrossValGenotypes.gt(0).sum(axis=1) / len(resolvedIDs)
            ).loc[
            lambda x: x.between(config["vcfLike"]["minAlleleFrequency"], config["vcfLike"]["maxAlleleFrequency"])
        ]
        matched_allele_frequencies = findClosestFrequencyVariant(reference_allele_frequencies, allele_frequencies)
        frequencyFilteredGenotypes = allCrossValGenotypes.loc[
            matched_allele_frequencies
        ].sort_index()
        referenceFrequencyFilteredGenotypes = freqReferenceAllCrossValGenotypes.loc[reference_allele_frequencies].sort_index()
        print(f"Matched {len(frequencyFilteredGenotypes)} alleles to reference VCF {config['vcfLike']['frequencyMatchReference']}")
    else:
        frequencyFilteredGenotypes = allCrossValGenotypes.loc[
            allele_frequencies.index
        ]
        
    print(
        f"Filtered {len(filteredVCF) - len(frequencyFilteredGenotypes)} alleles with frequency below {'{:.3%}'.format(config['vcfLike']['minAlleleFrequency'])} or above {'{:.3%}'.format(config['vcfLike']['maxAlleleFrequency'])}"
    )
    print(f"Kept {len(frequencyFilteredGenotypes)} alleles")
    
    caseGenotypesDataframe = createGenotypeDataframe(caseGenotypeDict["crossval"], filteredVCF).loc[frequencyFilteredGenotypes.index]
    controlGenotypesDataframe = createGenotypeDataframe(
        controlGenotypeDict["crossval"], filteredVCF
    ).loc[frequencyFilteredGenotypes.index]

    holdoutCaseGenotypesDataframe = (
        createGenotypeDataframe({id:genotype for holdoutSet in holdoutCaseGenotypeDict.values() for id, genotype in holdoutSet.items()}, filteredVCF).loc[frequencyFilteredGenotypes.index]
        if resolvedHoldoutCaseIDs
        else pd.DataFrame(index=frequencyFilteredGenotypes.index)
    )
    holdoutControlGenotypesDataframe = (
        createGenotypeDataframe({id:genotype for holdoutSet in holdoutControlGenotypeDict.values() for id, genotype in holdoutSet.items()}, filteredVCF).loc[frequencyFilteredGenotypes.index]
        if resolvedHoldoutControlIDs
        else pd.DataFrame(index=frequencyFilteredGenotypes.index)
    )
    
    if referenceVCF is not None:
        referenceCaseGenotypesDataframe = createGenotypeDataframe(freqReferenceCaseGenotypeDict["crossval"], filteredReferenceVCF).loc[referenceFrequencyFilteredGenotypes.index]
        referenceControlGenotypesDataframe = createGenotypeDataframe(
            freqReferenceControlGenotypeDict["crossval"], filteredReferenceVCF
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

    caseGenotypes = Genotype(caseGenotypesDataframe, resolvedCaseIDs["crossval"], "Case")
    controlGenotypes = Genotype(
        controlGenotypesDataframe, resolvedControlIDs["crossval"], "Control"
    )
    
    holdoutCaseGenotypes = {
        setName: Genotype(
            holdoutCaseGenotypesDataframe[list(holdoutCaseGenotypeDict[setName].keys())], resolvedHoldoutCaseIDs[setName], setName)
        for setName in holdoutCaseGenotypeDict
    }
    holdoutControlGenotypes = {
        setName: Genotype(
            holdoutControlGenotypesDataframe[list(holdoutControlGenotypeDict[setName].keys())],
            resolvedHoldoutControlIDs[setName],
            setName)
        for setName in holdoutControlGenotypeDict
    }
    
    if referenceVCF is not None:
        referenceCaseGenotypes = Genotype(referenceCaseGenotypesDataframe, resolvedFreqReferenceCaseIDs["crossval"], "Reference Case")
        referenceControlGenotypes = Genotype(referenceControlGenotypesDataframe, resolvedFreqReferenceControlIDs["crossval"], "Reference Control")

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

    print(f"\n{len(resolvedCaseIDs['crossval'])} cases")
    print(f"\n{len(resolvedControlIDs['crossval'])} controls\n--")
    if resolvedHoldoutCaseIDs:
        for holdoutSetName, holdoutSetIDs in resolvedHoldoutCaseIDs.items():
            print(f"\n{len(holdoutSetIDs)} {holdoutSetName} cases")
    if resolvedHoldoutControlIDs:
        for holdoutSetName, holdoutSetIDs in resolvedHoldoutControlIDs.items():
            print(f"\n{len(holdoutSetIDs)} {holdoutSetName} controls")
    
    saveSampleEmbeddings(genotypeData, config)

    return genotypeData, frequencyReferenceGenotypeData, filteredClinicalData

def saveSampleEmbeddings(genotypeData: GenotypeData, config=config):
    runPath = f"projects/{config['tracking']['project']}"
    os.makedirs(runPath, exist_ok=True)
    for attr in ["case", "control", "holdout_case", "holdout_control"]:
        if "holdout" not in attr:
            currentGenotypeData = getattr(genotypeData, attr).genotype
            currentGenotypeData.to_csv(f"{runPath}/embedding_{attr}.csv")
        else:
            holdoutData = getattr(genotypeData, attr)
            for setName in holdoutData:
                currentGenotypeData = holdoutData[setName].genotype
                if len(currentGenotypeData) == 0:
                    continue
                os.makedirs(f"{runPath}/holdout/{setName}", exist_ok=True)
                currentGenotypeData.to_csv(f"{runPath}/holdout/{setName}/embedding_{attr}__{setName}.csv")


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
