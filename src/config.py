config = {
    "vcfLike": {
        "path": "../adhoc analysis/cardiacGenotypes.csv",  # variant call table with annotations
        "sheet": "all cases vs all controls",  # sheet name if Excel spreadsheet
        "indexColumn": [
            "chrom",
            "position",
            "Gene",
        ],  # header that indexes variants (set as list with multiple columns)
        "geneMultiIndexLevel": 2,  # level of gene index in indexColumn
        "aggregateGenesBy": None,  # aggregate variants by mean or sum across genes. Set to None to disable.
        "compoundSampleIdDelimiter": "__",  # delimiter for compound sample IDs in column names
        "compoundSampleIdStartIndex": 1,  # index of genotype ID in compound sample ID
        "compoundSampleMetaIdStartIndex": 1,  # index of clinical ID in compound sample ID
        "binarize": True,  # binarize variants to 0/1, or sum to weigh allele frequency,
        "zygosity": False,  # bin variants by zygosity (homozygous + rare heterozygous, heterozygous)
        "minAlleleFrequency": 0.005,  # filter out variants with allele frequency less than this
        "maxAlleleFrequency": 1.00,  # filter out variants with allele frequency greater than this
        "maxVariants": None, # set max number of variants for control; set to None to disable
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "Cardiac variants (dominant model), Caucasian individuals",  # name of the experiment
        "entity": "ejmockler",
        "project": "cardiac60-zygosityRareBinned-0.005MAF",
        "plotAllSampleImportances": True,  # if calculating Shapely explanations, plot each sample in Neptune
        "remote": False,  # if True, log to Neptune
    },
    "clinicalTable": {
        "path": "../adhoc analysis/ACWM.xlsx",  # clinical data as Excel spreadsheet
        "idColumn": "ExternalSampleId",  # genotype ID header
        "subjectIdColumn": "ExternalSubjectId",  # unique ID for each patient
        "labelColumn": "Subject Group",  # header that has case/control labels
        "controlLabels": [
            "Non-Neurological Control"
        ],  # these labels include external sample IDs (like 1000 Genomes)
        "caseLabels": ["ALS Spectrum MND"],  # "ALS Spectrum MND"
        "controlAlias": "control",
        "caseAlias": "case",
        "filters": "pct_european>=0.85",  # filter out nonhomogenous samples with less than 85% European ancestry
    },
    "externalTables": {
        "path": [
            "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
            # "../adhoc analysis/ALS-NUPS-2000__accurateSamples_>=97.5%.csv",
            "../adhoc analysis/ACWM_ethnicallyVariable.tsv",
            "../adhoc analysis/ACWM_ethnicallyVariable.tsv",
            "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
        ],  # external sample table
        "label": [
            "control",
            # "case",
            "case",
            "control",
            "control",
        ],  # case | control
        "setType": [
            "crossval",
            # "crossval",
            "holdout",
            "holdout",
            "holdout",
        ],
        "idColumn": [
            "Sample name",
            # "id",
            "ExternalSampleId",
            "ExternalSampleId",
            "Sample name",
        ],  # sample ID header
        "filters": [
            # TODO add other motor neuron disease labels
            "`Superpopulation code`=='EUR'",
            # "`testLabel`==1",
            "`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
            "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            "`Superpopulation code`!='EUR'",
        ],
    },
    "sampling": {
        "bootstrapIterations": 60,
        "crossValIterations": 10,  # number of validations per bootstrap iteration
        "lastIteration": 0,
        "sequesteredIDs": [],
    },
    "model": {
        "hyperparameterOptimization": True,
        "calculateShapelyExplanations": False,
    },
}