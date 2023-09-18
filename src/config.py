config = {
    "vcfLike": {
        "path": "../adhoc analysis/exampleVCFlike.xlsx",  # variant call table with annotations
        "sheet": "Sheet1",  # sheet name if Excel spreadsheet
        "indexColumn": [
            "chrom",
            "position",
            "Gene",
        ],  # header that indexes variants (set as list with multiple columns)
        "compoundSampleIdDelimiter": "__",  # delimiter for compound sample IDs in column names
        "compoundSampleIdStartIndex": 1,  # index of genotype ID in compound sample ID
        "compoundSampleMetaIdStartIndex": 1,  # index of clinical ID in compound sample ID
        "binarize": True,  # binarize variants to 0/1, or sum to weigh allele frequency
        "minAlleleFrequency": 0.01,  # filter out variants with allele frequency less than this
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "ALSoD genes, female individuals",  # name of the experiment
        "entity": "ejmockler",
        "project": "ALS-ALSoD-females-1MAF",
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
        "filters": "pct_european>=0.85 & Sex=='Female'",  # filter out nonhomogenous samples with less than 85% European ancestry
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
            "ExternalSubjectId",
            "ExternalSubjectId",
            "Sample name",
        ],  # sample ID header
        "filters": [
            "`Superpopulation code`=='EUR' & `Sex`=='female'",
            # "`testLabel`==1",
            "`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85 & `Sex`=='Female'",
            "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85 & `Sex`=='Female'",
            "`Superpopulation code`!='EUR' & `Sex`=='female'",
        ],
    },
    "sampling": {
        "bootstrapIterations": 3,
        "crossValIterations": 2,  # number of validations per bootstrap iteration
        "holdoutSplit": 0.1,
        "lastIteration": 0,
        "sequesteredIDs": [],
    },
    "model": {
        "hyperparameterOptimization": True,
        "calculateShapelyExplanations": True,
    },
}
