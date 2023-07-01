from env import neptune_api_token


config = {
    "vcfLike": {
        "path": "../notebook/Variant_report_NUPs_fixed_2022-03-28.xlsx",  # variant call table with annotations
        "sheet": "all cases vs all controls",  # sheet name if Excel spreadsheet
        "indexColumn": [
            "chrom",
            "position",
            "Gene",
        ],  # header that indexes variants (set as list with multiple columns)
        "compoundSampleIdDelimiter": "__",  # delimiter for compound sample IDs in column names
        "compoundSampleIdStartIndex": 1,  # index of first sample ID in compound sample ID
        "binarize": True,  # binarize variants to 0/1, or sum to weigh allele frequency
        "minAlleleFrequency": 0.05,  # filter out variants with allele frequency less than this
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "Nucleoporin genes",  # name of the experiment
        "entity": "ejmockler",
        "project": "ALS-NUPS-60-dbg",
        "plotAllSampleImportances": True,  # if calculating Shapely explanations, plot each sample in Neptune
        "token": neptune_api_token,
        "remote": False,  # if True, log to Neptune
    },
    "clinicalTable": {
        "path": "../notebook/ACWM.xlsx",  # clinical data as Excel spreadsheet
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
            "../notebook/igsr-1000 genomes phase 3 release.tsv",
            # "../notebook/ALS-NUPS-2000__accurateSamples_>=97.5%.csv",
            "../notebook/ACWM_ethnicallyVariable.tsv",
            "../notebook/ACWM_ethnicallyVariable.tsv",
            "../notebook/igsr-1000 genomes phase 3 release.tsv",
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
            "`Superpopulation code`=='EUR' & `Population name`!='Finnish'",  # remove finnish samples due to unusual homogeneity (verify w/ PCA)
            # "`testLabel`==1",
            "`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
            "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            "`Superpopulation code`!='EUR' & `Population name`!='Finnish'",
        ],
    },
    "sampling": {
        "bootstrapIterations": 2,
        "crossValIterations": 3,  # number of validations per bootstrap iteration
        "holdoutSplit": 0.1,
        "lastIteration": 0,
        "sequesteredIDs": [],
    },
    "model": {
        "hyperparameterOptimization": True,
        "calculateShapelyExplanations": False,
    },
}
