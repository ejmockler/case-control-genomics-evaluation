from pytest import fixture


@fixture
def config():
    return {
        "vcfLike": {
            "path": "src/test/files/exampleVCFlike.xlsx",  # variant call table with annotations
            "sheet": "Sheet1",  # sheet name if Excel spreadsheet
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
            "name": "Test",  # name of the experiment
            "project": "Test",
            "plotAllSampleImportances": True,  # if calculating Shapely explanations, plot each sample in Neptune
            "remote": False,  # if True, log to MLflow
        },
        "clinicalTable": {
            "path": "src/test/files/exampleClinicalData.xlsx",  # clinical data as Excel spreadsheet
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
                "src/test/files/exampleControlMetadata.tsv",
                "src/test/files/exampleExternalMetadata.tsv",
                "src/test/files/exampleExternalMetadata.tsv",
                "src/test/files/exampleControlMetadata.tsv",
            ],  # external sample table
            "label": [
                "control",
                "case",
                "control",
                "control",
            ],  # case | control
            "setType": [
                "crossval",
                "holdout",
                "holdout",
                "holdout",
            ],
            "idColumn": [
                "Sample name",
                "ExternalSubjectId",
                "ExternalSubjectId",
                "Sample name",
            ],  # sample ID header
            "filters": [
                "`Superpopulation code`=='EUR'",
                "`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
                "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
                "`Superpopulation code`!='EUR'",
            ],
        },
        "sampling": {
            "bootstrapIterations": 2,
            "crossValIterations": 5,  # number of validations per bootstrap iteration
            "holdoutSplit": 0.1,
            "lastIteration": 0,
            "sequesteredIDs": [],
        },
        "model": {
            "hyperparameterOptimization": True,
            "calculateShapelyExplanations": True,
        },
    }
