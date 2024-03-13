config = {
    "vcfLike": {
        "path": "../adhoc analysis/NUP_aals_mine_genotypes.tsv",  # variant call table with annotations
        "sheet": "all cases vs all controls",  # sheet name if Excel spreadsheet
        "indexColumn": [
            "chrom",
            "position",
            "rsID",
            "Gene",
        ],  # header that indexes variants (list if multiple columns)
        "geneMultiIndexLevel": 3,  # level of gene index in indexColumn
        "aggregateGenesBy": None,  # aggregate variants by mean or sum across genes. Set to None to disable.
        "compoundSampleIdDelimiter": "__",  # delimiter for compound sample IDs in column names
        "compoundSampleIdStartIndex": 1,  # index of genotype ID in compound sample ID
        "compoundSampleMetaIdStartIndex": 1,  # index of clinical ID in compound sample ID
        "binarize": False,  # binarize variants to 0/1, or sum to weigh allele frequency,
        "zygosity": True,  # bin variants by zygosity (homozygous + rare heterozygous, heterozygous)
        "minAlleleFrequency": 0.005,  # filter out variants with allele frequency less than this
        "maxAlleleFrequency": 1.00,  # filter out variants with allele frequency greater than this
        "maxVariants": None, # set max number of variants for control; set to None to disable
        "frequencyMatchReference": None, # reference VCF-like to frequency-match alleles; must have same indices as vcfLike
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "NUP variants (rare-binned, rsID only)",  # name of the experiment
        "entity": "ejmockler",
        "project": "NUPs15-rsID-rareBinned-0.005MAF",
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
        "holdoutSetName": "AALS >=85% Accurate Caucasian Cases and Ethnically-Variable Controls",
        "path": [
            "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
            "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
            #"../adhoc analysis/>=85%accurateCases_LogisticRegression_NUPs60-rsID-rareBinned-0.005MAF.tsv",
            "../adhoc analysis/ACWM_ethnicallyVariable.tsv",
            "../adhoc analysis/ACWM_ethnicallyVariable.tsv",
            #"../adhoc analysis/aals_case_ids.tsv",
            #"../adhoc analysis/mine_control_ids.tsv",
            #"../adhoc analysis/>=85%accurateCases_highDraw_LogisticRegression_NUPs100-projMine-rsID-rareBinned-0.005MAF.tsv"
        ],  # external sample table
        "label": [
            "control",
            "control",
            "case",
            "control",
            #"case", 
            #"control",
            #"case"
        ],  # case | control
        "setType": [
            "crossval",
            "holdout",
            "holdout",
            "holdout",
            #"holdout",
            #"crossval",
            #"crossval"
        ],
        "idColumn": [
            "Sample name",
            "Sample name",
            "id",
            "id",
            #"id",
            #"id",
            #"id"
        ],  # sample ID header
        "filters": [
            "`Superpopulation code`=='EUR'",
            "`Superpopulation code`!='EUR'",
            "`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
            "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            #"",
            "",
            ""
        ],
    },
    "sampling": {
        "bootstrapIterations": 15,
        "crossValIterations": 10,  # number of validations per bootstrap iteration
        "lastIteration": 0,
        "sequesteredIDs": [],
    },
    "model": {
        "hyperparameterOptimization": True,
        "calculateShapelyExplanations": False,
    },
}