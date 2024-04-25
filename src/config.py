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
        "minAlleleFrequency": 0.0025,  # filter out variants with allele frequency less than this
        "maxAlleleFrequency": 1.00,  # filter out variants with allele frequency greater than this
        "maxVariants": None, # set max number of variants for control; set to None to disable
        "frequencyMatchReference": None, # reference VCF-like to frequency-match alleles; must have same indices as vcfLike
        # 'alleleModel': ['dominant', 'recessive', 'overDominant'],  # biallelic allele models to test on gene sets
        "filters": {},
    },  # TODO handle genotypes from related individuals
    "geneSets": {},  # TODO gene sets
    "tracking": {
        "name": "NUP variants (rare-binned, rsID only)\nTrained on: >=85% accurate AnswerALS cases & non-neurological controls (Caucasian)",  # name of the experiment
        "entity": "ejmockler",
        "project": "NUPs60->=85%accurateCasesLR-aals-rsID-rareBinned-0.0025MAF",
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
        "caseLabels": [],  # "ALS Spectrum MND"
        "controlAlias": "control",
        "caseAlias": "case",
        "filters": "pct_european>=0.85",  # filter out nonhomogenous samples with less than 85% European ancestry
    },
    "externalTables": {
        "holdoutSetNames": [
            "AnswerALS Cases vs. Controls (Ethnically-Variable)",
            "Other Neurological Cases vs. Controls (Ethnically-Variable)",
            # "MinE Cases vs. MinE Controls",
            # "MinE Cases vs. Ethnically-Variable Controls",
        ],
        "metadata": [
            {
                "setType": "crossval", 
                "path": "../adhoc analysis/>=85%accurateCases_LogisticRegression_NUPs60-aals-rsID-rareBinned-0.0025MAF.tsv", 
                "label": "case", 
                "idColumn": "id", 
                "filters":  "",
            },
            
            {
                "setType": "crossval", 
                "path": "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv", 
                "label": "control", 
                "idColumn": "Sample name", 
                "filters":  "`Superpopulation code`=='EUR'",
            },
            
            {
                "setType": "AnswerALS Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv", 
                "label": "control", 
                "idColumn": "Sample name", 
                "filters":  "`Superpopulation code`!='EUR'",
            },
            {
                "setType": "AnswerALS Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/ACWM_ethnicallyVariable.tsv", 
                "label": "case", 
                "idColumn": "ExternalSampleId", 
                "filters": "`Subject Group`.str.contains('ALS Spectrum MND') & `pct_european`<0.85",
            },
            {
                "setType": "AnswerALS Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/ACWM_ethnicallyVariable.tsv", 
                "label": "control", 
                "idColumn": "ExternalSampleId", 
                "filters":  "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            },
            
            {
                "setType": "Other Neurological Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/ACWM_otherNeurological.tsv", 
                "label": "case", 
                "idColumn": "ExternalSampleId", 
                "filters":  "",
            },
            {
                "setType": "Other Neurological Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv", 
                "label": "control", 
                "idColumn": "Sample name", 
                "filters":  "`Superpopulation code`!='EUR'",
            },
            {
                "setType": "Other Neurological Cases vs. Controls (Ethnically-Variable)", 
                "path": "../adhoc analysis/ACWM_ethnicallyVariable.tsv", 
                "label": "control", 
                "idColumn": "ExternalSampleId", 
                "filters":  "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            },
            
            # {
            #     "setType": "MinE Cases vs. MinE Controls",
            #     "path": "../adhoc analysis/mine_case_ids.tsv",
            #     "label": "case",
            #     "idColumn": "id",
            #     "filters":""
            # },
            # {
            #     "setType": "MinE Cases vs. MinE Controls",
            #     "path": "../adhoc analysis/mine_control_ids.tsv",
            #     "label": "control",
            #     "idColumn": "id",
            #     "filters":""
            # },
            
            # {
            #     "setType": "MinE Cases vs. Ethnically-Variable Controls",
            #     "path": "../adhoc analysis/mine_case_ids.tsv",
            #     "label": "case",
            #     "idColumn": "id",
            #     "filters":""
            # },
            # {
            #     "setType": "MinE Cases vs. Ethnically-Variable Controls", 
            #     "path": "../adhoc analysis/igsr-1000 genomes phase 3 release.tsv", 
            #     "label": "control", 
            #     "idColumn": "Sample name", 
            #     "filters":  "`Superpopulation code`!='EUR'",
            # },
            # {
            #     "setType": "MinE Cases vs. Ethnically-Variable Controls", 
            #     "path": "../adhoc analysis/ACWM_ethnicallyVariable.tsv", 
            #     "label": "control", 
            #     "idColumn": "ExternalSampleId", 
            #     "filters":  "`Subject Group`=='Non-Neurological Control' & `pct_european`<0.85",
            # },
            
            
        ],  # external sample tables
    },
    "sampling": {
        "bootstrapIterations": 60,
        "crossValIterations": 10,  # number of validations per bootstrap iteration
        "lastIteration": 0,
        "sequesteredIDs": [],  # crossval IDs to withhold from training
    },
    "model": {
        "hyperparameterOptimization": True,
        "calculateShapelyExplanations": False,
    },
}