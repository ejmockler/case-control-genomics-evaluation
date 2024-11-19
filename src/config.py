from typing import List, Optional, Dict
from pydantic import BaseModel


class VCFConfig(BaseModel):
    path: str
    binarize: bool = False
    zygosity: bool = True
    min_allele_frequency: float = 0.0025
    max_allele_frequency: float = 1.0
    max_variants: Optional[int] = None
    min_variance: float = 0.01
    filter: str = ""

class GTFConfig(BaseModel):
    path: str
    gene_name_field: str = "gene_name"
    filter: str = ""

class GMTConfig(BaseModel):
    # requires GTF annotations with a defined gene_name field
    path: str
    filter: str = ""

class TrackingConfig(BaseModel):
    name: str
    entity: str
    experiment_name: str
    plot_all_sample_importances: bool = False
    tracking_uri: str = "http://localhost:5000"
    
class TableMetadata(BaseModel):
    name: str
    path: str
    label: str
    id_column: str
    filter: str
    # Change strata_columns to a mapping from standard strata to table-specific column names
    strata_mapping: Optional[Dict[str, str]] = None  # e.g., {"sex": "Sex Call", "age": "Age"}

class SampleTableConfig(BaseModel):
    tables: List[TableMetadata]

class SamplingConfig(BaseModel):
    outer_bootstrap_iterations: int = 10
    bootstrap_iterations: int = 60
    crossval_folds: int = 10
    test_size: float = 0.2
    sequestered_ids: List[str] = []
    shuffle_labels: bool = False
    feature_credible_interval: float = 0.95
    # Specify which strata to use for sampling
    strata: Optional[List[str]] = ["sex"]  # Allowed values: "sex", "age"
    random_state: int = 42

class ModelConfig(BaseModel):
    hyperparameter_optimization: bool = True
    calculate_shapely_explanations: bool = False

class Config(BaseModel):
    vcf: VCFConfig
    gtf: GTFConfig
    gmt: Optional[GMTConfig] = None  # Add GMTConfig as optional
    tracking: TrackingConfig
    crossval_tables: SampleTableConfig
    holdout_tables: SampleTableConfig
    sampling: SamplingConfig
    model: ModelConfig

config = Config(
    vcf=VCFConfig(
        path="../adhoc analysis/mock.vcf.gz",
        binarize=False,
        zygosity=True,
        min_allele_frequency=0.005,
        max_allele_frequency=1.0,
        max_variants=None,
        filter="",
    ),
    gtf=GTFConfig(
        path="../adhoc analysis/gencode.v46.chr_patch_hapl_scaff.annotation.gtf.gz",
        filter="(ht.transcript_type == 'protein_coding') | (ht.transcript_type == 'protein_coding_LoF')",
    ),
    # gmt=GMTConfig(
    #     path="../adhoc analysis/kiaa1217_only.gmt",
    #     filter="",
    # ),
    tracking=TrackingConfig(
        name="KIAA1217 variants, MAF>=0.5% (zygosity-binned)\nTrained on: AnswerALS cases & non-neurological controls (Caucasian)",
        entity="ejmockler",
        experiment_name="dbg",
        plot_all_sample_importances=False,  
        tracking_uri="http://127.0.0.1:5001/",
    ),
    crossval_tables=SampleTableConfig(
        tables=[
            TableMetadata(
                name="AnswerALS cases, EUR",
                path="../adhoc analysis/ALS Consortium WGS Metadata 03112022.xlsx",
                label="case",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='ALS Spectrum MND' & `pct_european`>=0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
            TableMetadata(
                name="AnswerALS non-neurological controls, EUR",
                path="../adhoc analysis/ALS Consortium WGS Metadata 03112022.xlsx",
                label="control",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='Non-Neurological Control' & `pct_european`>=0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
            TableMetadata(
                name="1000 Genomes EUR",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filter="`Superpopulation code`=='EUR'",
                strata_mapping={
                    "sex": "Sex",
                    # "age": "Age"  # Include if available
                },
            ),
        ]
    ),
    holdout_tables=SampleTableConfig(
        # TODO option to define comparison tables other than crossval
        tables=[
            TableMetadata(
                name="1000 Genomes ethnically-variable, non-EUR",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filter="`Superpopulation code`!='EUR'",
                strata_mapping={
                    "sex": "Sex",
                    # "age": "Age"  # Include if available
                },
            ),
            TableMetadata(
                name="AnswerALS cases, ethnically-variable, non-EUR",
                path="../adhoc analysis/ALS Consortium WGS Metadata 03112022.xlsx",
                label="case",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
        ]
    ),
    sampling=SamplingConfig(
        outer_bootstrap_iterations=2,
        bootstrap_iterations=2,
        crossval_folds=10,
        feature_credible_interval=0.5,
        test_size=0.2,
        sequestered_ids=[],
        shuffle_labels=False,
        strata=["sex"],  # Define which strata to use,
        random_state=42
    ),
    model=ModelConfig(
        hyperparameter_optimization=True,
        calculate_shapely_explanations=False,
    ),
)