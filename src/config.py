from typing import List, Optional, Dict, Union
from pydantic import BaseModel


class VcfLikeConfig(BaseModel):
    path: str
    sheet: Optional[str] = None
    index_column: List[str]
    gene_multi_index_level: int
    aggregate_genes_by: Optional[str] = None
    compound_sample_id_delimiter: str
    compound_sample_id_start_index: int
    compound_sample_meta_id_start_index: int
    binarize: bool = False
    zygosity: bool = True
    min_allele_frequency: float = 0.0025
    max_allele_frequency: float = 1.0
    max_variants: Optional[int] = None
    frequency_match_reference: Optional[str] = None
    filters: Dict = {}


class TrackingConfig(BaseModel):
    name: str
    entity: str
    project: str
    plot_all_sample_importances: bool = True
    remote: bool = False


class ClinicalTableConfig(BaseModel):
    path: str
    id_column: str
    subject_id_column: str
    label_column: str
    control_labels: List[str]
    case_labels: List[str]
    control_alias: str
    case_alias: str
    filters: str


class ExternalTableMetadata(BaseModel):
    set_type: str
    path: str
    label: str
    id_column: str
    filters: str


class ExternalTablesConfig(BaseModel):
    holdout_set_names: List[str]
    metadata: List[ExternalTableMetadata]


class SamplingConfig(BaseModel):
    bootstrap_iterations: int = 60
    cross_val_iterations: int = 10
    last_iteration: int = 0
    sequestered_ids: List[str] = []
    shuffle_labels: bool = False


class ModelConfig(BaseModel):
    hyperparameter_optimization: bool = True
    calculate_shapely_explanations: bool = False


class Config(BaseModel):
    vcf_like: VcfLikeConfig
    tracking: TrackingConfig
    clinical_table: ClinicalTableConfig
    external_tables: ExternalTablesConfig
    sampling: SamplingConfig
    model: ModelConfig


config = Config(
    vcf_like=VcfLikeConfig(
        path="../adhoc analysis/Variant_report_NUPs_fixed_2022-03-28.xlsx",
        sheet="all cases vs all controls",
        index_column=["chrom", "position", "rsID", "Gene"],
        gene_multi_index_level=3,
        aggregate_genes_by=None,
        compound_sample_id_delimiter="__",
        compound_sample_id_start_index=1,
        compound_sample_meta_id_start_index=1,
        binarize=False,
        zygosity=True,
        min_allele_frequency=0.0025,
        max_allele_frequency=1.0,
        max_variants=None,
        frequency_match_reference=None,
        filters={},
    ),
    tracking=TrackingConfig(
        name="NUP variants (rare-binned, rsID only)\nTrained on: AnswerALS cases & non-neurological controls (Caucasian)",
        entity="ejmockler",
        project="highReg-l1-NUPs60-aals-rsID-rareBinned-0.0025MAF",
        plot_all_sample_importances=True,
        remote=False,
    ),
    clinical_table=ClinicalTableConfig(
        path="../adhoc analysis/ACWM.xlsx",
        id_column="ExternalSampleId",
        subject_id_column="ExternalSubjectId",
        label_column="Subject Group",
        control_labels=["Non-Neurological Control"],
        case_labels=["ALS Spectrum MND"],
        control_alias="control",
        case_alias="case",
        filters="pct_european>=0.85",
    ),
    externa_tables=ExternalTablesConfig(
        holdout_set_names=[
            "AnswerALS Cases vs. Controls (Ethnically-Variable)",
            "Other Neurological Cases vs. Controls (Ethnically-Variable)",
        ],
        metadata=[
            ExternalTableMetadata(
                set_type="crossval",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filters="`Superpopulation code`=='EUR'",
            ),
            # ... Include other ExternalTableMetadata instances here
        ],
    ),
    sampling=SamplingConfig(
        bootstrap_iterations=60,
        cross_val_iterations=10,
        last_iteration=0,
        sequestered_ids=[],
        shuffle_labels=False,
    ),
    model=ModelConfig(
        hyperparameter_optimization=True,
        calculate_shapely_explanations=False,
    ),
)
