from typing import List, Optional, Dict
from pydantic import BaseModel


class VCFConfig(BaseModel):
    path: str
    binarize: bool = False
    zygosity: bool = True
    min_allele_frequency: float = 0.0025
    max_allele_frequency: float = 1.0
    max_variants: Optional[int] = None
    filters: list = []

class GTFConfig(BaseModel):
    path: str
    filters: list = []

class TrackingConfig(BaseModel):
    name: str
    entity: str
    project: str
    plot_all_sample_importances: bool = False

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

class TableMetadata(BaseModel):
    name: str
    set_type: str
    path: str
    label: str
    id_column: str
    filters: str


class HoldoutTablesConfig(BaseModel):
    metadata: List[TableMetadata]


class SamplingConfig(BaseModel):
    bootstrap_iterations: int = 60
    cross_val_iterations: int = 10
    sequestered_ids: List[str] = []
    shuffle_labels: bool = False


class ModelConfig(BaseModel):
    hyperparameter_optimization: bool = True
    calculate_shapely_explanations: bool = False


class Config(BaseModel):
    vcf: VCFConfig
    gtf: GTFConfig
    tracking: TrackingConfig
    clinical_table: ClinicalTableConfig
    holdout_tables: HoldoutTablesConfig
    sampling: SamplingConfig
    model: ModelConfig


config = Config(
    vcf=VCFConfig(
        path="../adhoc analysis/whole_genome_merged_no_vqsr_no_annotation_KarenRegions_MICROGLIAL_ANNOTATED.vcf.gz",
        binarize=False,
        zygosity=True,
        min_allele_frequency=0.0025,
        max_allele_frequency=1.0,
        max_variants=None,
        filters=[],
    ),
    gtf=GTFConfig(
        path="../adhoc analysis/gencode.v46.chr_patch_hapl_scaff.annotation.gtf.gz",
        filters=["(mt.transcript_type == 'protein_coding') | (mt.transcript_type == 'protein_coding_LoF')"], # to keep
    ),
    tracking=TrackingConfig(
        name="NUP variants (rare-binned, rsID only)\nTrained on: AnswerALS cases & non-neurological controls (Caucasian)",
        entity="ejmockler",
        project="highReg-l1-NUPs60-aals-rsID-rareBinned-0.0025MAF",
        plot_all_sample_importances=False,
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
    holdout_tables=HoldoutTablesConfig(
        metadata=[
            TableMetadata(
                name="1000 Genomes EUR",
                set_type="crossval",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filters="`Superpopulation code`=='EUR'",
            ),
        ],
    ),
    sampling=SamplingConfig(
        bootstrap_iterations=60,
        cross_val_iterations=10,
        sequestered_ids=[],
        shuffle_labels=False,
    ),
    model=ModelConfig(
        hyperparameter_optimization=True,
        calculate_shapely_explanations=False,
    ),
)
