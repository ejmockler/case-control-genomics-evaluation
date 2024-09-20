# main.py

import logging
from typing import Optional
import ray
from ray.util import queue

# Import your configuration and modules
from config import (
    Config,
    VcfLikeConfig,
    TrackingConfig,
    ClinicalTableConfig,
    ExternalTablesConfig,
    ExternalTableMetadata,
    SamplingConfig,
    ModelConfig,
)
from data import GenotypeData, ClassificationResults
from models import (
    model_stack,
)  # Assuming models.py defines your models and hyperparameter spaces
from tasks.input import process_input_files
from tasks.predict import bootstrap
from tasks.visualize import track_project_visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config: Config):
    """
    Main function to orchestrate the classification pipeline.
    """
    # Initialize Ray
    ray.init()

    # Step 1: Process Input Files
    logger.info("Processing input files...")
    genotype_data, freq_reference_genotype_data, clinical_data = process_input_files(
        config
    )

    # Step 2: Prepare Cross-Validation Iterators
    from sklearn.model_selection import StratifiedKFold

    outer_cv_iterator = StratifiedKFold(
        n_splits=config.sampling.cross_val_iterations, shuffle=True, random_state=42
    )
    inner_cv_iterator = StratifiedKFold(
        n_splits=config.sampling.cross_val_iterations, shuffle=True, random_state=42
    )

    # Step 3: Prepare Bootstrap Arguments
    bootstrap_args_list = []
    for model, hyper_parameter_space in model_stack.items():
        bootstrap_args = (
            genotype_data,
            freq_reference_genotype_data,
            clinical_data,
            model,
            hyper_parameter_space,
            inner_cv_iterator,
            outer_cv_iterator,
            config,
            True,  # track results
        )
        bootstrap_args_list.append(bootstrap_args)

    # Step 4: Run Bootstrapping in Parallel using Ray
    logger.info("Starting bootstrapping...")
    bootstrap_results = []
    for args in bootstrap_args_list:
        bootstrap_results.append(bootstrap.remote(*args))

    # Retrieve results
    classification_results = ray.get(bootstrap_results)

    # Step 5: Track and Visualize Results
    logger.info("Tracking and visualizing results...")
    classification_results = ClassificationResults(model_results=classification_results)
    track_project_visualizations(classification_results, config)

    # Step 6: Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    # Load configuration
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
        external_tables=ExternalTablesConfig(
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
                # Add other ExternalTableMetadata instances here...
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

    # Run main function
    main(config)
