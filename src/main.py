import hail as hl
from analysis.results_fetcher import MLflowResultFetcher
from analysis.summarize import Summarizer, save_summary
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from models import stack
from eval.evaluation import bootstrap_models
from logging_config import setup_logging
import os

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize Hail
    hl.init(
        default_reference='GRCh38',
        spark_conf={
            'spark.driver.memory': '12g',
            'spark.executor.memory': '12g',
            'spark.executor.cores': '4',
        }
    )

    # Load VCF Data
    logger.info("Loading VCF data.")
    vcf_loader = create_loader(config.vcf)
    vcf_mt = vcf_loader.load_data()
    logger.info("VCF data loaded successfully.")

    # Load GTF Data
    logger.info("Loading GTF data.")
    gtf_loader = create_loader(config.gtf)
    gtf_ht = gtf_loader.load_data()
    logger.info("GTF data loaded successfully.")

    # Load GMT Data
    if config.gmt:
        logger.info("Loading GMT data.")
        gmt_loader = create_loader(config.gmt)
        gmt_df = gmt_loader.load_data()
        logger.info("GMT data loaded successfully.")
    else:
        logger.warning("No GMT configuration found. Skipping gene set filtering.")
        gmt_df = None

    # Initialize GenotypeProcessor
    genotype_processor = GenotypeProcessor(config.vcf, config.gtf)

    # Process Genotype Data with GTF and GMT
    logger.info("Processing genotype data.")
    mt_processed = genotype_processor.process(vcf_mt, gtf_ht, gmt_df)
    logger.info("Genotype data processed successfully.")

    # Initialize SampleProcessor with config
    sample_ids = mt_processed.s.collect()
    sample_processor = SampleProcessor(config, sample_ids)

    # Perform bootstrapped model evaluations
    bootstrap_models(
        sample_processor=sample_processor,
        genotype_processor=genotype_processor,
        data=mt_processed,  # Pass the processed MatrixTable
        samplingConfig=config.sampling,
        trackingConfig=config.tracking,
        stack=stack
    )

    logger.info("Starting summarization of run data.")

    # Initialize the ResultFetcher

    # Construct the mlartifacts base path (for local runs)
    mlartifacts_base_path = os.path.join(os.path.dirname(os.getcwd()), 'src', 'mlartifacts')

    # Initialize the ResultFetcher with the mlartifacts base path
    result_fetcher = MLflowResultFetcher(
        experiment_name=config.tracking.experiment_name,
        tracking_uri=config.tracking.tracking_uri,
        mlartifacts_base_path=mlartifacts_base_path
    )

        # Fetch all runs
    runs = result_fetcher.fetch_all_runs(run_type='model')

    # Define artifact paths to fetch
    artifact_paths = [
        'feature_importances.json', 
        'train/confusion_matrix.json',
        'train/roc_curve.json',
        'test/confusion_matrix.json',
        'test/roc_curve.json',
        'test_sample_metrics.json',
        'train_sample_metrics.json',
        ] + [
            f"holdout/{tableMetadata.name}_sample_metrics.json" 
            for tableMetadata in config.holdout_tables.tables
        ]

    # Fetch run data in parallel
    run_data = result_fetcher.fetch_runs_parallel(runs, artifact_paths)

    # Sanitize the run data
    sanitized_run_data = result_fetcher.sanitize_run_data(run_data)

    logger.info("Summarizing run data.")
    # Initialize Summarizer
    summarizer = Summarizer(sanitized_run_data)

    # Generate summaries
    summaries = summarizer.summarize_all()

    # Save summaries to output directory
    save_summary(summaries, config.tracking)

    logger.info("Summarization completed and summaries saved.")

    # Clean up Hail
    hl.stop()

if __name__ == "__main__":
    main()
