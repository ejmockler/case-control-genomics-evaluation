import os
from pathlib import Path
import shutil
os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '1200'
import hail as hl
from analysis.results_fetcher import MLflowResultFetcher
from analysis.summarize import OuterBootstrapSummarizer, save_summary, FeatureSelectionSummarizer, BootstrapSummarizer
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from models import stack
from eval.evaluation import bootstrap_models
from logging_config import setup_logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import tempfile
import gc

def cleanup_hail_files(tmp_dir: Path, logger: logging.Logger):
    """Clean up all Hail temporary files and directories"""
    try:
        # Stop Hail and clear memory first
        hl.stop()
        gc.collect()
        
        # Add MatrixTable-specific patterns
        hail_patterns = [
            "*.mt/*",  # MatrixTable directory contents
            "*.mt",    # MatrixTable directories
            "*.ht",
            "*.bgen",
            "*.idx",
            "*.vds",
            "*.tmp",
            "*.log",
            "hail_temporary_*",
            "hail_*"
        ]
        
        # Find and remove all Hail temp files/directories
        for pattern in hail_patterns:
            for path in tmp_dir.rglob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    logger.info(f"Cleaned up Hail temporary path: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")
                    
        # Also clean specific Hail directories if they exist
        hail_dirs = [
            tmp_dir / "hail_temporary",
            tmp_dir / "hail_logs",
            tmp_dir / "spark-warehouse"
        ]
        
        for dir_path in hail_dirs:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed Hail directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {dir_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during Hail cleanup: {e}")
        

def setup_mlflow_experiment(trackingConfig):
    logger = logging.getLogger(__name__)
    # Set up MLflow tracking
    mlflow.set_tracking_uri(trackingConfig.tracking_uri)
    client = MlflowClient()
    
    # Try to create the experiment, handle the case where it already exists
    try:
        experiment_id = client.create_experiment(trackingConfig.experiment_name)
    except MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            experiment = client.get_experiment_by_name(trackingConfig.experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
                logger.info(f"Experiment '{trackingConfig.experiment_name}' already exists. Using existing experiment.")
            else:
                raise e
        else:
            raise e
    return experiment_id

def summarize_inner_bootstrap_results(outer_iteration: int):
    logger = logging.getLogger(__name__)
    logger.info("Starting summarization of run data.")

    # Construct the mlartifacts base path (for local runs)
    mlartifacts_base_path = os.path.join(os.path.dirname(os.getcwd()), 'src', 'mlartifacts')

    # Initialize the ResultFetcher with the mlartifacts base path
    result_fetcher = MLflowResultFetcher(
        experiment_name=config.tracking.experiment_name,
        tracking_uri=config.tracking.tracking_uri,
        mlartifacts_base_path=mlartifacts_base_path
    )

    # Fetch all runs
    bootstrap_runs = result_fetcher.fetch_all_run_metadata(run_type='model')
    feature_selection_runs = result_fetcher.fetch_all_run_metadata(run_type='feature_selection')

    # Define artifact paths to fetch
    # Define artifact paths for different run types
    bootstrap_artifact_paths = [
        'feature_importances.json', 
        'train/confusion_matrix.json',
        'train/roc_curve.json',
        'train/pr_curve.json',
        'test/confusion_matrix.json',
        'test/roc_curve.json',
        'test/pr_curve.json',
        'test_sample_metrics.json',
        'train_sample_metrics.json',
    ] + [
        f"holdout/{tableMetadata.name}_sample_metrics.json" 
        for tableMetadata in config.holdout_tables.tables
    ]

    feature_selection_artifact_paths = [
        'validation_aggregated_results.json',
    ]

    # Fetch run data in parallel
    bootstrap_run_data = result_fetcher.fetch_runs_parallel(bootstrap_runs, bootstrap_artifact_paths)
    feature_selection_run_data = result_fetcher.fetch_runs_parallel(feature_selection_runs, feature_selection_artifact_paths)

    # Sanitize the bootstrap run data
    sanitized_bootstrap_data = result_fetcher.sanitize_run_data(bootstrap_run_data)

    logger.info("Summarizing run data.")
    
    # Initialize BootstrapSummarizer
    bootstrap_summarizer = BootstrapSummarizer(sanitized_bootstrap_data)
    # Generate bootstrap summaries
    bootstrap_summaries = bootstrap_summarizer.summarize_all()
    
    # Initialize FeatureSelectionSummarizer
    feature_selection_summarizer = FeatureSelectionSummarizer(feature_selection_run_data)
    # Generate feature selection summaries
    feature_selection_summaries = feature_selection_summarizer.summarize_all()
    
    # Combine all summaries
    all_summaries = {**bootstrap_summaries, **feature_selection_summaries}
    
    # Save all summaries to output directory
    save_summary(all_summaries, config.tracking, outer_iteration=outer_iteration)

    logger.info("Summarization completed and summaries saved.")


def summarize_outer_bootstrap_results():
    """
    Separate function to handle final summarization
    
    Args:
        experiment_id (str): MLflow experiment ID
        max_retries (int): Maximum number of times to check for completed runs
        retry_interval (int): Time in seconds between retries
    """
    logger = logging.getLogger(__name__)
    logger.info("Collecting and summarizing all outer bootstrap summaries.")

    # Initialize the ResultFetcher
    mlartifacts_base_path = os.path.join(os.path.dirname(os.getcwd()), 'src', 'mlruns')
    result_fetcher = MLflowResultFetcher(
        experiment_name=config.tracking.experiment_name,
        tracking_uri=config.tracking.tracking_uri,
        mlartifacts_base_path=mlartifacts_base_path
    )

    # Get all summary runs from outer bootstrap iterations
    summary_runs = result_fetcher.fetch_all_run_metadata(run_type='summary')
    
    # Define artifact paths to fetch
    summary_artifact_paths = [
        'crossval_model_metrics_summary.csv',
        'crossval_sample_metrics_summary.csv',  
        'holdout_model_metrics_summary.csv',
        'holdout_sample_metrics_summary.csv',
        'feature_importances_summary_all_models.csv',
        'feature_selection_validation_summary.csv',
        'feature_selection_sample_validation_metrics_summary.csv'
    ] + [
            f"feature_importances_summary_{model.__class__.__name__}.csv"
            for model in stack
        ]

    # Fetch summary data in parallel (only for completed runs)
    bootstrap_summaries = result_fetcher.fetch_runs_parallel(summary_runs, summary_artifact_paths)

    # Initialize OuterBootstrapSummarizer and generate final summaries
    outer_summarizer = OuterBootstrapSummarizer(bootstrap_summaries)
    final_summaries = outer_summarizer.summarize_all()

    # Save the final summaries
    save_summary(final_summaries, config.tracking, run_name='outer_bootstrap_final_summary')

    logger.info("Completed outer bootstrap summarization.")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    tmp_dir = Path(tempfile.gettempdir())


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
    sample_ids = mt_processed.s.collect()
    logger.info("Genotype data processed successfully.")

    # Save processed data as Parquet for downstream analysis
    logger.info("Converting processed data to Parquet format.")
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        parquet_path = tmp.name
        spark_df = genotype_processor.to_spark_df(mt_processed)
        pivoted_df = genotype_processor.pivot_genotypes(spark_df)
        pivoted_df.write.parquet(parquet_path, mode="overwrite")
        logger.info(f"Processed data saved to {parquet_path}")

        try:
            # Clean up Spark/Hail resources
            spark_df.unpersist()
            pivoted_df.unpersist()
            hl.stop()
            cleanup_hail_files(tmp_dir, logger)
            logger.info("Completed Hail cleanup")

            experiment_id = setup_mlflow_experiment(config.tracking)
            # Initialize SampleProcessor with config
            sample_processor = SampleProcessor(config, sample_ids)

            for i in range(config.sampling.outer_bootstrap_iterations):
                bootstrap_models(
                    sample_processor=sample_processor,
                    parquet_path=parquet_path,
                    samplingConfig=config.sampling,
                    trackingConfig=config.tracking,
                    experiment_id=experiment_id,
                    stack=stack,
                    outer_iteration=i+1
                )
                summarize_inner_bootstrap_results(outer_iteration=i+1)
            summarize_outer_bootstrap_results()
        finally:
            # Clean up temporary parquet file
            os.unlink(parquet_path)
            logger.info(f"Cleaned up temporary file: {parquet_path}")

    
if __name__ == "__main__":
    main()
