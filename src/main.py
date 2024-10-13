import hail as hl
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from models import stack
from eval.evaluation import bootstrap_models

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize Hail
    hl.init(
        default_reference='GRCh38',
        spark_conf={
            'spark.driver.memory': '8g',
            'spark.executor.memory': '8g',
            'spark.executor.cores': '4',
        }
    )

    try:
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
        results_df = bootstrap_models(
            sample_processor=sample_processor,
            genotype_processor=genotype_processor,
            data=mt_processed,  # Pass the processed MatrixTable
            samplingConfig=config.sampling,
            trackingConfig=config.tracking,
            stack=stack
        )

        # Log the aggregated results
        logger.info("Aggregated Bootstrapping Results:")
        for index, row in results_df.iterrows():
            logger.info(f"Iteration {row['iteration']}: Model {row['model_name']} achieved AUC {row['test_auc']} with params {row['best_params']}")

        # # Optionally, save the results to a CSV file
        # if hasattr(config, 'output') and hasattr(config.output, 'results_csv'):
        #     results_file = config.output.results_csv
        #     logger.info(f"Saving results to {results_file}")
        #     results_df.to_csv(results_file, index=False)
        # else:
        #     logger.warning("Output path for results CSV not found in configuration. Skipping saving results.")

    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")
        raise e

    finally:
        # Clean up Ray and Hail
        hl.stop()

if __name__ == "__main__":
    main()
