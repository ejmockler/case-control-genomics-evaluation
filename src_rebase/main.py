# main.py

import hail as hl
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from models import stack
from eval.evaluation import bootstrap_models
import pandas as pd

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize Hail
    hl.init(default_reference='GRCh38')

    # Load and process data
    logger.info("Loading VCF data.")
    vcf_mt = create_loader(config.vcf).load_data()
    logger.info("Loading GTF data.")
    gtf_ht = create_loader(config.gtf).load_data()

    genotype_processor = GenotypeProcessor(config.vcf)
    logger.info("Processing genotype data.")
    mt_processed = genotype_processor.process(vcf_mt, gtf_ht)
    mt_processed = genotype_processor.align_to_annotations(mt_processed, gtf_ht)

    # Initialize SampleProcessor with config
    sample_processor = SampleProcessor(config, mt_processed.s.collect())

    # Perform bootstrapped model evaluations
    logger.info("Starting bootstrapped model evaluations.")
    results_df = bootstrap_models(
        sample_processor=sample_processor,
        genotype_processor=genotype_processor,
        mt_processed=mt_processed,
        config=config.sampling,
        stack=stack
    )

    # Log the aggregated results
    logger.info("Aggregated Bootstrapping Results:")
    for index, row in results_df.iterrows():
        logger.info(f"Iteration {row['iteration']}: Model {row['model']} achieved AUC {row['best_auc']} with params {row['best_params']}")

    # Optionally, save the results to a CSV file
    results_file = config.output.results_csv
    logger.info(f"Saving results to {results_file}")
    results_df.to_csv(results_file, index=False)

    logger.info("Bootstrapped model evaluations completed.")

if __name__ == "__main__":
    main()
