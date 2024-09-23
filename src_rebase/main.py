# main.py

import hail as hl
from config import config
import logging
from collections import defaultdict
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor

def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize Hail
    hl.init(default_reference='GRCh38')

    # Load and process data
    vcf_mt = create_loader(config.vcf).load_data()
    gtf_ht = create_loader(config.gtf).load_data()

    processor = GenotypeProcessor(config.vcf)
    mt_processed = processor.process(vcf_mt, gtf_ht)
    mt_processed = processor.align_to_annotations(mt_processed, gtf_ht)

    # Get sample and genotype IDs
    sample_processor = SampleProcessor(config, mt_processed.s.collect())

    # Balance cross-validation samples
    crossval_sample_ids = sample_processor.draw_samples(dataset='crossval')
    logging.info(f"Balanced cross-validation sample IDs: {crossval_sample_ids}")

    holdout_sample_ids = sample_processor.draw_samples(dataset='holdout')
    for holdout_name, ids in holdout_sample_ids.items():
        logging.info(f"Processing dataset: {holdout_name} with sample IDs: {ids}")
        # Fetch genotypes for the drawn samples
        holdout_mt = processor.fetch_genotypes(mt_processed, ids)
        pass
        # Further processing for holdout_mt...

    # Continue with other processing as needed

if __name__ == "__main__":
    main()
