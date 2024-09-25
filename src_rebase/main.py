# main.py

import hail as hl
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor
from data.sample_processor import SampleProcessor
from models import stack

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

    # Collect all sample IDs from the processed MatrixTable
    all_genotype_ids = mt_processed.s.collect()

    # Initialize SampleProcessor with config and genotype IDs
    sample_processor = SampleProcessor(config, all_genotype_ids)

    # Draw train and test samples with a test split of 0.15
    train_test_sample_ids = sample_processor.draw_train_test_split(test_size=0.15)

    # Extract training samples, mapping, and labels
    train_samples = train_test_sample_ids['train']['samples']
    train_table_mapping = train_test_sample_ids['train']['table_mapping']
    train_label_mapping = train_test_sample_ids['train']['label_mapping']

    # Extract testing samples, mapping, and labels
    test_samples = train_test_sample_ids['test']['samples']
    test_table_mapping = train_test_sample_ids['test']['table_mapping']
    test_label_mapping = train_test_sample_ids['test']['label_mapping']

    # Count cases and controls
    train_cases = sum(1 for label in train_label_mapping.values() if label == 'case')
    train_controls = sum(1 for label in train_label_mapping.values() if label == 'control')
    test_cases = sum(1 for label in test_label_mapping.values() if label == 'case')
    test_controls = sum(1 for label in test_label_mapping.values() if label == 'control')

    # Log the counts
    logging.info(f"Number of training cases: {train_cases}")
    logging.info(f"Number of training controls: {train_controls}")
    logging.info(f"Number of testing cases: {test_cases}")
    logging.info(f"Number of testing controls: {test_controls}")

    # Proceed with model training using train_samples
    # Example:
    # train_mt = processor.fetch_genotypes(mt_processed, list(train_samples.keys()))
    # model = stack.train_model(train_mt, ...)

    # Proceed with model evaluation using test_samples
    # Example:
    # test_mt = processor.fetch_genotypes(mt_processed, list(test_samples.keys()))
    # evaluation_metrics = stack.evaluate_model(model, test_mt, ...)

    # Continue with other processing as needed

if __name__ == "__main__":
    main()
