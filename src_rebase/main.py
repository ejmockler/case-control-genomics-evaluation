# main.py

import hail as hl
from config import config
import logging
from data.dataloader import create_loader
from data.genotype_processor import GenotypeProcessor

def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize Hail
    hl.init(default_reference='GRCh38')

    # Load VCF data
    vcf_loader = create_loader(config.vcf)
    vcf_mt = vcf_loader.load_data()
    print("VCF data loaded:")
    print(vcf_mt.describe())
    print(vcf_mt.show(n_rows=1, n_cols=1, include_row_fields=True))

    # Load clinical data
    clinical_loader = create_loader(config.clinical_table)
    clinical_data = clinical_loader.load_data()
    # Set genotype id as the row key
    # Set the index first
    clinical_data = clinical_data.set_index(config.clinical_table.id_column)
    # Then filter out duplicates
    clinical_data = clinical_data[
        ~clinical_data.index.duplicated(keep='first') & 
        ~clinical_data[config.clinical_table.subject_id_column].duplicated(keep='first')
    ]
    print("\nClinical data loaded:")
    print(clinical_data.head())

    # Load GTF data
    gtf_loader = create_loader(config.gtf)
    gtf_ht = gtf_loader.load_data()
    print("\nGTF data loaded:")
    print(gtf_ht.describe())

    # Process genotype data
    processor = GenotypeProcessor(config.vcf)
    mt_processed = processor.process(
        vcf_mt, gtf_ht)
    mt_processed =processor.align_to_annotations(mt_processed, gtf_ht)
    print("\nProcessed genotype data:")
    print(mt_processed.describe())

    mt_processed.filter_rows(hl.is_defined(gtf_ht[mt_processed.locus]))


if __name__ == "__main__":
    main()
