import hail as hl
import logging
from config import VCFConfig
import re
from typing import List

class GenotypeProcessor:
    def __init__(self, config: VCFConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, mt: hl.MatrixTable, gtf_ht: hl.Table) -> hl.MatrixTable:
        # Apply allele models
        mt = self._apply_allele_model(mt)

        # Drop invariant variants
        mt = self._drop_invariant_variants(mt)

        # Filter by chromosome pattern (include biologically useful contigs)
        mt = self._filter_biologically_useful_contigs(mt)

        # Filter to only include SNPs
        mt = self._filter_snps(mt)

        # Calculate allele frequencies and apply MAF filter
        mt = self._apply_maf_filter(mt)

        # Aggregate genes if specified
        if hasattr(self.config, 'aggregate_genes_by'):
            mt = self._aggregate_genes(mt)

        return mt

    def align_to_annotations(self, mt: hl.MatrixTable, gtf_ht: hl.Table) -> hl.MatrixTable:
        # Ensure gtf_ht is keyed by interval
        if not isinstance(gtf_ht.key[0], hl.expr.IntervalExpression):
            raise ValueError("GTF table must be keyed by interval")

        # Annotate with a boolean indicating if the variant is within any GTF interval
        mt = mt.annotate_rows(in_gtf_interval = hl.is_defined(gtf_ht[mt.locus]))

        # Count variants before filtering
        n_variants_before = mt.count_rows()

        # Filter rows
        mt = mt.filter_rows(mt.in_gtf_interval)

        # Count variants after filtering
        n_variants_after = mt.count_rows()

        # Calculate and log the number of dropped variants
        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Dropped {n_variants_dropped} variants after aligning to annotations.")

        return mt.drop('in_gtf_interval')

    def _apply_allele_model(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        if self.config.binarize:
            mt = mt.annotate_entries(GT_processed=hl.if_else(mt.GT.is_non_ref(), 1, 0))
        elif self.config.zygosity:
            mt = mt.annotate_entries(
                GT_processed=hl.case()
                .when(mt.GT.is_hom_ref(), 0)
                .when(mt.GT.is_het(), 1)
                .when(mt.GT.is_hom_var(), 2)
                .default(hl.null('int'))
            )
        else:
            mt = mt.annotate_entries(GT_processed=mt.GT.n_alt_alleles())
        return mt

    def _apply_maf_filter(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        mt = mt.annotate_rows(
            maf=hl.agg.sum(mt.GT_processed) / (2 * hl.agg.count_where(hl.is_defined(mt.GT_processed)))
        )
        min_maf = self.config.min_allele_frequency
        max_maf = self.config.max_allele_frequency
        
        n_variants_before = mt.count_rows()
        mt = mt.filter_rows((mt.maf >= min_maf) & (mt.maf <= max_maf))
        n_variants_after = mt.count_rows()
        
        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Filtered variants based on minor allele frequency. "
                         f"Min MAF: {min_maf}, Max MAF: {max_maf}. "
                         f"Dropped {n_variants_dropped} variants.")
        
        return mt

    def _drop_invariant_variants(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        # Count variants before filtering
        n_variants_before = mt.count_rows()

        # Calculate the number of unique genotypes
        mt = mt.annotate_rows(
            n_unique_genotypes = hl.agg.collect_as_set(mt.GT_processed).length()
        )
        
        # Filter out variants with only one unique genotype (invariant)
        mt = mt.filter_rows(mt.n_unique_genotypes > 1)
        
        # Count remaining variants
        n_variants_after = mt.count_rows()
        
        # Calculate and log the number of dropped variants
        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Dropped {n_variants_dropped} invariant variants.")
        
        return mt.drop('n_unique_genotypes')

    def _filter_biologically_useful_contigs(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        # Regular expression to match primary chromosomes, mitochondrial DNA, PAR regions, and alternate haplotypes
        contig_pattern = re.compile(
            r'^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|M|.*_alt|.*_PAR)$'  
            # Match 1-22, X, Y, M, and any contig with "_alt" or "_PAR"
        )

        # Count variants before and after filtering
        n_variants_before = mt.count_rows()
        mt = mt.filter_rows(mt.locus.contig.matches(contig_pattern.pattern))
        n_variants_after = mt.count_rows()

        # Log the number of dropped variants
        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Filtered irrelevant contigs. Dropped {n_variants_dropped} variants.")

        return mt

    def _filter_snps(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        # Count variants before and after filtering
        n_variants_before = mt.count_rows()
        mt = mt.filter_rows(
            hl.all(
                hl.len(mt.alleles) == 2,  # Ensure there are exactly two alleles
                hl.len(mt.alleles[0]) == 1,  # Reference allele is a single nucleotide
                hl.len(mt.alleles[1]) == 1   # Alternate allele is a single nucleotide
            )
        )
        n_variants_after = mt.count_rows()

        # Log the number of dropped variants
        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Filtered to only include SNPs. Dropped {n_variants_dropped} variants.")

        return mt

    def _aggregate_genes(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        # Implement gene aggregation logic here
        # This is a placeholder and should be implemented based on specific requirements
        return mt

    def fetch_genotypes(self, mt: hl.MatrixTable, sample_ids: List[str]) -> hl.MatrixTable:
        """
        Fetch genotypes for the given sample IDs.

        :param mt: Hail MatrixTable containing genotype data.
        :param sample_ids: List of sample IDs to fetch genotypes for.
        :return: Filtered MatrixTable containing only the specified samples.
        """
        filtered_mt = mt.filter_cols(hl.literal(sample_ids).contains(mt.s))
        return filtered_mt