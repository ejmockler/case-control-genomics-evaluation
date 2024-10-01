import hail as hl
import logging
from config import VCFConfig, GMTConfig, GTFConfig
import re
from typing import List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
import pandas as pd

class GenotypeProcessor:
    def __init__(self, config: VCFConfig, gtf_config: GTFConfig):
        self.config = config
        self.gtf_config = gtf_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, mt: hl.MatrixTable, gtf_ht: hl.Table, gmt_df: Optional[pd.DataFrame] = None) -> hl.MatrixTable:
        """
        Processes the MatrixTable by applying various filters, including GMT-based filtering.

        Args:
            mt (hl.MatrixTable): Input MatrixTable.
            gtf_ht (hl.Table): Hail Table with GTF annotations.
            gmt_df (Optional[pd.DataFrame]): DataFrame containing gene sets.

        Returns:
            hl.MatrixTable: Processed MatrixTable.
        """
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

        # Filter GTF to include only genes present in GMT DataFrame
        if gmt_df is not None and gtf_ht is not None and not gmt_df.empty:
            # Cast as set since hail can't convert numpy dtypes
            gene_names = set(gmt_df['genes'].explode().unique())
            self.logger.info(f"Filtering GTF annotations to include only {len(gene_names)} genes from GMT.")
            
            # Count GTF intervals before filtering
            n_intervals_before = gtf_ht.count()
            
            # Filter GTF
            gtf_ht = gtf_ht.filter(hl.literal(gene_names).contains(gtf_ht[self.gtf_config.gene_name_field]))
            
            # Count GTF intervals after filtering
            n_intervals_after = gtf_ht.count()
            
            # Calculate and log the number of dropped intervals
            n_intervals_dropped = n_intervals_before - n_intervals_after
            self.logger.info(f"Dropped {n_intervals_dropped} GTF intervals not containing genes in the set.")

        # Align to annotations using the filtered GTF
        mt = self.align_to_annotations(mt, gtf_ht)

        return mt

    def align_to_annotations(self, mt: hl.MatrixTable, gtf_ht: hl.Table) -> hl.MatrixTable:
        # Ensure gtf_ht is keyed by interval
        if not isinstance(gtf_ht.key[0], hl.expr.IntervalExpression):
            raise ValueError("GTF table must be keyed by interval")

        # Annotate with a boolean indicating if the variant is within any GTF interval
        mt = mt.annotate_rows(in_gtf_interval=hl.is_defined(gtf_ht[mt.locus]))

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
            n_unique_genotypes=hl.agg.collect_as_set(mt.GT_processed).length()
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

    def fetch_genotypes(
        self, 
        mt: hl.MatrixTable, 
        sample_ids: List[str], 
        return_spark: bool = False
    ) -> Optional[SparkDataFrame]:
        """
        Fetch genotypes for the given sample IDs.

        :param mt: Hail MatrixTable containing genotype data.
        :param sample_ids: List of sample IDs to fetch genotypes for.
        :param return_spark: If True, returns a PySpark DataFrame containing only 'GT_processed'.
                              If False, returns a Hail MatrixTable.
        :return: Filtered MatrixTable or PySpark DataFrame based on 'return_spark' flag.
        """
        filtered_mt = mt.filter_cols(hl.literal(sample_ids).contains(mt.s))
        self.logger.info(f"Filtered MatrixTable to include {len(sample_ids)} samples.")

        if return_spark:
            self.logger.info("Converting filtered MatrixTable to PySpark DataFrame with only 'GT_processed'.")
            try:
                # Step 1: Access entry fields
                entries_table = filtered_mt.entries()

                # Step 2: Create 'variant_id' by combining 'locus.contig' and 'locus.position'
                entries_table = entries_table.annotate(
                    variant_id=hl.str(entries_table.locus.contig) + ':' + hl.str(entries_table.locus.position)
                )

                # Step 3: Select and rename fields without altering key fields directly
                gt_table = entries_table.select(
                    sample_id=entries_table.s,
                    variant_id=entries_table.variant_id,
                    GT_processed=entries_table.GT_processed
                )

                # Step 4: Rekey the table by 'sample_id' and 'variant_id'
                gt_table = gt_table.key_by('sample_id', 'variant_id')

                # Step 5: Convert Hail Table to PySpark DataFrame
                spark_df = gt_table.to_spark()

                # Step 6: Pivot the DataFrame to have samples as rows and variants as columns
                spark_df_pivot = spark_df.groupBy("sample_id") \
                    .pivot("variant_id") \
                    .agg(F.first("GT_processed")) \
                    .fillna(0)  # Replace missing values with 0 or an appropriate default

                return spark_df_pivot
            except Exception as e:
                self.logger.error(f"Failed to convert MatrixTable to PySpark DataFrame: {e}")
                raise e
        else:
            self.logger.info("Returning filtered Hail MatrixTable.")
            return filtered_mt

def calculate_ld_matrix(mt, window_size=1000000, min_af=0.01):
    """
    Calculate the LD matrix for a given MatrixTable.
    
    :param mt: Input MatrixTable
    :param window_size: Window size in base pairs for LD calculation (default: 1,000,000)
    :param min_af: Minimum allele frequency threshold (default: 0.01)
    :return: LD matrix as a BlockMatrix
    """
    # Ensure the MatrixTable is row-keyed by locus
    if 'locus' not in mt.row_key:
        mt = mt.key_rows_by('locus')
    
    # Filter variants by allele frequency
    mt = mt.filter_rows(hl.agg.stats(mt.GT.n_alt_alleles()).mean / 2 >= min_af)
    
    # Prune variants in linkage equilibrium
    pruned_variant_table = hl.ld_prune(mt.GT, r2=0.2, bp_window_size=window_size)
    mt = mt.filter_rows(hl.is_defined(pruned_variant_table[mt.row_key]))
    
    # Calculate LD matrix
    ld_matrix = hl.ld_matrix(mt.GT, window_size)
    
    return ld_matrix