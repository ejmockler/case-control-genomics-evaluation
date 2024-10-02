import hail as hl
import logging
from config import VCFConfig, GTFConfig
from typing import List, Union, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pandas as pd

class GenotypeProcessor:
    def __init__(self, config: VCFConfig, gtf_config: GTFConfig):
        self.config = config
        self.gtf_config = gtf_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(
        self, 
        mt: hl.MatrixTable, 
        gtf_ht: hl.Table, 
        gmt_df: Optional[pd.DataFrame] = None
    ) -> hl.MatrixTable:
        """
        Processes the MatrixTable by applying various filters, including GMT-based filtering.

        Args:
            mt (hl.MatrixTable): Input MatrixTable.
            gtf_ht (hl.Table): Hail Table with GTF annotations.
            gmt_df (Optional[pd.DataFrame]): DataFrame containing gene sets.

        Returns:
            hl.MatrixTable: Processed MatrixTable.
        """
        mt = self._preprocess_and_filter(mt)
        
        # Filter GTF to include only genes present in GMT DataFrame
        if gmt_df is not None and gtf_ht is not None and not gmt_df.empty:
            gtf_ht = self._filter_gtf_by_gmt(gtf_ht, gmt_df)
        
        # Align to annotations using the filtered GTF
        mt = self._align_to_annotations(mt, gtf_ht)
        
        # Create variant_id as a row field
        mt = mt.annotate_rows(variant_id = hl.str(mt.locus.contig) + ':' + hl.str(mt.locus.position))
        
        return mt

    def _preprocess_and_filter(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        """Applies all preprocessing steps and filters in a single pass."""
        mt = mt.annotate_entries(GT_processed=self._get_processed_gt(mt.GT))
        
        mt = mt.annotate_rows(
            maf=hl.agg.mean(mt.GT_processed) / 2,
            n_unique_genotypes=hl.agg.collect_as_set(mt.GT_processed).length(),
            has_missing=hl.agg.any(hl.is_missing(mt.GT_processed))
        )

        mt = mt.filter_rows(
            (mt.n_unique_genotypes > 1) &  # Drop invariant variants
            (mt.locus.contig.matches(self._get_contig_pattern())) &  # Filter biologically useful contigs
            hl.all(lambda a: hl.len(a) == 1, mt.alleles) &  # Filter to only include SNPs
            (hl.len(mt.alleles) == 2) &  # Ensure there are exactly two alleles
            (mt.maf >= self.config.min_allele_frequency) &  # Apply MAF filter
            (mt.maf <= self.config.max_allele_frequency) &
            (~mt.has_missing)  # Drop variants with any missing values
        )

        self._log_filtering_results(mt)
        return mt

    def _get_processed_gt(self, gt):
        if self.config.binarize:
            return hl.if_else(gt.is_non_ref(), 1, 0)
        elif self.config.zygosity:
            return hl.case().when(gt.is_hom_ref(), 0).when(gt.is_het(), 1).when(gt.is_hom_var(), 2).default(hl.null('int'))
        else:
            return gt.n_alt_alleles()

    @staticmethod
    def _get_contig_pattern():
        return r'^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|M|.*_alt|.*_PAR)$'

    def _log_filtering_results(self, mt: hl.MatrixTable):
        n_variants = mt.count_rows()
        self.logger.info(f"After filtering, {n_variants} variants remain.")

    def _filter_gtf_by_gmt(self, gtf_ht: hl.Table, gmt_df: pd.DataFrame) -> hl.Table:
        gene_names = set(gmt_df['genes'].explode().unique())
        self.logger.info(f"Filtering GTF annotations to include only {len(gene_names)} genes from GMT.")
        
        n_intervals_before = gtf_ht.count()
        gtf_ht = gtf_ht.filter(hl.literal(gene_names).contains(gtf_ht[self.gtf_config.gene_name_field]))
        n_intervals_after = gtf_ht.count()
        
        n_intervals_dropped = n_intervals_before - n_intervals_after
        self.logger.info(f"Dropped {n_intervals_dropped} GTF intervals not containing genes in the set.")
        
        return gtf_ht

    def _align_to_annotations(self, mt: hl.MatrixTable, gtf_ht: hl.Table) -> hl.MatrixTable:
        if not isinstance(gtf_ht.key[0], hl.expr.IntervalExpression):
            raise ValueError("GTF table must be keyed by interval")

        mt = mt.annotate_rows(in_gtf_interval=hl.is_defined(gtf_ht[mt.locus]))
        n_variants_before = mt.count_rows()
        mt = mt.filter_rows(mt.in_gtf_interval)
        n_variants_after = mt.count_rows()

        n_variants_dropped = n_variants_before - n_variants_after
        self.logger.info(f"Dropped {n_variants_dropped} variants after aligning to annotations.")
        self.logger.info(f"{n_variants_after} variants remain after alignment.")
        
        return mt.drop('in_gtf_interval')

    def to_spark_df(self, mt: hl.MatrixTable) -> SparkDataFrame:
        """
        Convert the preprocessed MatrixTable to a Spark DataFrame.

        Args:
            mt (hl.MatrixTable): Processed Hail MatrixTable.

        Returns:
            SparkDataFrame: Converted Spark DataFrame.
        """
        self.logger.info("Converting processed MatrixTable to Spark DataFrame.")
        
        # First, create a table with row and entry fields
        row_entry_table = mt.select_entries('GT_processed') 

        # Now, flatten this to a table
        flat_table = row_entry_table.entries()

        # Convert to Spark DataFrame
        spark_df = flat_table.select(
            flat_table.variant_id,
            sample_id = flat_table.s,
            GT_processed = flat_table.GT_processed
        ).to_spark()

        return spark_df

    def fetch_genotypes(
        self, 
        data: Union[hl.MatrixTable, SparkDataFrame],
        sample_ids: List[str], 
        return_spark: bool = False
    ) -> Union[hl.MatrixTable, SparkDataFrame]:
        """
        Fetch genotypes for the given sample IDs.

        :param data: Input dataset, either a Hail MatrixTable or a Spark DataFrame.
        :param sample_ids: List of sample IDs to fetch genotypes for.
        :param return_spark: If True, returns a PySpark DataFrame.
                             If False, returns a Hail MatrixTable.
        :return: Filtered MatrixTable or PySpark DataFrame based on 'return_spark' flag.
        """
        if isinstance(data, hl.MatrixTable):
            self.logger.info("Processing Hail MatrixTable.")
            # Step 1: Filter the MatrixTable to include only specified samples
            filtered_mt = data.filter_cols(hl.literal(sample_ids).contains(data.s))
            self.logger.info(f"Filtered MatrixTable to include {len(sample_ids)} samples.")

            if return_spark:
                self.logger.info("Converting filtered MatrixTable to PySpark DataFrame.")
                try:
                    # Step 2: Access entry fields
                    entries_table = filtered_mt.entries()

                    # Step 3: Select and rename fields without altering key fields directly
                    # It's crucial to remove keys before selecting fields to avoid key conflicts
                    unkeyed_entries = entries_table.key_by()

                    # Step 4: Annotate or select fields as needed
                    selected_table = unkeyed_entries.select(
                        variant_id = unkeyed_entries.variant_id,
                        sample_id = unkeyed_entries.s,
                        GT_processed = unkeyed_entries.GT_processed
                    )

                    # Step 5: Rekey the table by 'sample_id' and 'variant_id' to prepare for pivoting
                    rekeyed_table = selected_table.key_by('sample_id', 'variant_id')

                    # Step 6: Convert Hail Table to PySpark DataFrame
                    spark_df = rekeyed_table.to_spark()

                    # Step 7: Pivot the DataFrame to have samples as rows and variants as columns
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

        elif isinstance(data, SparkDataFrame):
            if return_spark:
                self.logger.info("Returning the provided Spark DataFrame.")
                return data.filter(F.col("sample_id").isin(sample_ids))
        else:
            raise TypeError("Input data must be either a Hail MatrixTable or a Spark DataFrame.")