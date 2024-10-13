import hail as hl
import logging

from pyspark import SparkContext
from config import VCFConfig, GTFConfig
from typing import List, Union, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from ray.data import Dataset as RayDataset
import ray
import pandas as pd
import numpy as np

from eval.feature_selection import detect_data_threshold

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
        
        return mt

    def _preprocess_and_filter(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        """Applies all preprocessing steps and filters in a single pass."""
        # Annotate entries with processed genotype
        mt = mt.annotate_entries(GT_processed=self._get_processed_gt(mt.GT))
        
        # Annotate rows with necessary statistics including variance
        mt = mt.annotate_rows(
            maf=hl.if_else(
                self.config.binarize,
                hl.agg.mean(mt.GT_processed),        # Proportion of carriers
                hl.agg.mean(mt.GT_processed) / 2     # Traditional allele frequency
            ),
            stdev=hl.agg.stats(mt.GT_processed).stdev,
            has_missing=hl.agg.any(hl.is_missing(mt.GT_processed)),
            variant_id=hl.str(mt.locus.contig) + ':' + hl.str(mt.locus.position)
        )
        
        # Collect stdev values for threshold detection
        # self.logger.info("Collecting standard deviation values for threshold detection.")
        # stdev_rows = mt.rows().select('stdev').collect()
        # stdevs = np.array([row['stdev'] for row in stdev_rows if row['stdev'] is not None])
        
        # if len(stdevs) == 0:
        #     raise ValueError("No variance values collected. Please check your data.")
        
        # Detect threshold using KDE with Dip Test
        # self.logger.info("Detecting stdev threshold using KDE with Dip Test.")
        # threshold = detect_data_threshold(stdevs, bandwidth='silverman', plot=False)  # Set plot=True for visualization
        # self.config.min_variance = threshold
        # self.logger.info(f"Detected stdev threshold: {threshold:.4f}")
        
        # Apply row filters based on detected variance threshold and other criteria
        mt = mt.filter_rows(
            (mt.maf > 0) & (mt.maf < 1) &                             # Remove invariant variants
            (mt.locus.contig.matches(self._get_contig_pattern())) &   # Filter biologically useful contigs
            hl.all(lambda a: hl.len(a) == 1, mt.alleles) &            # Filter to only include SNPs
            (mt.maf >= self.config.min_allele_frequency) &            # Apply MAF filter
            (mt.maf <= self.config.max_allele_frequency) &            # Upper MAF filter
            (~mt.has_missing)                                         # Drop variants with any missing values
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

        spark_df = flat_table.select(
            variant_id=flat_table.variant_id,
            sample_id=flat_table.s,
            GT_processed=flat_table.GT_processed
        ).to_spark()

        return spark_df

    def fetch_genotypes(
        self, 
        data: Union[hl.MatrixTable, SparkDataFrame, RayDataset],
        sample_ids: List[str]
    ) -> SparkDataFrame:
        """
        Fetch genotypes for the specified sample IDs.

        Args:
            data (Union[hl.MatrixTable, SparkDataFrame, RayDataset]): Input data.
            sample_ids (List[str]): List of sample IDs to fetch.

        Returns:
            SparkDataFrame: Genotype data as Spark DataFrame.
        """
        if isinstance(data, hl.MatrixTable):
            return self._fetch_from_matrix_table(data, sample_ids)
        elif isinstance(data, SparkDataFrame):
            return data.filter(data.sample_id.isin(sample_ids))
        elif isinstance(data, RayDataset):
            return ray.data.from_spark(data.filter(lambda row: row["sample_id"] in sample_ids).to_spark())
        else:
            raise TypeError("Input data must be either a Hail MatrixTable, Spark DataFrame, or Ray Dataset.")

    def _fetch_from_matrix_table(self, mt: hl.MatrixTable, sample_ids: List[str]) -> SparkDataFrame:
        self.logger.info("Processing Hail MatrixTable.")
        filtered_mt = mt.filter_cols(hl.literal(sample_ids).contains(mt.s))
        
        filtered_sample_count = filtered_mt.count_cols()
        self.logger.info(f"Filtered MatrixTable to include {filtered_sample_count} samples out of {len(sample_ids)} requested.")
        
        if filtered_sample_count != len(sample_ids):
            self.logger.warning(f"Not all requested samples were found in the data. Expected {len(sample_ids)}, but got {filtered_sample_count}.")

        self.logger.info("Converting filtered MatrixTable to PySpark DataFrame.")
        try:
            entries_table = filtered_mt.entries().key_by()
            selected_table = entries_table.select(
                variant_id = entries_table.variant_id,
                sample_id = entries_table.s,
                GT_processed = entries_table.GT_processed
            )
            return selected_table.to_spark()
        except Exception as e:
            self.logger.error(f"Failed to convert MatrixTable to PySpark DataFrame: {e}")
            raise e

    def pivot_genotypes(self, spark_df: SparkDataFrame) -> SparkDataFrame:
        """
        Pivot the Spark DataFrame to create a feature matrix with samples as rows
        and variant_ids as columns.

        Args:
            spark_df (SparkDataFrame): Spark DataFrame with columns ['variant_id', 'sample_id', 'GT_processed'].

        Returns:
            SparkDataFrame: Pivoted Spark DataFrame with samples as rows and variant_ids as columns.
        """
        self.logger.info("Pivoting genotypes into feature matrix.")

        try:
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.pivotMaxValues", 100000)

            pivoted_df = spark_df.groupBy("sample_id") \
                .pivot("variant_id") \
                .agg(F.first("GT_processed"))

            self.logger.info("Successfully pivoted genotypes into feature matrix.")
            return pivoted_df
        except Exception as e:
            self.logger.error(f"Failed to pivot genotypes: {e}")
            raise e
