# dataloader.py

import os
import pandas as pd
from typing import List, Optional, Dict
import logging
from abc import ABC, abstractmethod

import hail as hl
from config import GTFConfig, VCFConfig, GMTConfig
import gzip
import shutil
import subprocess

from config import TableMetadata
import gseapy as gp  # Import gseapy for GMT parsing

class DataLoader(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def load_data(self) -> hl.MatrixTable | pd.DataFrame:
        pass
    
    @abstractmethod
    def _apply_filters(self, data: hl.MatrixTable | pd.DataFrame | hl.Table) -> hl.MatrixTable | pd.DataFrame | hl.Table:
        pass

class BgzipMixin:
    def _ensure_bgz_compression(self, file_path: str) -> str:
        self.logger.info(f"Ensuring bgzip compression for: {file_path}")

        bgz_path = file_path[:-3] + ".bgz"
        self.logger.info(f"Converting {file_path} to block gzip format: {bgz_path}")

        try:
            # Decompress the existing .gz file
            decompressed_file_path = file_path[:-3]  # Remove .gz suffix
            with gzip.open(file_path, 'rb') as gz_in:
                with open(decompressed_file_path, 'wb') as decompressed_out:
                    shutil.copyfileobj(gz_in, decompressed_out)

            # Recompress the file as .bgz using the bgzip command
            result = subprocess.run(["bgzip", '-f', decompressed_file_path], check=True)

            if result.returncode == 0:
                # Rename .bgz back to the original .gz file
                os.rename(decompressed_file_path + ".gz", bgz_path)
                os.rename(bgz_path, file_path)
                self.logger.info(f"Successfully converted {file_path} to block-gzipped format.")

            return file_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during bgzip compression: {e}")
            raise RuntimeError(f"bgzip failed with error code {e.returncode}")
        except Exception as e:
            self.logger.error(f"Error during conversion to bgz: {e}")
            raise RuntimeError(f"Conversion to block gzip failed: {e}")

class GTFLoader(DataLoader, BgzipMixin):
    def __init__(self, config: GTFConfig):
        super().__init__(config)

    def load_data(self):
        gtf_path = self._ensure_bgz_compression(self.config.path)
        gtf_basepath = os.path.splitext(gtf_path)[0]
        ht_path = f"{gtf_basepath}.ht"

        if hl.hadoop_exists(ht_path):
            self.logger.info(f"Loading existing HailTable from disk: {ht_path}")
            gtf_ht = hl.read_table(ht_path)
        else:

            # Load the GTF file as a Hail Table using hl.import_table
            gtf_ht = hl.import_table(
                gtf_path,
                no_header=True,
                delimiter='\t',
                impute=True,
                comment='#',
                force_bgz=True,
                min_partitions=100
            )

            # Define the expected column names manually based on GTF structure
            gtf_ht = gtf_ht.rename({
                'f0': 'chromosome',
                'f1': 'source',
                'f2': 'feature',
                'f3': 'start',
                'f4': 'end',
                'f5': 'score',
                'f6': 'strand',
                'f7': 'frame',
                'f8': 'attribute'
            })

            # Filter out rows with contigs not in GRCh38
            valid_contigs = set(hl.get_reference('GRCh38').contigs)
            valid_contigs_set = hl.literal(valid_contigs)
            gtf_ht = gtf_ht.filter(valid_contigs_set.contains(gtf_ht.chromosome))

            # Create intervals from GTF start and end positions
            locus_start = hl.locus(gtf_ht.chromosome, hl.int(gtf_ht.start), reference_genome='GRCh38')
            locus_end = hl.locus(gtf_ht.chromosome, hl.int(gtf_ht.end), reference_genome='GRCh38')
            gtf_ht = gtf_ht.annotate(
                interval=hl.interval(
                    locus_start,
                    locus_end,
                    includes_start=True,
                    includes_end=True
                )
            )

            # Parse the attribute column using regex into a dictionary
            attribute_dict = gtf_ht.attribute.split(";").map(
                lambda x: x.strip()
            ).filter(
                lambda x: x != ""
            ).map(
                lambda x: (
                    x.split(' ')[0],  # key
                    hl.if_else(
                        x.contains('"'), 
                        x.split('"')[1], 
                        hl.null('str')
                    )  # value
                )
            )
            gtf_ht = gtf_ht.annotate(attributes=hl.dict(attribute_dict))

            # Extract specific attributes from the dictionary as columns (e.g., gene_id, gene_name)
            desired_attributes = gtf_ht.aggregate(hl.agg.explode(lambda x: hl.agg.collect_as_set(x), gtf_ht.attributes.keys())) # ['gene_id', 'gene_name', 'transcript_id']  # Add other attributes as needed
            for attr in desired_attributes:
                gtf_ht = gtf_ht.annotate(**{attr: gtf_ht.attributes.get(attr, hl.null(hl.tstr))})

            # Drop the attributes dictionary and only keep the desired fields
            gtf_ht = gtf_ht.drop('attributes')

            # Key the GTF table by the interval for fast joins
            gtf_ht = gtf_ht.key_by('interval')

            # Write the processed table to disk for future use
            gtf_ht.write(ht_path, overwrite=True)

        # Apply filters if specified in the config
        gtf_ht = self._apply_filters(gtf_ht)

        return gtf_ht
    
    def _apply_filters(self, ht: hl.Table) -> hl.Table:
        if self.config.filter:
            ht = ht.filter(eval(self.config.filter))
        return ht

class VCFLoader(DataLoader, BgzipMixin):
    def __init__(self, config: VCFConfig):
        super().__init__(config)

    def load_data(self):
        file_path = self.config.path
        self.logger.info(f"Loading VCF file with Hail: {file_path}")
        vcf_basepath = os.path.splitext(file_path)[0]
        mt_path = f"{vcf_basepath}.mt"

        if hl.hadoop_exists(mt_path):
            self.logger.info(f"Loading existing MatrixTable from disk: {mt_path}")
            mt = hl.read_matrix_table(mt_path)
        else:
            self.logger.info(f"Importing VCF and writing MatrixTable to disk: {mt_path}")
            mt = hl.import_vcf(file_path, reference_genome='GRCh38', force_bgz=True, min_partitions=10, array_elements_required=False)
            mt.write(mt_path, overwrite=True)
        
        # Apply filters if specified in the config
        mt = self._apply_filters(mt)

        return mt
    
    def _apply_filters(self, mt: hl.MatrixTable) -> hl.MatrixTable:
        if self.config.filter:
            mt = mt.filter_rows(eval(self.config.filter))
        return mt

class SmallTableLoader(DataLoader):
    def __init__(self, config: TableMetadata):
        super().__init__(config)

    def load_data(self) -> pd.DataFrame:
        file_path = self.config.path
        self.logger.info(f"Loading tabular file with pandas: {file_path}")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".tsv", ".txt")):
            df = pd.read_csv(file_path, sep="\t")
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular file type: {file_path}")
        df = self._apply_filters(df)
        df.set_index(self.config.id_column, inplace=True)
        # Drop duplicate index values, keeping the first occurrence
        duplicates = df.index.duplicated(keep='first')
        if duplicates.any():
            num_duplicates = duplicates.sum()
            self.logger.warning(f"Found {num_duplicates} duplicate index values. Dropping duplicates, keeping the first occurrence.")
            df = df[~duplicates]
        return df
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.filter:
            self.logger.info(f"Applying filter: {self.config.filter}")
            len_before = len(df)
            df = df.query(self.config.filter)
            len_after = len(df)
            self.logger.info(f"Filtered {len_before - len_after} rows")
        return df

class GMTLoader(DataLoader):
    def __init__(self, config: GMTConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_data(self) -> pd.DataFrame:
        """
        Loads and processes the GMT file into a pandas DataFrame using gseapy.
        
        Returns:
            pd.DataFrame: DataFrame containing gene sets.
        """
        file_path = self.config.path
        self.logger.info(f"Loading GMT file: {file_path}")
        
        # Read GMT file using gseapy
        gene_sets = self._parse_gmt(file_path)
        
        # Convert to DataFrame by exploding the gene lists
        gmt_df = self._convert_to_dataframe(gene_sets)
        
        # Apply filters if specified
        gmt_df = self._apply_filters(gmt_df)
        
        return gmt_df

    def _parse_gmt(self, file_path: str) -> List[Dict[str, any]]:
        """
        Parses the GMT file using gseapy.
        
        Args:
            file_path (str): Path to the GMT file.
        
        Returns:
            List[Dict[str, any]]: List of gene sets with their descriptions and genes.
        """
        try:
            gene_sets = gp.parser.read_gmt(file_path)  # Returns List[Tuple[str, str, List[str]]]
            self.logger.info(f"Parsed {len(gene_sets)} gene sets from GMT file using gseapy.")
            
            # Convert tuples to dictionaries for easier handling
            gene_set_records = [
                {"set": name, "genes": genes}
                for name, genes in gene_sets.items()
            ]
            return gene_set_records
        except Exception as e:
            self.logger.error(f"Failed to parse GMT file {file_path} with gseapy: {e}")
            raise e

    def _convert_to_dataframe(self, gene_sets: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Converts the list of gene sets into a pandas DataFrame by exploding the gene lists.
        
        Args:
            gene_sets (List[Dict[str, any]]): List of gene sets.
        
        Returns:
            pd.DataFrame: Exploded DataFrame with each gene in a separate row.
        """
        # Create initial DataFrame
        gmt_df = pd.DataFrame.from_dict(gene_sets)
        
        return gmt_df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies filtering to the GMT DataFrame based on the config.

        Args:
            df (pd.DataFrame): DataFrame with gene sets.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if self.config.filter:
            self.logger.info(f"Applying filter: {self.config.filter}")
            try:
                # Example filter: sets containing a specific gene
                df = df[df['gene'].apply(lambda gene: self.config.filter in gene)]
                self.logger.info(f"Filtered GMT DataFrame to {len(df)} entries.")
            except Exception as e:
                self.logger.error(f"Failed to apply filter '{self.config.filter}' on GMT DataFrame: {e}")
                raise e
        return df

def create_loader(config) -> DataLoader:
    """
    Factory function to create appropriate DataLoader instances based on config type.

    Args:
        config: Configuration object.

    Returns:
        DataLoader: An instance of a DataLoader subclass.

    Raises:
        ValueError: If config type is unsupported.
    """
    if isinstance(config, VCFConfig):
        return VCFLoader(config)
    elif isinstance(config, GMTConfig):
        return GMTLoader(config)
    elif isinstance(config, GTFConfig):
        return GTFLoader(config)
    elif isinstance(config, TableMetadata):
        return SmallTableLoader(config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")