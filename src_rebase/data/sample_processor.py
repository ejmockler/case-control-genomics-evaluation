import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from typing import Dict, List, Tuple
from collections import defaultdict
from functools import cached_property
from multiprocessing import Pool, cpu_count
import logging

from config import Config, TableMetadata
from data.dataloader import create_loader

class SampleProcessor:
    def __init__(self, config: Config, genotype_ids: List[str]):
        self.config = config
        self.genotype_ids = genotype_ids
        self.logger = logging.getLogger(__name__)
        self.crossval_data = self._load_and_process_tables(self.config.crossval_tables.tables)
        self.holdout_data = self._load_and_process_tables(self.config.holdout_tables.tables)
    
    @cached_property
    def overlapping_ids(self) -> List[str]:
        return self.find_overlapping_ids()
    
    def get_overlapping_ids(self, dataset: str, label: str) -> List[str]:
        ids = self.get_ids(dataset=dataset, label=label)
        return [id for id in ids if id in self.overlapping_ids]

    @cached_property
    def overlapping_case_ids(self) -> Dict[str, List[str]]:
        return {
            'crossval': self.get_overlapping_ids(dataset='crossval', label='case'),
            'holdout': self.get_overlapping_ids(dataset='holdout', label='case')
        }

    @cached_property
    def overlapping_control_ids(self) -> Dict[str, List[str]]:
        return {
            'crossval': self.get_overlapping_ids(dataset='crossval', label='control'),
            'holdout': self.get_overlapping_ids(dataset='holdout', label='control')
        }
    
    def draw_samples(self, dataset: str = 'crossval') -> Dict[str, List[str]]:
        if dataset == 'crossval':
            # Determine the smaller class size
            overlapping_case_ids = self.overlapping_case_ids['crossval']
            overlapping_control_ids = self.overlapping_control_ids['crossval']
            min_class_size = min(len(overlapping_case_ids), len(overlapping_control_ids))
            
            # Randomly downsample IDs from both classes
            sampled_case_ids = pd.Series(overlapping_case_ids).sample(n=min_class_size, random_state=42).tolist()
            sampled_control_ids = pd.Series(overlapping_control_ids).sample(n=min_class_size, random_state=42).tolist()
            
            return {"case": sampled_case_ids, "control": sampled_control_ids}
        elif dataset == 'holdout':
            # Process each holdout dataset independently and include only overlapping IDs
            holdout_sample_ids = {}
            for table_name, table_info in self.holdout_data.items():
                df = table_info['data']
                overlapping_ids = [id for id in df.index.tolist() if id in self.overlapping_ids]
                holdout_sample_ids[table_name] = overlapping_ids
            return holdout_sample_ids
        else:
            raise ValueError("Dataset must be 'crossval' or 'holdout'")

    def get_ids(self, dataset: str = 'all', label: str = None) -> List[str]:
        if dataset not in ['crossval', 'holdout', 'all']:
            raise ValueError("Dataset must be 'crossval', 'holdout', or 'all'")
        
        ids = []
        
        if dataset in ['crossval', 'all']:
            ids.extend(self._get_ids_from_data(self.crossval_data, label))
        
        if dataset in ['holdout', 'all']:
            ids.extend(self._get_ids_from_data(self.holdout_data, label))
        
        return ids
    
    def find_overlapping_ids(self) -> List[str]:
        sample_ids = self.get_ids('all')
        matches = set()

        # Determine the number of processes to use
        num_processes = min(cpu_count(), len(sample_ids))

        # Split the sample_ids into chunks for multiprocessing
        chunk_size = (len(sample_ids) + num_processes - 1) // num_processes
        sample_id_chunks = [sample_ids[i:i + chunk_size] for i in range(0, len(sample_ids), chunk_size)]

        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self._find_matches_in_chunk, [(chunk, self.genotype_ids) for chunk in sample_id_chunks])

        for result in results:
            matches.update(result)

        self.logger.info(f"""
                         Found {len(matches)} overlapping IDs
                          {len(self.genotype_ids) - len(matches)} IDs do not overlap
                          """)
        return list(matches)

    def _find_matches_in_chunk(self, sample_id_chunk: List[str], genotype_ids: List[str]) -> List[str]:
        matches = set()
        genotype_id_set = set(genotype_ids)
        
        for sample_id in sample_id_chunk:
            # Check for exact match
            if sample_id in genotype_id_set:
                matches.add(sample_id)
            
            # Check for partial matches
            for genotype_id in genotype_ids:
                if genotype_id != sample_id and (sample_id in genotype_id or genotype_id in sample_id):
                    matches.add(genotype_id)
                    
        return list(matches)

    def _load_and_process_tables(self, table_configs: List[TableMetadata]) -> Dict[str, pd.DataFrame]:
        processed_data = {}
        for table_config in table_configs:
            loader = create_loader(table_config)
            df = loader.load_data()
            processed_data[table_config.name] = {'data': df, 'metadata': table_config}
        return processed_data

    def _get_ids_from_data(self, data: Dict[str, Dict[str, pd.DataFrame | TableMetadata]], label: str = None) -> List[str]:
        ids = []
        for table_data in data.values():
            df = table_data['data']
            metadata = table_data['metadata']
            
            if label is None or metadata.label.lower() == label.lower():
                ids.extend(df.index.tolist())
        
        return ids

    def _get_label_for_id(self, id: str) -> str:
        for dataset in [self.crossval_data, self.holdout_data]:
            for table_data in dataset.values():
                df = table_data['data']
                metadata = table_data['metadata']
                if id in df.index.tolist():
                    return metadata.label.lower()
        return None
