import pandas as pd
from typing import Dict, List, Optional, Tuple
from functools import cached_property
import logging
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from config import Config, TableMetadata
from data.dataloader import create_loader

class SampleProcessor:
    def __init__(self, config: Config, genotype_ids: List[str]):
        self.config = config
        self.genotype_ids = set(genotype_ids)
        self.logger = logging.getLogger(__name__)
        self.crossval_data = self._load_and_process_tables(self.config.crossval_tables.tables)
        self.holdout_data = self._load_and_process_tables(self.config.holdout_tables.tables)
        self.id_mapping = self._create_id_mapping()

        self.sample_draw_counts: Dict[str, int] = {sample_id: 0 for sample_id in self.id_mapping.keys()}

    def _create_id_mapping(self) -> Dict[str, str]:
        mapping = {}
        missing_samples = []
        for dataset_name, dataset in [('crossval', self.crossval_data), ('holdout', self.holdout_data)]:
            for table_name, table in dataset.items():
                for sample_id in table['data'].index:
                    if sample_id in mapping:
                        continue  # Skip if already mapped
                    if sample_id in self.genotype_ids:
                        mapping[sample_id] = sample_id
                        self.logger.debug(f"Mapped sample_id '{sample_id}' directly to genotype_id.")
                    else:
                        # Attempt to map using genotype_ids
                        matched = False
                        for genotype_id in self.genotype_ids:
                            if genotype_id in sample_id or sample_id in genotype_id:
                                mapping[sample_id] = genotype_id
                                matched = True
                                self.logger.debug(f"Mapped sample_id '{sample_id}' to genotype_id '{genotype_id}'.")
                                break
                        if not matched:
                            missing_samples.append(sample_id)
                            self.logger.warning(f"No genotype_id mapping found for sample_id '{sample_id}' in table '{table_name}' of dataset '{dataset_name}'.")

        return mapping

    def _load_and_process_tables(self, table_configs: List[TableMetadata]) -> Dict[str, Dict]:
        processed_tables = {}

        for table_config in table_configs:
            data_loader = create_loader(table_config)
            data = data_loader.load_data()

            # Add the 'table' column to identify the source table
            data = data.assign(table=table_config.name)

            # Assign the label based on the configuration
            data['label'] = table_config.label.lower()
            self.logger.info(f"Assigned label '{table_config.label}' to table '{table_config.name}'.")

            # Validate strata columns
            if table_config.strata_mapping:
                for standard_stratum, col in table_config.strata_mapping.items():
                    if col not in data.columns:
                        self.logger.error(f"Strata column '{col}' not found in table '{table_config.name}'.")
                        raise ValueError(f"Strata column '{col}' not found in table '{table_config.name}'.")

            # Drop rows with missing values in the strata columns
            if table_config.strata_mapping:
                strata_cols = list(table_config.strata_mapping.values())
                before_drop = len(data)
                data = data.dropna(subset=strata_cols)
                after_drop = len(data)
                self.logger.info(f"Dropped {before_drop - after_drop} samples with missing strata values in {table_config.name}.")

            # Validate label values
            valid_labels = {'case', 'control'}
            if not set(data['label']).issubset(valid_labels):
                invalid_labels = set(data['label']) - valid_labels
                self.logger.error(f"Invalid labels {invalid_labels} found in table '{table_config.name}'.")
                raise ValueError(f"Invalid labels {invalid_labels} in table '{table_config.name}'.")

            self.logger.info(f"Processed table '{table_config.name}' with {len(data)} samples.")

            processed_tables[table_config.name] = {
                'data': data,
                'metadata': table_config
            }

        return processed_tables

    @cached_property
    def overlapping_ids(self) -> set:
        return set(self.id_mapping.keys())

    @cached_property
    def overlapping_case_ids(self) -> Dict[str, List[str]]:
        return {
            'crossval': self._get_overlapping_ids('crossval', 'case'),
            'holdout': self._get_overlapping_ids('holdout', 'case')
        }

    @cached_property
    def overlapping_control_ids(self) -> Dict[str, List[str]]:
        return {
            'crossval': self._get_overlapping_ids('crossval', 'control'),
            'holdout': self._get_overlapping_ids('holdout', 'control')
        }

    def _get_overlapping_ids(self, dataset: str, label: str) -> List[str]:
        return [id for id in self.get_ids(dataset, label) if id in self.overlapping_ids]

    def get_ids(self, dataset: str = 'all', label: str = None) -> List[str]:
        if dataset not in ['crossval', 'holdout', 'all']:
            raise ValueError("Dataset must be 'crossval', 'holdout', or 'all'")

        ids = []
        data_sources = []
        if dataset in ['crossval', 'all']:
            data_sources.append(self.crossval_data)
        if dataset in ['holdout', 'all']:
            data_sources.append(self.holdout_data)

        for data in data_sources:
            for table_data in data.values():
                if label is None or table_data['metadata'].label.lower() == label.lower():
                    ids.extend(table_data['data'].index)

        return ids

    def get_labels(self, sample_ids: List[str], dataset: str = 'crossval') -> pd.Series:
        """
        Retrieves labels for the given sample IDs from the specified dataset.

        Args:
            sample_ids (List[str]): List of sample IDs to retrieve labels for.
            dataset (str): Dataset type ('crossval' or 'holdout'). Defaults to 'crossval'.

        Returns:
            pd.Series: A Pandas Series mapping sample IDs to their labels (1 for case, 0 for control).

        Raises:
            ValueError: If an invalid dataset is specified or if a label is missing for any sample.
        """
        if dataset not in ['crossval', 'holdout']:
            raise ValueError("Dataset must be 'crossval' or 'holdout'")

        label_mapping = {}

        for sample_id in sample_ids:
            # Check if the sample_id is directly in id_mapping
            if sample_id in self.id_mapping:
                resolved_id = self.id_mapping[sample_id]
            else:
                # If not, check if it's a genotype_id that's a value in id_mapping
                resolved_id = next((k for k, v in self.id_mapping.items() if v == sample_id), None)
            
            if resolved_id is None:
                self.logger.error(f"Sample ID '{sample_id}' not found in id_mapping.")
                raise ValueError(f"Sample ID '{sample_id}' not found in id_mapping.")
            label = self._get_label(resolved_id, dataset)
            label_mapping[sample_id] = 1 if label.lower() == 'case' else 0

        # Convert the dictionary to a Pandas Series
        labels_series = pd.Series(label_mapping)

        # Ensure all samples have labels
        if len(labels_series) != len(sample_ids):
            missing_samples = set(sample_ids) - set(labels_series.index)
            self.logger.error(f"Labels are missing for the following samples: {missing_samples}")
            raise ValueError(f"Labels are missing for the following samples: {missing_samples}")

        return labels_series

    def _get_label(self, sample_id: str, dataset: str) -> str:
        """Get the label for a given sample ID."""
        data_source = self.crossval_data if dataset == 'crossval' else self.holdout_data
        for table_info in data_source.values():
            if sample_id in table_info['data'].index:
                if isinstance(table_info['data'].loc[sample_id, 'label'], pd.Series):
                    print(sample_id)
                    print(table_info['data'].loc[sample_id, 'label'])
                return table_info['data'].loc[sample_id, 'label']
        raise ValueError(f"Sample ID '{sample_id}' not found in any table.")

    def draw_train_test_split(self, test_size: float = 0.15, random_state: int = 42, subset: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Draws a stratified train-test split from the cross-validation dataset.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing 'train' and 'test' sample IDs and their table mappings.
        """
        # Combine case and control IDs
        combined_ids = self.overlapping_case_ids['crossval'] + self.overlapping_control_ids['crossval']
        if subset is not None:
            combined_ids = [id for id in combined_ids if id in subset]

        # Initialize lists to collect strata and labels
        strata_list = []
        labels = []
        sample_ids = []

        # Iterate over each table to collect strata information
        for table_name, table_info in self.crossval_data.items():
            # Determine available IDs for the current table
            available_ids = table_info['data'].index.intersection(combined_ids)

            # Access only available IDs
            df_table = table_info['data'].loc[available_ids]

            if df_table.empty:
                self.logger.warning(f"No available data in table '{table_name}' after filtering.")
                continue

            # Retrieve strata columns for the current table based on sampling strata
            strata_columns = [
                table_info['metadata'].strata_mapping[stratum]
                for stratum in self.config.sampling.strata
                if table_info['metadata'].strata_mapping and stratum in table_info['metadata'].strata_mapping
            ]

            if not strata_columns:
                self.logger.warning(f"No valid strata columns found for table '{table_name}'. Skipping.")
                continue

            # Process strata
            df_table = self._process_strata(df_table, strata_columns)

            # Assign 'id' from the index
            df_table = df_table.assign(id=df_table.index)

            # Collect strata and labels
            strata_list.extend(df_table['stratum'])
            labels.extend([1 if table_info['metadata'].label == 'case' else 0] * len(df_table))
            sample_ids.extend(df_table['id'])

        if not strata_list:
            raise ValueError("No strata information available across crossval tables for sampling.")

        # Initialize StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        # Generate train and test indices
        for train_idx, test_idx in splitter.split(sample_ids, strata_list):
            train_sample_ids = [sample_ids[i] for i in train_idx]
            test_sample_ids = [sample_ids[i] for i in test_idx]
            break  # Only one split

        # Balance the training set
        balanced_train_ids = self._balance_classes(train_sample_ids, 'crossval')

        # Balance the test set
        balanced_test_ids = self._balance_classes(test_sample_ids, 'crossval')

        # Shuffle the balanced sample IDs
        np.random.seed(random_state)
        np.random.shuffle(balanced_train_ids)
        np.random.shuffle(balanced_test_ids)

        # Map sample IDs to genotype IDs and table names
        train_samples = {sample_id: self.id_mapping[sample_id] for sample_id in balanced_train_ids}
        train_table_mapping = {
            sample_id: self._get_table_name(sample_id, 'crossval')
            for sample_id in balanced_train_ids
        }
        train_label_mapping = {
            sample_id: self._get_label(sample_id, 'crossval')
            for sample_id in balanced_train_ids
        }

        test_samples = {sample_id: self.id_mapping[sample_id] for sample_id in balanced_test_ids}
        test_table_mapping = {
            sample_id: self._get_table_name(sample_id, 'crossval')
            for sample_id in balanced_test_ids
        }
        test_label_mapping = {
            sample_id: self._get_label(sample_id, 'crossval')
            for sample_id in balanced_test_ids
        }

        return {
            'train': {
                'samples': train_samples,
                'table_mapping': train_table_mapping,
                'label_mapping': train_label_mapping
            },
            'test': {
                'samples': test_samples,
                'table_mapping': test_table_mapping,
                'label_mapping': test_label_mapping
            }
        }

    def _balance_classes(self, sample_ids: List[str], dataset: str) -> List[str]:
        """
        Balances the classes by ensuring equal representation of cases and controls.
        Prefers samples that have been drawn less frequently to ensure uniform sampling.

        Args:
            sample_ids (List[str]): List of sample IDs to balance.
            dataset (str): The dataset type ('crossval' or 'holdout').

        Returns:
            List[str]: Balanced list of sample IDs.
        """
        cases = [id_ for id_ in sample_ids if id_ in self.overlapping_case_ids[dataset]]
        controls = [id_ for id_ in sample_ids if id_ in self.overlapping_control_ids[dataset]]

        # Sort cases and controls by their draw counts (ascending)
        cases_sorted = sorted(cases, key=lambda x: self.sample_draw_counts.get(x, 0))
        controls_sorted = sorted(controls, key=lambda x: self.sample_draw_counts.get(x, 0))

        min_count = min(len(cases_sorted), len(controls_sorted))
        balanced_cases = cases_sorted[:min_count]
        balanced_controls = controls_sorted[:min_count]
        balanced_sample_ids = balanced_cases + balanced_controls

        # Update draw counts in bulk using a dictionary comprehension
        self.sample_draw_counts.update({
            sample_id: self.sample_draw_counts.get(sample_id, 0) + 1 
            for sample_id in balanced_sample_ids})

        return balanced_sample_ids

    def _get_strata_columns(self, dataset: str) -> List[str]:
        """
        Retrieve strata columns from all tables within the specified dataset.
        """
        if dataset == 'crossval':
            tables = self.config.crossval_tables.tables
        elif dataset == 'holdout':
            tables = self.config.holdout_tables.tables
        else:
            raise ValueError("Dataset must be 'crossval' or 'holdout'")

        strata_columns = set()
        for table in tables:
            if table.strata_mapping:
                strata_columns.update(table.strata_mapping.values())

        return list(strata_columns)

    def _process_strata(self, df: pd.DataFrame, strata_columns: List[str]) -> pd.DataFrame:
        """
        Process strata columns by handling continuous variables through binning.
        Adds a 'stratum' column to the DataFrame for stratification.
        """
        df = df.copy()

        # Initialize stratum as empty strings
        df['stratum'] = ''

        for col in strata_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Continuous variable: bin into quartiles
                try:
                    df[f'{col}_binned'] = pd.qcut(df[col], q=4, duplicates='drop').astype(str)
                    df['stratum'] += df[f'{col}_binned'] + '_'
                except ValueError as e:
                    self.logger.warning(f"Could not bin column '{col}' due to error: {e}. Assigning 'Unknown'.")
                    df['stratum'] += 'Unknown_'
            else:
                # Categorical variable: use as is
                df['stratum'] += df[col].astype(str) + '_'

        # Remove trailing underscore
        df['stratum'] = df['stratum'].str.rstrip('_')

        # Assign the 'stratum' as a categorical column
        df['stratum'] = df['stratum'].astype('category')

        return df

    def _get_table_name(self, sample_id: str, dataset: str) -> str:
        """
        Helper method to retrieve the table name for a given sample ID.
        """
        data = self.crossval_data if dataset == 'crossval' else self.holdout_data
        for table_name, table_info in data.items():
            if sample_id in table_info['data'].index:
                return table_name
        return "Unknown"
