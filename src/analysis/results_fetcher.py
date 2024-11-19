import logging
from typing import List, Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from urllib.parse import urlparse
from mlflow.exceptions import MlflowException

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

from config import TrackingConfig
from eval.evaluation import sanitize_mlflow_name


class ResultFetcher(ABC):
    """
    Abstract base class for fetching experiment results from different tracking frameworks.
    """

    @abstractmethod
    def fetch_all_run_metadata(
        self, run_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all runs from the experiment, optionally filtering by run type.

        Args:
            run_type (str, optional): Type of runs to fetch (e.g., 'model', 'bootstrap', 'feature_selection').

        Returns:
            List of runs with their details.
        """
        pass

    @abstractmethod
    def get_run_metrics(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch metrics for a specific run.

        Args:
            run: A dictionary containing run details.

        Returns:
            Dictionary of metrics.
        """
        pass

    @abstractmethod
    def download_run_artifacts(self, run: Dict[str, Any], artifact_paths: List[str]) -> Dict[str, Any]:
        """
        Fetch artifacts for a specific run.

        Args:
            run: A dictionary containing run details.
            artifact_paths: List of artifact file paths to fetch.

        Returns:
            Dictionary of artifacts with their contents.
        """
        pass

    @abstractmethod
    def get_run_params(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch parameters for a specific run.

        Args:
            run: A dictionary containing run details.

        Returns:
            Dictionary of parameters.
        """
        pass

    @abstractmethod
    def fetch_runs_parallel(self, runs: List[Dict[str, Any]], artifact_paths: List[str], max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        Fetch data for all runs in parallel.

        Args:
            runs: List of runs to fetch data for.
            artifact_paths: List of artifact file paths to fetch.
            max_workers: Number of parallel threads.

        Returns:
            List of dictionaries containing run data.
        """
        pass


class MLflowResultFetcher(ResultFetcher):
    """
    MLflow implementation of the ResultFetcher interface.
    """

    def __init__(
        self, 
        experiment_name: str, 
        tracking_uri: str = None, 
        logger: logging.Logger = None,
        mlartifacts_base_path: str = None
    ):
        self.client = MlflowClient(tracking_uri)
        self.experiment = self.client.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' does not exist.")
        self.experiment_id = self.experiment.experiment_id
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger(__name__)
        self.mlartifacts_base_path = mlartifacts_base_path

    def _extract_run_type(self, run_name: str) -> Optional[str]:
        """
        Extract run type from the run name based on predefined prefixes.

        Args:
            run_name (str): The name of the run.

        Returns:
            str or None: Extracted run type or None if not matched.
        """
        if run_name.startswith("Bootstrap_"):
            return "bootstrap"
        elif run_name.startswith("Feature_Selection_"):
            return "feature_selection"
        elif run_name.startswith("Outer_Bootstrap_"):
            return "outer_bootstrap"
        elif run_name.startswith("summary"):
            return "summary"
        else:
            return "model"

    def fetch_all_run_metadata(
        self, run_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all runs from the experiment, optionally filtering by run type.

        Args:
            run_type (str, optional): Type of runs to fetch ('model', 'bootstrap', 'feature_selection', 'outer_bootstrap').

        Returns:
            List of runs with their details.
        """
        all_runs = []
        page_token = None
        filter_string = None

        # Map simplified run types to actual run name prefixes
        run_type_map = {
            'model': 'model',
            'bootstrap': 'Bootstrap_',
            'feature_selection': 'Feature_Selection_',
            'outer_bootstrap': 'outer_bootstrap_',
            'summary': 'summary'
        }

        if run_type:
            if run_type not in run_type_map:
                raise ValueError(f"Invalid run_type: {run_type}. Must be 'model', 'bootstrap', 'feature_selection', 'outer_bootstrap', or 'summary'.")

            if run_type == 'summary':
                filter_string = "tags.mlflow.runName = 'summary'"
            elif run_type in ['bootstrap', 'feature_selection', 'outer_bootstrap']:
                prefix = run_type_map[run_type]
                filter_string = f"tags.mlflow.runName LIKE '{prefix}%'"
            elif run_type == 'model':
                # For 'model' runs, exclude bootstrap and feature selection runs
                filter_string = "tags.mlflow.runName != 'Bootstrap_%' AND tags.mlflow.runName != 'Feature_Selection_%' AND tags.mlflow.runName != 'outer_bootstrap_%' AND tags.mlflow.runName != 'summary'"

        self.logger.info(f"Fetching runs for experiment {self.experiment.name} with filter: {filter_string}")

        while True:
            try:
                runs_page = self.client.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=filter_string,
                    max_results=10000,     # Use a safe value below the server threshold
                    page_token=page_token
                )
                
                all_runs.extend(runs_page)
                
                if runs_page.token is None:
                    break
                
                page_token = runs_page.token
                
            except MlflowException as e:
                self.logger.error(f"Error fetching runs: {e}")
                break

        processed_runs = []
        for run in all_runs:
            run_name = run.data.tags.get("mlflow.runName", "Unknown")
            extracted_run_type = self._extract_run_type(run_name)
            
            # If run_type is 'model', exclude if it's 'bootstrap' or 'feature_selection'
            if run_type == 'model' and extracted_run_type in ['bootstrap', 'feature_selection', 'outer_bootstrap']:
                continue  # Skip non-model runs

            run_dict = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "run_type": extracted_run_type,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
            processed_runs.append(run_dict)

        self.logger.info(f"Fetched {len(processed_runs)} runs.")
        return processed_runs

    def get_run_metrics(self, run) -> Dict[str, Any]:
        if isinstance(run, dict):
            return run.get('metrics', {})
        return {key: value for key, value in run.data.metrics.items()}

    def _fetch_single_artifact(self, run_id: str, artifact_path: str) -> Any:
        tracking_uri = mlflow.get_tracking_uri()
        is_local = urlparse(tracking_uri).scheme in ('', 'file', 'localhost', '127.0.0.1')

        try:
            if is_local and self.mlartifacts_base_path:
                local_path = os.path.join(self.mlartifacts_base_path, self.experiment_id, run_id, 'artifacts', sanitize_mlflow_name(artifact_path))
            else:
                local_path = self.client.download_artifacts(run_id, sanitize_mlflow_name(artifact_path), "/tmp")
            
            if os.path.exists(local_path):
                if artifact_path.endswith(".json"):
                    return pd.read_json(local_path, orient="split")
                elif artifact_path.endswith(".csv"):
                    return pd.read_csv(local_path)
                else:
                    return local_path
            else:
                self.logger.warning(f"Artifact not found: {local_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching artifact '{artifact_path}' for run '{run_id}': {e}")
            return None

    def download_run_artifacts(self, run, artifact_paths: List[str]) -> Dict[str, Any]:
        run_id = run['run_id'] if isinstance(run, dict) else run.info.run_id
        
        with ThreadPoolExecutor(max_workers=min(len(artifact_paths), 8)) as executor:
            futures = [executor.submit(self._fetch_single_artifact, run_id, path) for path in artifact_paths]
            artifacts = {}
            for future, path in zip(futures, artifact_paths):
                try:
                    result = future.result()
                    artifacts[sanitize_mlflow_name(path)] = result
                except Exception as e:
                    self.logger.error(f"Failed to retrieve artifact '{sanitize_mlflow_name(path)}' for run '{run_id}': {e}")
                    artifacts[sanitize_mlflow_name(path)] = None
        return artifacts

    def get_run_params(self, run) -> Dict[str, Any]:
        if isinstance(run, dict):
            return run.get('params', {})
        return {key: value for key, value in run.data.params.items()}

    def fetch_runs_parallel(
        self,
        runs: List[Dict[str, Any]],
        artifact_paths: List[str],
        max_workers: int = 8,
    ) -> List[Dict[str, Any]]:
        run_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._fetch_single_run, run, [sanitize_mlflow_name(artifact) for artifact in artifact_paths])
                for run in runs
            ]
            for future in tqdm(futures, desc="Fetching runs"):
                try:
                    data = future.result()
                    run_data.append(data)
                except Exception as e:
                    self.logger.error(f"Error fetching run data: {e}")
                    raise e
        return run_data

    def _fetch_single_run(
        self, run: Dict[str, Any], artifact_paths: List[str]
    ) -> Dict[str, Any]:
        run_id = run["run_id"]
        run_name = run["run_name"]
        metrics = self.get_run_metrics(run)
        artifacts = self.download_run_artifacts(run, artifact_paths)
        params = self.get_run_params(run)
        return {
            "run_id": run_id,
            "run_name": run_name,
            "metrics": metrics,
            "artifacts": artifacts,
            "params": params,
            "run_type": run.get("run_type")
        }
    
    def _is_complete_run(self, run) -> bool:
        """
        Check if a run is complete based on the presence of metrics and required artifacts.
        """
        required_artifacts = [
            'train/confusion_matrix.json',
            'train/roc_curve.json',
            'test/confusion_matrix.json',
            'test/roc_curve.json',
            'test_sample_metrics.json',
            'train_sample_metrics.json'
        ]
        
        return (
            bool(run['metrics']) and 
            all(run['artifacts'].get(sanitize_mlflow_name(artifact)) is not None for artifact in required_artifacts)
        )

    def sanitize_run_data(self, run_data):
        """
        Filter out incomplete runs from the run_data.
        """
        complete_runs = [run for run in run_data if self._is_complete_run(run)]
        return complete_runs