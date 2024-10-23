from abc import ABC, abstractmethod
import pickle
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame

# src/tidyML/experimentTracker.py

import os
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from io import BytesIO, StringIO

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure
from plotly.graph_objs._figure import Figure as PlotlyFigure
from plotly.io import to_html

# Import necessary neptune utility functions if any
import neptune.integrations.sklearn as npt_utils
from neptune.utils import stringify_unsupported


class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(self, projectID: str, entityID: str, analysisName: str, **kwargs):
        self.entityID = entityID
        self.projectID = projectID
        self.analysisName = analysisName

    @abstractmethod
    def start(self, model):
        """
        Initialize tracker with a given model.
        """
        pass

    @abstractmethod
    def summarize(
        self,
        model,
        trainingData: pd.DataFrame,
        testingData: pd.DataFrame,
        trainingLabels: pd.Series,
        testingLabels: pd.Series,
        **kwargs,
    ):
        """
        Generate classifier summary.
        """
        pass

    @abstractmethod
    def log(
        self,
        path: str,
        valueMap: Dict,
        projectLevel: bool = False,
        **kwargs,
    ):
        """
        Log a dictionary of values to a specific path.
        """
        pass

    @abstractmethod
    def addTags(self, tags: List[str]):
        """
        Append tags to the current tracking run.
        """
        pass

    @abstractmethod
    def getRuns(
        self,
        runID: Union[List[str], str] = None,
        tag: Union[List[str], str] = None,
    ) -> pd.DataFrame:
        """
        Fetch the latest runs by ID or tag. All runs are fetched by default.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Send halt signal to experiment tracker and avoid memory leaks.
        """
        pass


class NeptuneExperimentTracker(ExperimentTracker):
    """
    Interface for experiment tracking using Neptune.
    """

    def __init__(self, projectID: str, entityID: str, analysisName: str, **kwargs):
        super().__init__(projectID, entityID, analysisName, **kwargs)
        self.apiToken = kwargs.get("apiToken", os.getenv("NEPTUNE_API_TOKEN"))

        if not self.apiToken:
            raise ValueError("API Token for Neptune is required.")

        self.project = None
        self.tracker = None
        self.runs = None

    def loadProject(self):
        """
        Initialize and load the Neptune project.
        """
        self.project = neptune.init_project(
            name=f"{self.entityID}/{self.projectID}",
            api_token=self.apiToken
        )

    def start(self, model):
        """
        Start a new Neptune run for the given model.
        """
        self.tracker = neptune.init(
            project=f"{self.entityID}/{self.projectID}",
            api_token=self.apiToken,
            name=self.analysisName,
            capture_hardware_metrics=True,
        )
        self.addTags([model.__class__.__name__])
        self.model = model

    def summarize(
        self,
        model,
        trainingData: pd.DataFrame,
        testingData: pd.DataFrame,
        trainingLabels: pd.Series,
        testingLabels: pd.Series,
        **kwargs,
    ):
        """
        Generate and log classifier summary to Neptune.
        """
        summary = npt_utils.create_classifier_summary(
            model, trainingData, testingData, trainingLabels, testingLabels
        )
        self.tracker["summary"].upload(File.as_html(summary))
        self.tracker["model/params"] = stringify_unsupported(npt_utils.get_estimator_params(model))

    def log(
        self,
        path: str,
        valueMap: Dict,
        projectLevel: bool = False,
        **kwargs,
    ):
        """
        Log a dictionary of values to a specific path in Neptune.
        """
        loggable = self.project if projectLevel else self.tracker

        for key, value in valueMap.items():
            existingData = None
            try:
                if f"{path}/{key}" in loggable:
                    # Reset data type by erasing existing entry
                    existingData = loggable[f"{path}/{key}"]
                    del loggable[f"{path}/{key}"]

                if isinstance(value, Figure):
                    with BytesIO() as fileHandle:
                        value.savefig(fileHandle, format="svg", bbox_inches="tight")
                        fileHandle.seek(0)
                        loggable[f"{path}/{key}"].upload(File.from_stream(fileHandle, extension="svg"))

                    with BytesIO() as previewHandle:
                        value.savefig(previewHandle, format="png", bbox_inches="tight")
                        previewHandle.seek(0)
                        loggable[f"{path}/{key} preview"].upload(File.from_stream(previewHandle, extension="png"))
                elif isinstance(value, PlotlyFigure):
                    html_content = to_html(value)
                    loggable[f"{path}/{key}"].upload(File.from_content(html_content, extension="html"))
                elif isinstance(value, pd.DataFrame):
                    csv_buffer = StringIO()
                    value.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    loggable[f"{path}/{key}"].upload(File.from_stream(csv_buffer, extension="csv"))
                elif key == "model":
                    pickle_buffer = BytesIO()
                    pickle.dump(value, pickle_buffer)
                    pickle_buffer.seek(0)
                    loggable[f"{path}/{key}"].upload(File.from_stream(pickle_buffer, extension="pkl"))
                elif isinstance(value, (BytesIO, StringIO)):
                    value.seek(0)
                    loggable[f"{path}/{key}"].upload(File.from_stream(value))
                else:
                    # For simple types (int, float, str, etc.)
                    loggable[f"{path}/{key}"] = value
            except Exception as e:
                if existingData:
                    # Reset existing data upon error
                    loggable[f"{path}/{key}"] = existingData
                # Log the exception
                loggable[f"{path}/logging_errors"].log_text(f"Failed to log {key} at {path}: {str(e)}")

    def addTags(self, tags: List[str]):
        """
        Append tags to the current tracking run.
        """
        if self.tracker:
            self.tracker["sys/tags"].add(tags)
        elif self.project:
            self.project["sys/tags"].add(tags)
        else:
            raise ValueError("No active Neptune run or project to add tags to.")

    def getRuns(
        self,
        runID: Union[List[str], str] = None,
        tag: Union[List[str], str] = None,
    ) -> pd.DataFrame:
        """
        Fetch runs from the Neptune project, optionally filtering by runID or tag.
        """
        if not self.project:
            self.loadProject()

        query = ""
        if runID:
            if isinstance(runID, list):
                run_ids = " OR ".join([f'sys/id = "{rid}"' for rid in runID])
                query += f'({run_ids})'
            else:
                query += f'sys/id = "{runID}"'

        if tag:
            tag_filter = f'sys/tags CONTAINS "{tag}"' if isinstance(tag, str) else ' OR '.join([f'sys/tags CONTAINS "{t}"' for t in tag])
            if query:
                query += f' AND {tag_filter}'
            else:
                query += f'{tag_filter}'

        try:
            runs_table = self.project.fetch_runs_table(filter=query).to_pandas()
            self.runs = runs_table
            return runs_table
        except Exception as e:
            raise RuntimeError(f"Failed to fetch runs with filter '{query}': {str(e)}")

    def stop(self):
        """
        Stop the active Neptune run or project.
        """
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
        if self.project:
            self.project.stop()
            self.project = None

class MLflowTracker(ExperimentTracker):
    """
    MLflow implementation of the Tracker interface.
    """

    def __init__(self, projectID: str, entityID: str, analysisName: str, tracking_uri: str = "http://localhost:5000", experiment_name: str = "Default_Experiment", **kwargs):
        super().__init__(projectID, entityID, analysisName, **kwargs)
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.current_run = None

    def start(self, model: Any) -> None:
        """
        Start a new MLflow tracking run.

        Args:
            model (Any): The model to be tracked.
        """
        self.current_run = self.mlflow.start_run(run_name=self.analysisName)
        self.add_tags([model.__class__.__name__])

    def end_run(self) -> None:
        """
        End the current MLflow tracking run.
        """
        if self.current_run is not None:
            self.mlflow.end_run()
            self.current_run = None

    def log(self, path: str, valueMap: Dict[str, Any], projectLevel: bool = False) -> None:
        """
        Log a dictionary of values to MLflow.

        Args:
            path (str): The path where the values will be logged. (Not directly applicable in MLflow)
            valueMap (Dict[str, Any]): A dictionary of values to log.
            projectLevel (bool): Whether to log at the project level. MLflow does not support project-level logging. Ignored.
        """
        if self.current_run is not None:
            for key, value in valueMap.items():
                if isinstance(value, (int, float, str)):
                    self.mlflow.log_param(key, value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.mlflow.log_metric(f"{key}.{sub_key}", sub_value)
                elif isinstance(value, DataFrame):
                    self.mlflow.log_artifact(value.to_csv(), artifact_path=path)
                else:
                    self.mlflow.log_param(key, str(value))
        else:
            raise RuntimeError("MLflow run not started. Call start() before logging.")

    def add_tags(self, tags: List[str]) -> None:
        """
        Append tags to the current MLflow tracking run.

        Args:
            tags (List[str]): A list of tags to add.
        """
        if self.current_run is not None:
            for tag in tags:
                self.mlflow.set_tag(tag, True)
        else:
            raise RuntimeError("MLflow run not started. Call start() before adding tags.")

    def get_runs(self, runID: Union[List[str], str] = None, tag: Union[List[str], str] = None) -> pd.DataFrame:
        """
        Fetch runs from the MLflow experiment, optionally filtering by runID or tag.

        Args:
            runID (Union[List[str], str], optional): Specific run IDs to filter. Defaults to None.
            tag (Union[List[str], str], optional): Specific tags to filter. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched runs.
        """
        client = self.mlflow.tracking.MlflowClient()
        experiment = self.mlflow.get_experiment_by_name(self.mlflow.get_experiment_info().name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=self._construct_filter(runID, tag),
            run_view_type=self.mlflow.entities.ViewType.ACTIVE_ONLY
        )
        runs_data = []
        for run in runs:
            run_dict = {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri
            }
            run_dict.update(run.data.params)
            run_dict.update(run.data.metrics)
            runs_data.append(run_dict)
        return pd.DataFrame(runs_data)

    def _construct_filter(self, runID: Union[List[str], str], tag: Union[List[str], str]) -> str:
        """
        Construct MLflow filter string based on runID and tag.

        Args:
            runID (Union[List[str], str]): Run IDs to filter.
            tag (Union[List[str], str]): Tags to filter.

        Returns:
            str: The constructed filter string.
        """
        filters = []
        if runID:
            if isinstance(runID, list):
                run_filters = " OR ".join([f"tags.mlflow.runName = '{rid}'" for rid in runID])
                filters.append(f"({run_filters})")
            else:
                filters.append(f"tags.mlflow.runName = '{runID}'")
        if tag:
            if isinstance(tag, list):
                tag_filters = " OR ".join([f"tags.{t} = 'true'" for t in tag])
                filters.append(f"({tag_filters})")
            else:
                filters.append(f"tags.{tag} = 'true'")
        return " AND ".join(filters) if filters else ""

    def stop(self) -> None:
        """
        Stop the MLflow tracking run.
        """
        self.end_run()