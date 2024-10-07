from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pandas import DataFrame

class Tracker(ABC):
    """
    Abstract base class for experiment tracking.
    Defines a standard interface for tracking methods.
    """

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new tracking run.

        Args:
            run_name (Optional[str]): Name of the run.
        """
        pass

    @abstractmethod
    def end_run(self) -> None:
        """
        End the current tracking run.
        """
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.

        Args:
            key (str): Parameter name.
            value (Any): Parameter value.
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.

        Args:
            params (Dict[str, Any]): Dictionary of parameters.
        """
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric.

        Args:
            key (str): Metric name.
            value (float): Metric value.
            step (Optional[int]): Step or epoch number.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics.
            step (Optional[int]): Step or epoch number.
        """
        pass

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> None:
        """
        Log a model artifact.

        Args:
            model (Any): The model object to log.
            artifact_path (str): The artifact path within the run.
        """
        pass

    @abstractmethod
    def log_table(self, key: str, value: Any) -> None:
        """
        Log a table.

        Args:
            key (str): Table name.
            value (Any): Table value.
        """
        pass

    @abstractmethod
    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a tag for the current run.

        Args:
            key (str): Tag name.
            value (Any): Tag value.
        """
        pass    

class MLflowTracker(Tracker):
    """
    MLflow implementation of the Tracker interface.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "Default_Experiment"):
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.current_run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        self.current_run = self.mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self.current_run is not None:
            self.mlflow.end_run()
            self.current_run = None

    def log_param(self, key: str, value: Any) -> None:
        if self.current_run is not None:
            self.mlflow.log_param(key, value)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging parameters.")

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.current_run is not None:
            self.mlflow.log_params(params)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging parameters.")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self.current_run is not None:
            self.mlflow.log_metric(key, value, step=step)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging metrics.")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.current_run is not None:
            self.mlflow.log_metrics(metrics, step=step)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging metrics.")

    def log_model(self, model: Any, artifact_path: str) -> None:
        if self.current_run is not None:
            self.mlflow.sklearn.log_model(model, artifact_path)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging models.")
        
    def log_table(self, key: str, value: DataFrame) -> None:
        if self.current_run is not None:
            self.mlflow.log_table(key, value)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before logging tables.")
        
    def set_tag(self, key: str, value: Any) -> None:
        if self.current_run is not None:
            self.mlflow.set_tag(key, value)
        else:
            raise RuntimeError("MLflow run not started. Call start_run() before setting tags.")
        