"""MLflow Experiment Tracking Module."""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.models.model_evaluator import (
    compute_confusion_matrix,
    format_feature_importance,
)

logger = logging.getLogger(__name__)


def get_or_create_experiment(
    experiment_name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Get existing or create new MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        artifact_location: Optional custom artifact storage path
        tags: Optional experiment-level tags

    Returns:
        Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location, tags=tags
        )
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Using existing experiment: {experiment_name} (ID: {experiment_id})"
        )
    return experiment_id


class MLflowTracker:
    """MLflow tracking wrapper with support for nested runs and model comparison."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
            artifact_location: Custom artifact storage path
            tags: Experiment-level tags
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_id = get_or_create_experiment(
            experiment_name, artifact_location, tags
        )
        mlflow.set_experiment(experiment_name)
        self.active_run = None
        self.client = MlflowClient()
        logger.info(f"MLflow Tracker initialized for experiment: {experiment_name}")

    @contextmanager
    def start_run(
        self,
        run_name: str,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = {},
        description: Optional[str] = None,
    ):
        """Context manager for starting an MLflow run.

        Args:
            run_name: Name for this run
            nested: Whether this is a nested run (for hyperparameter tuning)
            tags: Run-level tags
            description: Run description

        Yields:
            Self for method chaining
        """
        tags["timestamp"] = datetime.now().isoformat()
        if description:
            tags["mlflow.note.content"] = description

        try:
            self.active_run = mlflow.start_run(
                run_name=run_name,
                experiment_id=self.experiment_id,
                nested=nested,
                tags=tags,
            )
            logger.info(
                f"Started MLflow run: {run_name} (ID: {self.active_run.info.run_id})"
            )
            yield self
        finally:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {run_name}")
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Handles nested dictionaries by flattening with dot notation.

        Args:
            params: Dictionary of parameters to log
        """
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_metrics(
        self, metrics: Dict[str, float], prefix: Optional[str] = ""
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name -> value
            prefix: Optional prefix for metric name
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_name = f"{prefix}{name}"
                mlflow.log_metric(metric_name, float(value))
        logger.debug(f"Logged metrics with prefix '{prefix}'")

    def log_model(
        self,
        model,
        artifact_path: str,
        model_type: str = "sklearn",
        X_sample: Optional[pd.DataFrame] = None,
    ) -> None:
        """Log a model to MLflow with automatic framework detection.

        Args:
            model: Trained model instance
            artifact_path: Path within artifacts for model
            model_type: Framework type ('sklearn', 'xgboost', 'lightgbm', 'pytorch')
            X_sample: Sample input for signature inference
        """
        log_functions = {
            "sklearn": mlflow.sklearn.log_model,
            "xgboost": mlflow.xgboost.log_model,
            "lightgbm": mlflow.lightgbm.log_model,
            "pytorch": mlflow.pytorch.log_model,
        }

        if model_type not in log_functions:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Supported: {list(log_functions.keys())}"
            )

        signature = None
        input_example = None

        if X_sample is not None:
            y_pred = model.predict(X_sample.head())
            signature = infer_signature(X_sample.head(), y_pred)
            input_example = X_sample.head(1).to_dict(orient="records")[0]

        log_functions[model_type](
            model, artifact_path, signature=signature, input_example=input_example
        )
        logger.info(f"Logged {model_type} model to: {artifact_path}")

    def log_feature_importance(
        self, feature_names: List[str], importances: np.ndarray
    ) -> None:
        """Log feature importance as both JSON data and plot.

        Args:
            feature_names: List of feature names
            importances: Array of importance values
        """
        importance_df = format_feature_importance(feature_names, importances)
        mlflow.log_table(importance_df, "feature_importance.json")

    def log_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None
    ) -> None:
        """Log confusion matrix as artifact.

        Computes confusion matrix using model_evaluator, then logs to MLflow.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional class labels
        """
        cm_dict = compute_confusion_matrix(y_true, y_pred, labels)
        mlflow.log_dict(cm_dict, "confusion_matrix.json")

    def get_run_id(self) -> Optional[str]:
        if self.active_run:
            return self.active_run.info.run_id
        return None

    def get_artifact_uri(self) -> Optional[str]:
        if self.active_run:
            return self.active_run.info.artifact_uri
        return None

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator between nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """Search for runs in the experiment.

        Args:
            filter_string: MLflow filter query
            order_by: List of columns to sort by

        Returns:
            DataFrame of matching runs
        """
        return mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["metrics.roc_auc DESC"],
            max_results=max_results,
        )
