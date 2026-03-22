"""MLflow-integrated model training pipeline."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.hyperparameter_tuner import TunerFactory
from src.models.model_trainer import ModelFactory

from .mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class MLflowModelPipeline:
    """End-to-end ML pipeline with MLflow experiment tracking."""

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str = "hotel_noshow_prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """Initialize MLflow pipeline.

        Args:
            config: Full pipeline configuration
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            artifact_location: Custom artifact storage path
        """
        self.config = config
        self.experiment_name = experiment_name
        self.tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            tags={
                "project": "hotel_noshow_prediction",
                "version": config["general"]["version"],
                "environment": config["general"]["environment"],
            },
        )
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.run_ids: Dict[str, str] = {}

        logger.info(f"MLflow Pipeline initialized: {experiment_name}")

    def run_experiment(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_types: Optional[List[str]] = [
            "random_forest",
            "xgboost",
            "lightgbm",
            "neural_network",
        ],
        tune_hyperparameters: bool = False,
        n_trials: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Run complete experiment with MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_types: List of models to train
            tune_hyperparameters: Whether to run hyperparameter tuning
            n_trials: Number of tuning trials per model

        Returns:
            Dictionary of results per model
        """
        parent_run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.tracker.start_run(
            run_name=parent_run_name,
            description=f"Training {len(model_types)} models: {', '.join(model_types)}",
            tags={
                "experiment_type": "model_comparison",
                "tune_hyperparameters": str(tune_hyperparameters),
                "n_models": str(len(model_types)),
            },
        ):
            self._log_experiment_params(
                X_train, y_train, X_val, y_val, model_types, tune_hyperparameters
            )

            for i, model_type in enumerate(model_types, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(model_types)}] Training {model_type.upper()}")
                print(f"{'='*60}")
                if tune_hyperparameters:
                    self._train_with_tuning(
                        model_type, X_train, y_train, X_val, y_val, n_trials
                    )
                else:
                    self._train_model(model_type, X_train, y_train, X_val, y_val)

            self._log_comparison_metrics()
            self._log_experiment_summary()

        return self.results

    def _log_experiment_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_types: List[str],
        tune_hyperparameters: bool,
    ) -> None:
        """Log experiment-level parameters."""
        params = {
            "data.train_samples": len(X_train),
            "data.val_samples": len(X_val),
            "data.n_features": len(X_train.columns),
            "data.train_positive_rate": float(y_train.mean()),
            "data.val_positive_rate": float(y_val.mean()),
            "experiment.models": ", ".join(model_types),
            "experiment.n_models": len(model_types),
            "experiment.tune_hyperparameters": tune_hyperparameters,
            "config.random_state": 42,
        }
        self.tracker.log_params(params)

    def _train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Train a single model with MLflow tracking."""
        model_config = self.config["models"][model_type]

        with self.tracker.start_run(
            run_name=f"{model_type}",
            nested=True,
            tags={"model_type": model_type, "tuned": "false"},
        ):
            self.tracker.log_params({f"param.{k}": v for k, v in model_config.items()})

            model = ModelFactory.create(model_type, model_config)

            if model_type == "tabnet":
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)

            results = model.evaluate(X_val, y_val)
            self._log_model_metrics(results)
            self._log_feature_importance(model)
            y_pred = model.predict(X_val)
            self.tracker.log_confusion_matrix(y_val.values, y_pred)
            self._log_model_artifact(model, model_type, X_train)
            self.run_ids[model_type] = self.tracker.get_run_id()
            self.models[model_type] = model
            self.results[model_type] = results

            model.print_results(results)
            return results

    def _train_with_tuning(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 30,
    ) -> Dict[str, Any]:
        """Train model with hyperparameter tuning."""
        with self.tracker.start_run(
            run_name=f"{model_type}_tuned",
            nested=True,
            tags={"model_type": model_type, "tuned": "true", "n_trials": str(n_trials)},
        ):
            tuner = TunerFactory.create(
                model_type, n_trials=n_trials, cv_folds=5, scoring="roc_auc"
            )

            best_params = tuner.tune(X_train, y_train, verbose=True)
            self.tracker.log_params({f"tuned.{k}": v for k, v in best_params.items()})
            self.tracker.log_metrics(
                {"tuning_best_cv_score": tuner.best_score, "tuning_n_trials": n_trials}
            )

            base_config = self.config["models"][model_type].copy()
            base_config.update(best_params)

            model = ModelFactory.create(model_type, base_config)

            if model_type == "tabnet":
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)

            results = model.evaluate(X_val, y_val)
            results["tuned"] = True
            results["tuned_params"] = best_params
            self._log_model_metrics(results)
            self._log_feature_importance(model)
            y_pred = model.predict(X_val)
            self.tracker.log_confusion_matrix(y_val.values, y_pred)
            self._log_model_artifact(model, model_type, X_train)
            self.run_ids[model_type] = self.tracker.get_run_id()
            self.models[model_type] = model
            self.results[model_type] = results

            model.print_results(results)
            return results

    def _log_model_metrics(self, results: Dict[str, Any]) -> None:
        """Log model evaluation metrics."""
        classification_metrics = {
            "accuracy": results.get("accuracy"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "f1_score": results.get("f1_score"),
            "roc_auc": results.get("roc_auc"),
            "pr_auc": results.get("pr_auc"),
        }
        self.tracker.log_metrics(classification_metrics)

        if "cv_roc_auc_mean" in results:
            cv_metrics = {
                "cv_roc_auc_mean": results["cv_roc_auc_mean"],
                "cv_roc_auc_std": results["cv_roc_auc_std"],
            }
            self.tracker.log_metrics(cv_metrics)

    def _log_feature_importance(self, model: BaseModel) -> None:
        """Log feature importance if available."""
        importance = model.get_feature_importance()

        if importance:
            names = list(importance.keys())
            values = np.array(list(importance.values()))
            self.tracker.log_feature_importance(names, values)

    def _log_model_artifact(
        self, model: BaseModel, model_type: str, X_sample: pd.DataFrame
    ) -> None:
        """Log model artifact with the appropriate MLflow flavor."""
        try:
            sample = X_sample.head(5)
            if model_type == "tabnet":
                self.tracker.log_model(
                    model, "model", model_type="pytorch", X_sample=sample
                )
                logger.info(f"Logged {model_type} model (PyTorch)")

            elif model_type in ["xgboost"]:
                sklearn_model = getattr(model, "model", None)
                if sklearn_model:
                    self.tracker.log_model(
                        sklearn_model, "model", model_type="xgboost", X_sample=sample
                    )
                    logger.info(f"Logged {model_type} model (XGBoost)")

            elif model_type in ["lightgbm"]:
                sklearn_model = getattr(model, "model", None)
                if sklearn_model:
                    self.tracker.log_model(
                        sklearn_model, "model", model_type="lightgbm", X_sample=sample
                    )
                    logger.info(f"Logged {model_type} model (LightGBM)")

            else:
                sklearn_model = getattr(model, "model", None)
                if sklearn_model:
                    self.tracker.log_model(
                        sklearn_model, "model", model_type="sklearn", X_sample=sample
                    )
                    logger.info(f"Logged {model_type} model (Sklearn)")

        except Exception as e:
            logger.warning(f"Could not log model artifact for {model_type}: {e}")

    def _log_comparison_metrics(self) -> None:
        """Log comparison metrics across all models."""
        if not self.results:
            return

        metrics_to_compare = ["roc_auc", "accuracy", "precision", "recall", "f1_score"]

        for metric in metrics_to_compare:
            try:
                best_model = max(
                    self.results.keys(),
                    key=lambda m: self.results[m].get(metric, 0) or 0,
                )
                best_value = self.results[best_model].get(metric, 0)
                self.tracker.log_metrics({f"best_{metric}": best_value})
                mlflow.set_tag(f"best_{metric}_model", best_model)
            except Exception:
                pass

    def _log_experiment_summary(self) -> None:
        """Log experiment summary as artifact."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "n_models_trained": len(self.results),
            "models": list(self.results.keys()),
            "results": {},
        }

        for model_name, results in self.results.items():
            summary["results"][model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results.items()
                if k not in ["feature_importance", "optimal_threshold", "tuned_params"]
                and not isinstance(v, (dict, list, np.ndarray))
            }

        if self.results:
            best_model = max(
                self.results.keys(),
                key=lambda m: self.results[m].get("roc_auc", 0) or 0,
            )
            summary["best_model"] = {
                "name": best_model,
                "roc_auc": self.results[best_model].get("roc_auc"),
            }

        mlflow.log_dict(summary, "experiment_summary.json")


def run_mlflow_experiment(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    experiment_name: str = "hotel_noshow_prediction",
    model_types: Optional[List[str]] = None,
    tune: bool = False,
    artifact_location: Optional[str] = None,
) -> Tuple[MLflowModelPipeline, Dict[str, Any]]:
    """Convenience function to run MLflow experiment.

    Args:
        config: Pipeline configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        experiment_name: MLflow experiment name
        model_types: Models to train
        tune: Whether to tune hyperparameters
        artifact_location: Optional artifact storage path

    Returns:
        Tuple of (pipeline, results)
    """
    pipeline = MLflowModelPipeline(
        config, experiment_name, artifact_location=artifact_location
    )

    results = pipeline.run_experiment(
        X_train,
        y_train,
        X_val,
        y_val,
        model_types=model_types,
        tune_hyperparameters=tune,
    )

    return pipeline, results
