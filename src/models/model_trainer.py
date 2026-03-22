"""
Model factory and trainer orchestrator.

Provides factory pattern for model creation and orchestrates
training of multiple models with unified interface.
Includes Optuna hyperparameter tuning integration.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd

from .base_model import BaseModel
from .hyperparameter_tuner import TunerFactory
from .lightgbm_model import LightGBMModel
from .neural_network_model import NeuralNetworkModel
from .random_forest_model import RandomForestModel
from .tabnet_model import TabNetModel
from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances."""

    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "neural_network": NeuralNetworkModel,
        "tabnet": TabNetModel,
    }

    @classmethod
    def create(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type of model
            config: Model-specific configuration

        Returns:
            Instantiated model
        """
        model_class = cls._models[model_type]
        return model_class(config)


class ModelTrainer:
    """Orchestrates training and evaluation of multiple models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.

        Args:
            config: Full configuration dictionary with 'models' section
        """
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
        self.factory = ModelFactory()

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Train a single model.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Training {model_type} model...")

        model_config = self.config["models"][model_type]

        model = self.factory.create(model_type, model_config)
        model.fit(X_train, y_train)
        results = model.evaluate(X_val, y_val)

        self.models[model_type] = model
        self.results[model_type] = results
        model.print_results(results)
        return results

    def tune_and_train(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune hyperparameters with Optuna and train model.

        Args:
            model_type: Type of model to tune and train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            timeout: Maximum time in seconds

        Returns:
            Evaluation results dictionary with tuned parameters
        """
        logger.info(f"Tuning and training {model_type} model.")

        tuner = TunerFactory.create(
            model_type,
            n_trials=n_trials,
            cv_folds=cv_folds,
            scoring=scoring,
            timeout=timeout,
        )
        best_params = tuner.tune(X_train, y_train, verbose=True)
        self.tuning_results[model_type] = {
            "best_params": best_params,
            "best_score": tuner.best_score,
            "n_trials": n_trials,
        }
        model_config = self.config["models"].get(model_type, {}).copy()
        model_config.update(best_params)

        model = self.factory.create(model_type, model_config)
        model.fit(X_train, y_train)

        results = model.evaluate(X_val, y_val)
        results["tuned"] = True
        results["tuned_params"] = best_params
        results["tuning_best_cv_score"] = tuner.best_score

        self.models[model_type] = model
        self.results[model_type] = results
        model.print_results(results)
        return results

    def compare_models(self) -> pd.DataFrame:
        """Compare results across all trained models.

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        for model_name, results in self.results.items():
            row = {"Model": model_name}
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
            ]:
                if metric in results:
                    row[metric] = results[metric]
            if "net_value" in results:
                row["net_value"] = results["net_value"]
                row["value_per_booking"] = results["expected_value_per_booking"]
            if "cv_roc_auc_mean" in results:
                row["cv_roc_auc"] = (
                    f"{results['cv_roc_auc_mean']:.3f} ± {results['cv_roc_auc_std']:.3f}"
                )

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        logger.info("=" * 70)
        logger.info("model comparison")
        logger.info("=" * 70)
        logger.info("\n" + df.to_string(index=False))
        return df

    def save_models(self, output_dir: str = "models/") -> None:
        """Save all trained models.

        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)

        for model_name, model in self.models.items():
            filepath = output_path / f"{model_name}_model.pkl"
            model.save(str(filepath))
            logger.info(f"Saved {model_name} to {filepath}")

        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_trained": list(self.results.keys()),
                "total_models": len(self.results),
            },
            "models": self.results,
        }

        if self.results:
            best_name = max(
                self.results.keys(),
                key=lambda name: self.results[name].get("roc_auc", 0),
            )
            results_data["metadata"]["champion_model"] = best_name
            results_data["metadata"]["champion_roc_auc"] = float(
                self.results[best_name].get("roc_auc", 0)
            )

        results_file = output_path / "results.json"

        with open(results_file, "w") as f:
            json.dump(
                results_data,
                f,
                indent=2,
                default=lambda obj: (
                    float(obj) if isinstance(obj, (np.integer, np.floating)) else obj
                ),
            )
        logger.info(f"Saved results to {results_file}")
        logger.info(f"  - {len(self.results)} models saved")
        logger.info(
            f"  - Champion: {results_data['metadata'].get('champion_model', 'N/A')}"
        )
        logger.info(f"All models saved to {output_dir}")
