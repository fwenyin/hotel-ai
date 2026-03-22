"""
Hyperparameter tuning module using Optuna.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from .lightgbm_model import LightGBMModel
from .neural_network_model import NeuralNetworkModel
from .random_forest_model import RandomForestModel
from .tabnet_model import TabNetModel
from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuners.

    Each model type has its own tuner that defines the search space
    and objective function. Follows Strategy pattern.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        timeout: Optional[int] = None,
    ):
        """Initialize tuner.

        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            timeout: Maximum time in seconds (None = no limit)
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = 42
        self.timeout = timeout
        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0

    @abstractmethod
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters to try
        """
        pass

    @abstractmethod
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create model instance with given parameters.

        Args:
            params: Hyperparameters dictionary

        Returns:
            Model instance
        """
        pass

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object
            X: Features
            y: Labels

        Returns:
            Cross-validation score
        """
        params = self._get_param_space(trial)
        model = self._create_model(params)
        scores = cross_val_score(
            model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=1
        )
        return scores.mean()

    def tune(
        self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print progress

        Returns:
            Best hyperparameters found
        """
        if verbose:
            print(f"Hyperparameter Tuning: {self.__class__.__name__}")
            print(
                f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}, Metric: {self.scoring}"
            )

        self.study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_state)
        )

        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose,
            n_jobs=1,
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        if verbose:
            print(f"\n✅ Best {self.scoring}: {self.best_score:.3f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                print(f"  - {param}: {value}")
        return self.best_params


class RandomForestTuner(HyperparameterTuner):
    """Hyperparameter tuner for Random Forest."""

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Random Forest search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.8]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
        }

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create Random Forest model using wrapper class.

        Args:
            params: Hyperparameters suggested by Optuna

        Returns:
            sklearn RandomForestClassifier instance
        """
        model = RandomForestModel(params).model
        model.n_jobs = 1
        return model


class XGBoostTuner(HyperparameterTuner):
    """Hyperparameter tuner for XGBoost."""

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define XGBoost search space with improved ranges."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create XGBoost model using wrapper class.

        Args:
            params: Hyperparameters suggested by Optuna

        Returns:
            xgboost XGBClassifier instance
        """
        model = XGBoostModel(params).model
        model.n_jobs = 1
        return model


class LightGBMTuner(HyperparameterTuner):
    """Hyperparameter tuner for LightGBM."""

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define LightGBM search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create LightGBM model using wrapper class.

        Args:
            params: Hyperparameters suggested by Optuna

        Returns:
            lightgbm LGBMClassifier instance
        """
        model = LightGBMModel(params).model
        model.class_weight = "balanced"
        model.n_jobs = 1
        return model


class NeuralNetworkTuner(HyperparameterTuner):
    """Hyperparameter tuner for Neural Network (MLP)."""

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Neural Network search space."""
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_layers = []
        for i in range(n_layers):
            units = trial.suggest_int(f"n_units_l{i}", 32, 512)
            hidden_layers.append(units)

        return {
            "hidden_layers": hidden_layers,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create Neural Network model using existing model class."""
        full_params = {"epochs": 300, "early_stopping_patience": 15, **params}
        return NeuralNetworkModel(full_params).model


class TabNetTuner(HyperparameterTuner):
    """Hyperparameter tuner for TabNet.

    Note: Uses PyTorch-TabNet with custom objective function.
    """

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define TabNet search space."""
        return {
            "n_d": trial.suggest_int("n_d", 16, 64),
            "n_a": trial.suggest_int("n_a", 16, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 8),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            "mask_type": trial.suggest_categorical(
                "mask_type", ["sparsemax", "entmax"]
            ),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        }

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create TabNet model configuration."""
        return params

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Custom objective for PyTorch TabNet."""
        params = self._get_param_space(trial)

        params["max_epochs"] = 50
        params["patience"] = 10
        params["virtual_batch_size"] = 128

        # subsample for speed
        if len(X) > 30000:
            X_sample = X.sample(n=30000, random_state=42)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample, y_sample = X, y

        X_train, X_val, y_train, y_val = train_test_split(
            X_sample,
            y_sample,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_sample,
        )

        try:
            model = TabNetModel(params)
            model.fit(X_train, y_train, X_val, y_val)

            y_pred_proba = model.predict_proba(X_val)
            score = roc_auc_score(y_val, y_pred_proba)

            return score
        except Exception as e:
            logger.warning(f"TabNet trial failed: {e}")
            return 0.0


class TunerFactory:
    """Factory for creating hyperparameter tuners.

    Follows Factory pattern for consistent tuner creation.
    """

    _tuners: Dict[str, Type[HyperparameterTuner]] = {
        "random_forest": RandomForestTuner,
        "xgboost": XGBoostTuner,
        "lightgbm": LightGBMTuner,
        "neural_network": NeuralNetworkTuner,
        "tabnet": TabNetTuner,
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        **kwargs,
    ) -> HyperparameterTuner:
        """Create a tuner instance.

        Args:
            model_type: Type of model to tune
            n_trials: Number of Optuna trials
            cv_folds: Number of CV folds
            scoring: Scoring metric
            **kwargs: Additional tuner arguments

        Returns:
            Tuner instance
        """
        if model_type not in cls._tuners:
            available = ", ".join(cls._tuners.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. " f"Available: {available}"
            )

        tuner_class = cls._tuners[model_type]
        return tuner_class(
            n_trials=n_trials, cv_folds=cv_folds, scoring=scoring, **kwargs
        )
