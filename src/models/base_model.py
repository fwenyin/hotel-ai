"""Abstract base class for all ML models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models.

    This class defines the common interface and shared functionality
    for all model implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize base model.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.cv_scores = None

    def _perform_cross_validation(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, scoring: str = "roc_auc"
    ) -> np.ndarray:
        """Perform cross-validation (shared helper for tree-based models)."""
        if self.model is None:
            raise ValueError("Model must be initialized before cross-validation")

        logger.info(f"Performing {cv_folds}-fold cross-validation.")
        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,
        )
        logger.info(f"CV {scoring.upper()}: {scores.mean():.3f} (±{scores.std():.3f})")
        return scores

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates.

        Args:
            X: Feature matrix

        Returns:
            Probability of positive class
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        pass

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_val)
        y_pred_proba = self.predict_proba(X_val)
        results = calculate_all_metrics(y_val, y_pred, y_pred_proba)
        results["model_name"] = self.__class__.__name__

        feature_importance = self.get_feature_importance()
        if feature_importance:
            results["feature_importance"] = feature_importance

        if self.cv_scores is not None:
            results["cv_roc_auc_mean"] = float(self.cv_scores.mean())
            results["cv_roc_auc_std"] = float(self.cv_scores.std())

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation results.

        Args:
            results: Dictionary from evaluate() method
        """
        print("\n" + "=" * 80)
        print(f"{'🔍 ' + results['model_name'] + ' RESULTS':^80}")
        print("=" * 80)

        print("\nClassification Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1 Score:  {results['f1_score']:.3f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.3f}")
        print(f"  PR-AUC:    {results['pr_auc']:.3f}")

        if "cv_roc_auc_mean" in results:
            print("\nCross-Validation:")
            print(
                f"  CV ROC-AUC: {results['cv_roc_auc_mean']:.3f} (±{results['cv_roc_auc_std']:.3f})"
            )

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str, config: Dict[str, Any]) -> "BaseModel":
        """Load model from disk.

        Args:
            filepath: Path to saved model
            config: Model configuration

        Returns:
            Loaded model instance
        """
        instance = cls(config)
        instance.model = joblib.load(filepath)
        return instance
