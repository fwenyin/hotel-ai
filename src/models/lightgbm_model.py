"""LightGBM model implementation."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier for hotel no-show prediction."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.model = LGBMClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            num_leaves=self.config["num_leaves"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            min_child_samples=self.config["min_child_samples"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            class_weight="balanced",
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMModel":
        """Train LightGBM with cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        logger.info("Training LightGBM model.")

        self.cv_scores = self._perform_cross_validation(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0=show, 1=no-show)
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability of no-show.

        Args:
            X: Feature matrix

        Returns:
            Probability of no-show (class 1)
        """
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores (gain-based).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return dict(zip(self.feature_names, self.model.feature_importances_))
