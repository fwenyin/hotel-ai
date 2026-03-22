"""Random Forest model implementation."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for hotel no-show prediction."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.model = RandomForestClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            min_samples_split=self.config["min_samples_split"],
            min_samples_leaf=self.config["min_samples_leaf"],
            class_weight=self.config["class_weight"],
            max_features=self.config["max_features"],
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        """Train Random Forest with cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        logger.info("Training Random Forest model.")

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
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return dict(zip(self.feature_names, self.model.feature_importances_))
