"""XGBoost model implementation."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier for hotel no-show prediction."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.model = XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            min_child_weight=self.config["min_child_weight"],
            gamma=self.config["gamma"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            scale_pos_weight=self.config.get("scale_pos_weight", 1.0),
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostModel":
        """Train XGBoost with cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        logger.info("Training XGBoost model.")

        if self.model.scale_pos_weight == 1.0:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            if pos > 0:
                self.model.scale_pos_weight = neg / pos
                logger.info(
                    f"Auto-set scale_pos_weight={self.model.scale_pos_weight:.3f}"
                )

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
