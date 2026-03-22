"""
TabNet model implementation using PyTorch-TabNet.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class TabNetModel(BaseModel):
    """TabNet classifier for hotel no-show prediction.

    TabNet uses sequential attention mechanism to select relevant features
    at each decision step, providing both strong predictive performance
    and interpretability through attention masks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize TabNet model.

        Args:
            config: Configuration dictionary with TabNet hyperparameters
        """
        super().__init__(config)

        self.model = TabNetClassifier(
            n_d=self.config["n_d"],
            n_a=self.config["n_a"],
            n_steps=self.config["n_steps"],
            gamma=self.config["gamma"],
            lambda_sparse=self.config["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.config["learning_rate"]),
            mask_type=self.config["mask_type"],
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.9),
            verbose=0,
            seed=42,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TabNetModel":
        """Train TabNet model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels for early stopping

        Returns:
            Self for method chaining
        """
        logger.info("Training TabNet model.")

        self.feature_names = X_train.columns.tolist()

        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.int64)

        eval_set = None
        eval_name = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.int64)
            eval_set = [(X_val_np, y_val_np)]
            eval_name = ["val"]

        neg_count = (y_train_np == 0).sum()
        pos_count = (y_train_np == 1).sum()
        total = len(y_train_np)
        class_weights = {0: total / (2 * neg_count), 1: total / (2 * pos_count)}

        self.model.fit(
            X_train_np,
            y_train_np,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=["accuracy", "auc"],
            max_epochs=self.config["max_epochs"],
            patience=self.config["patience"],
            batch_size=self.config["batch_size"],
            virtual_batch_size=self.config["virtual_batch_size"],
            weights=class_weights,
        )

        logger.info("TabNet model training completed.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0=show, 1=no-show)
        """
        X_np = X.values.astype(np.float32)
        return self.model.predict(X_np)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability of no-show.

        Args:
            X: Feature matrix

        Returns:
            Probability of no-show (class 1)
        """
        X_np = X.values.astype(np.float32)
        proba = self.model.predict_proba(X_np)
        return proba[:, 1]

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from TabNet.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
