"""Neural Network model implementation (scikit-learn MLP)."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class NeuralNetworkModel(BaseModel):
    """Neural Network (MLP) classifier for hotel no-show prediction."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Neural Network model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Reconstruct hidden_layers from Optuna params if present
        if "n_layers" in self.config:
            n_layers = self.config["n_layers"]
            hidden_layers = [
                self.config[f"n_units_l{i}"]
                for i in range(n_layers)
                if f"n_units_l{i}" in self.config
            ]
            if hidden_layers:
                self.config["hidden_layers"] = hidden_layers

        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(self.config["hidden_layers"]),
            activation=self.config["activation"],
            learning_rate_init=self.config["learning_rate"],
            alpha=self.config["alpha"],
            batch_size=self.config["batch_size"],
            max_iter=self.config.get("epochs", 300),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.config.get("early_stopping_patience", 15),
            random_state=42,
            verbose=True,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NeuralNetworkModel":
        """Train Neural Network with early stopping.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        logger.info("Training Neural Network model.")

        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()

        logger.info(f"Training completed in {self.model.n_iter_} iterations")
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
        """Get feature importance using model weights as proxy."""
        input_weights = np.abs(self.model.coefs_[0]).mean(axis=1)
        return dict(zip(self.feature_names, input_weights))
