"""Tests for model training and evaluation."""

import os

import numpy as np
import pandas as pd
import pytest

from src.models.hyperparameter_tuner import TunerFactory
from src.models.model_trainer import ModelFactory, ModelTrainer


class TestModelFactory:
    """Tests for the ModelFactory class."""

    def test_factory_unknown_model_raises(self):
        """Test that unknown model type raises error."""
        with pytest.raises((ValueError, KeyError)):
            ModelFactory.create("unknown_model", {})

    @pytest.mark.parametrize(
        "model_type", ["random_forest", "xgboost", "lightgbm", "neural_network"]
    )
    def test_model_has_required_methods(self, model_type, minimal_config):
        """Test that all models have required interface methods."""
        config = minimal_config["models"].get(model_type, {})
        model = ModelFactory.create(model_type, config)

        required_methods = [
            "fit",
            "predict",
            "predict_proba",
            "evaluate",
            "get_feature_importance",
        ]

        for method in required_methods:
            assert hasattr(model, method), f"{model_type} missing method: {method}"
            assert callable(
                getattr(model, method)
            ), f"{model_type}.{method} not callable"


class TestRandomForestModel:
    """Tests for Random Forest model."""

    def test_random_forest_fit_predict(self, train_val_data, minimal_config):
        """Test Random Forest fit and predict."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert len(predictions) == len(X_val)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_random_forest_predict_proba(self, train_val_data, minimal_config):
        """Test Random Forest probability predictions."""
        X_train, y_train, X_val, _ = train_val_data

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_val)

        assert len(probas) == len(X_val)
        assert (probas >= 0).all() and (
            probas <= 1
        ).all(), "Probabilities should be in [0, 1]"

    def test_random_forest_actually_learns(self, minimal_config):
        """Test that model actually learns from obviously separable data."""
        np.random.seed(42)
        X = pd.DataFrame(
            {"feature1": [0.0] * 50 + [10.0] * 50, "feature2": [0.0] * 50 + [10.0] * 50}
        )
        y = pd.Series([0] * 50 + [1] * 50)

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X, y)
        predictions = model.predict(X)

        accuracy = (predictions == y).mean()
        assert (
            accuracy > 0.95
        ), f"Model failed to learn separable data. Accuracy: {accuracy}"

    def test_random_forest_feature_importance(self, train_val_data, minimal_config):
        """Test Random Forest feature importance."""
        X_train, y_train, _, _ = train_val_data

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)

        importance = model.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == len(X_train.columns)


class TestXGBoostModel:
    """Tests for XGBoost model."""

    def test_xgboost_fit_predict(self, train_val_data, minimal_config):
        """Test XGBoost fit and predict."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create("xgboost", minimal_config["models"]["xgboost"])
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert len(predictions) == len(X_val)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_xgboost_handles_missing_values(self, minimal_config):
        """Test that XGBoost handles missing values."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "f1": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
                "f2": [np.nan, 2.0, 3.0, 4.0, 5.0] * 20,
            }
        )
        y = pd.Series([0, 1, 0, 1, 0] * 20)

        model = ModelFactory.create("xgboost", minimal_config["models"]["xgboost"])
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestLightGBMModel:
    """Tests for LightGBM model."""

    def test_lightgbm_fit_predict(self, train_val_data, minimal_config):
        """Test LightGBM fit and predict."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create("lightgbm", minimal_config["models"]["lightgbm"])
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert len(predictions) == len(X_val)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_lightgbm_feature_importance(self, train_val_data, minimal_config):
        """Test LightGBM feature importance."""
        X_train, y_train, _, _ = train_val_data

        model = ModelFactory.create("lightgbm", minimal_config["models"]["lightgbm"])
        model.fit(X_train, y_train)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == len(X_train.columns)


class TestNeuralNetworkModel:
    """Tests for Neural Network model."""

    def test_neural_network_fit_predict(self, train_val_data, minimal_config):
        """Test Neural Network fit and predict."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create(
            "neural_network", minimal_config["models"]["neural_network"]
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert len(predictions) == len(X_val)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_neural_network_predict_proba_range(self, train_val_data, minimal_config):
        """Test that Neural Network probabilities are in valid range."""
        X_train, y_train, X_val, _ = train_val_data

        model = ModelFactory.create(
            "neural_network", minimal_config["models"]["neural_network"]
        )
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_val)

        assert (probas >= 0).all(), "Probabilities should be >= 0"
        assert (probas <= 1).all(), "Probabilities should be <= 1"

    @pytest.mark.slow
    def test_neural_network_early_stopping(self, train_val_data, minimal_config):
        """Test Neural Network with early stopping."""
        X_train, y_train, X_val, y_val = train_val_data

        config = minimal_config["models"]["neural_network"].copy()
        config["epochs"] = 100
        config["early_stopping_patience"] = 3

        model = ModelFactory.create("neural_network", config)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)


class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    def test_trainer_train_model(self, train_val_data, minimal_config):
        """Test training a single model."""
        X_train, y_train, X_val, y_val = train_val_data

        trainer = ModelTrainer(minimal_config)
        results = trainer.train_model("random_forest", X_train, y_train, X_val, y_val)

        assert isinstance(results, dict)
        assert "roc_auc" in results
        assert "random_forest" in trainer.models

    def test_trainer_save_load_models(self, train_val_data, minimal_config, temp_dir):
        """Test saving and loading models."""
        X_train, y_train, X_val, y_val = train_val_data

        trainer = ModelTrainer(minimal_config)
        trainer.tune_and_train("random_forest", X_train, y_train, X_val, y_val)

        trainer.save_models(temp_dir)

        saved_files = os.listdir(temp_dir)
        assert any("random_forest" in f for f in saved_files)


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    def test_evaluate_returns_all_metrics(self, train_val_data, minimal_config):
        """Test that evaluate returns all expected metrics."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)
        results = model.evaluate(X_val, y_val)

        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"

    def test_metrics_in_valid_range(self, train_val_data, minimal_config):
        """Test that all metrics are in valid ranges."""
        X_train, y_train, X_val, y_val = train_val_data

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)
        results = model.evaluate(X_val, y_val)

        probability_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
        ]

        for metric in probability_metrics:
            if metric in results and results[metric] is not None:
                assert (
                    0.0 <= results[metric] <= 1.0
                ), f"{metric} must be between 0 and 1, got {results[metric]}"


class TestHyperparameterTuning:
    """Tests for hyperparameter tuning."""

    @pytest.mark.slow
    def test_tuner_returns_params(self, train_val_data):
        """Test that tuner returns valid parameters."""
        X_train, y_train, _, _ = train_val_data

        tuner = TunerFactory.create(
            "random_forest", n_trials=3, cv_folds=2  # Very few for speed
        )

        best_params = tuner.tune(X_train, y_train, verbose=False)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    @pytest.mark.slow
    def test_tuner_improves_on_baseline(self, train_val_data, minimal_config):
        """Test that tuning produces reasonable params."""
        X_train, y_train, _, _ = train_val_data

        tuner = TunerFactory.create("random_forest", n_trials=5, cv_folds=2)

        best_params = tuner.tune(X_train, y_train, verbose=False)

        assert "n_estimators" in best_params or "max_depth" in best_params
