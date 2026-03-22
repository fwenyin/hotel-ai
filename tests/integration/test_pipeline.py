"""Integration tests for the end-to-end pipeline."""

import os

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelFactory, ModelTrainer
from src.utils.config import load_config


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_complete_training_pipeline(
        self, sample_dataframe, minimal_config, temp_dir
    ):
        """Test complete pipeline from data to predictions."""
        train_df, test_df = train_test_split(
            sample_dataframe,
            test_size=0.2,
            random_state=42,
            stratify=sample_dataframe["no_show"],
        )

        preprocessor = DataPreprocessor(minimal_config)
        X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
        X_test, y_test = preprocessor.prepare_features(test_df, fit=False)

        trainer = ModelTrainer(minimal_config)
        results = trainer.train_model("random_forest", X_train, y_train, X_test, y_test)

        assert "roc_auc" in results
        assert 0.0 <= results["roc_auc"] <= 1.0

        trainer.save_models(temp_dir)
        saved_files = os.listdir(temp_dir)
        assert any("random_forest" in f for f in saved_files)

    def test_prediction_pipeline(self, sample_dataframe, minimal_config):
        """Test prediction on new data."""
        train_df, test_df = train_test_split(
            sample_dataframe,
            test_size=0.2,
            random_state=42,
            stratify=sample_dataframe["no_show"],
        )

        preprocessor = DataPreprocessor(minimal_config)
        X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
        X_test, _ = preprocessor.prepare_features(test_df, fit=False)

        model = ModelFactory.create(
            "random_forest", minimal_config["models"]["random_forest"]
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert set(np.unique(predictions)).issubset({0, 1})
        assert (probabilities >= 0).all() and (probabilities <= 1).all()


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests requiring database access."""

    def test_full_pipeline_with_real_data(
        self, project_root, database_exists, minimal_config
    ):
        """Test full pipeline with real database data."""
        if not database_exists:
            pytest.skip("Database not found")

        db_path = project_root / "data" / "raw_data" / "noshow.db"

        query_file = project_root / "config" / "queries.sql"
        if query_file.exists():
            with open(query_file, "r") as f:
                query = f.read()
        else:
            query = "SELECT * FROM noshow WHERE no_show IS NOT NULL"

        loader = DataLoader(str(db_path))
        df = loader.load_data(query)

        df = df.sample(n=min(1000, len(df)), random_state=42)

        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["no_show"]
        )

        preprocessor = DataPreprocessor(minimal_config)
        X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
        X_val, y_val = preprocessor.prepare_features(val_df, fit=False)

        trainer = ModelTrainer(minimal_config)
        results = trainer.train_model("random_forest", X_train, y_train, X_val, y_val)

        assert results["roc_auc"] > 0.5


@pytest.mark.integration
class TestConfigurationIntegration:
    """Tests for configuration loading and validation."""

    def test_load_production_config(self, project_root):
        """Test loading the production configuration file."""
        config_path = project_root / "config" / "config.yaml"
        config = load_config(str(config_path))

        assert "data" in config
        assert "models" in config
        assert "features" in config

    def test_config_model_params_valid(self, config):
        """Test that model parameters in config are valid."""
        rf_config = config["models"]["random_forest"]
        if "n_estimators" in rf_config:
            assert rf_config["n_estimators"] > 0

        xgb_config = config["models"]["xgboost"]
        if "learning_rate" in xgb_config:
            assert 0 < xgb_config["learning_rate"] <= 1
