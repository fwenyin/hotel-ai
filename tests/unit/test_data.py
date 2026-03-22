"""Tests for data loading and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""

    def test_prepare_features_returns_dataframe_and_series(
        self, small_dataframe, minimal_config
    ):
        """Test that prepare_features returns correct types."""
        preprocessor = DataPreprocessor(minimal_config)
        X, y = preprocessor.prepare_features(small_dataframe, fit=True)

        assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
        assert isinstance(y, pd.Series), "y should be a Series"
        assert len(X) == len(y), "X and y should have same length"
        assert len(X) == len(small_dataframe), "Should preserve all rows"

    def test_prepare_features_no_missing_values(self, small_dataframe, minimal_config):
        """Test that prepared features have no missing values."""
        preprocessor = DataPreprocessor(minimal_config)
        X, y = preprocessor.prepare_features(small_dataframe, fit=True)

        assert X.isna().sum().sum() == 0, "X should have no missing values"
        assert y.isna().sum() == 0, "y should have no missing values"

    def test_prepare_features_numeric_columns(self, small_dataframe, minimal_config):
        """Test that all output columns are numeric."""
        preprocessor = DataPreprocessor(minimal_config)
        X, _ = preprocessor.prepare_features(small_dataframe, fit=True)

        for col in X.columns:
            assert np.issubdtype(
                X[col].dtype, np.number
            ), f"Column {col} should be numeric, got {X[col].dtype}"

    def test_prepare_features_target_binary(self, small_dataframe, minimal_config):
        """Test that target variable is binary."""
        preprocessor = DataPreprocessor(minimal_config)
        _, y = preprocessor.prepare_features(small_dataframe, fit=True)

        unique_values = y.unique()
        assert len(unique_values) == 2, "Target should be binary"
        assert set(unique_values).issubset({0, 1}), "Target should be 0 or 1"

    def test_prepare_features_consistent_columns(self, small_dataframe, minimal_config):
        """Test that train and val have same columns."""
        train_df, val_df = train_test_split(
            small_dataframe, test_size=0.2, random_state=42
        )

        preprocessor = DataPreprocessor(minimal_config)
        X_train, _ = preprocessor.prepare_features(train_df, fit=True)
        X_val, _ = preprocessor.prepare_features(val_df, fit=False)

        assert list(X_train.columns) == list(
            X_val.columns
        ), "Train and val should have same columns"

    @pytest.mark.parametrize("missing_col", ["price"])
    def test_prepare_features_missing_column_raises(
        self, small_dataframe, minimal_config, missing_col
    ):
        """Test that missing required columns raise appropriate errors."""
        df_missing = small_dataframe.drop(columns=[missing_col])
        preprocessor = DataPreprocessor(minimal_config)

        with pytest.raises((KeyError, ValueError)):
            preprocessor.prepare_features(df_missing, fit=True)

    def test_handles_missing_values(self, minimal_config):
        """Test that missing values are imputed correctly."""
        df_with_nulls = pd.DataFrame(
            {
                "booking_id": range(1, 21),
                "branch": ["NYC"] * 20,
                "country": ["US"] * 20,
                "room": ["standard"] * 20,
                "platform": ["direct"] * 20,
                "price": [100.0] * 18 + [np.nan, np.nan],  # 2 missing values
                "num_adults": [2] * 20,
                "num_children": [0] * 20,
                "first_time": [0] * 20,
                "booking_month": [1] * 20,
                "arrival_month": [2] * 20,
                "arrival_day": [15] * 20,
                "checkout_month": [2] * 20,
                "checkout_day": [17] * 20,
                "no_show": [0] * 18 + [1, 1],
            }
        )

        preprocessor = DataPreprocessor(minimal_config)
        X, _ = preprocessor.prepare_features(df_with_nulls, fit=True)

        assert (
            X.isna().sum().sum() == 0
        ), "Should have no missing values after preprocessing"


class TestDataLoader:
    """Tests for the DataLoader class."""

    @pytest.mark.integration
    def test_load_data_returns_dataframe(self, project_root, database_exists):
        """Test that load_data returns a DataFrame."""
        if not database_exists:
            pytest.skip("Database not found")

        db_path = project_root / "data" / "raw_data" / "noshow.db"
        loader = DataLoader(str(db_path))
        df = loader.load_data("SELECT * FROM noshow")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

    @pytest.mark.integration
    def test_load_data_has_required_columns(self, project_root, database_exists):
        """Test that loaded data has required columns."""
        if not database_exists:
            pytest.skip("Database not found")

        required_columns = [
            "booking_id",
            "branch",
            "country",
            "room",
            "platform",
            "price",
            "num_adults",
            "num_children",
            "first_time",
            "no_show",
        ]

        db_path = project_root / "data" / "raw_data" / "noshow.db"
        loader = DataLoader(str(db_path))
        df = loader.load_data("SELECT * FROM noshow")

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_loader_nonexistent_database_raises(self, temp_dir):
        """Test that loading from nonexistent database raises error."""
        fake_path = Path(temp_dir) / "nonexistent.db"
        loader = DataLoader(str(fake_path))

        with pytest.raises(Exception):
            loader.load_data("SELECT * FROM noshow")


class TestFeatureEngineering:
    """Tests for feature engineering logic."""

    def test_stay_length_feature_created(self, small_dataframe, minimal_config):
        """Test that stay length feature is created."""
        preprocessor = DataPreprocessor(minimal_config)
        X, _ = preprocessor.prepare_features(small_dataframe, fit=True)

        stay_cols = [c for c in X.columns if "stay_length" in c.lower()]
        assert len(stay_cols) > 0, "Stay length feature should be created"

    def test_cyclical_features_created(self, small_dataframe, minimal_config):
        """Test that cyclical features are created (sin/cos encoding)."""
        preprocessor = DataPreprocessor(minimal_config)
        X, _ = preprocessor.prepare_features(small_dataframe, fit=True)

        cyclical_cols = [c for c in X.columns if "_sin" in c or "_cos" in c]

        assert len(cyclical_cols) > 0, "Should have cyclical features"

    def test_one_hot_encoding_binary(self, small_dataframe, minimal_config):
        """Test that one-hot encoded features are binary."""
        preprocessor = DataPreprocessor(minimal_config)
        X, _ = preprocessor.prepare_features(small_dataframe, fit=True)

        categorical_cols = [
            c
            for c in X.columns
            if any(cat in c for cat in ["branch_", "country_", "room_", "platform_"])
        ]

        for col in categorical_cols:
            unique_vals = X[col].unique()
            assert set(unique_vals).issubset(
                {0, 1, 0.0, 1.0}
            ), f"{col} should be binary, got {unique_vals}"
