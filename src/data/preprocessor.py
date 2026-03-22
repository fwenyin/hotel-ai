"""Data preprocessing module for hotel no-show prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer."""

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed for these deterministic transforms."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transforms.

        Args:
            X: Input DataFrame with raw booking features.

        Returns:
            DataFrame with original features plus engineered features.
        """
        df = X.copy()
        month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        for col in ["booking_month", "arrival_month", "checkout_month"]:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_numeric_mask = numeric.isna() & df[col].notna()
                if non_numeric_mask.any():
                    str_vals = df[col].astype(str).str.strip().str.title()
                    numeric = numeric.fillna(str_vals.map(month_map))
                df[col] = numeric

        # USD→SGD conversion
        if "price" in df.columns:
            df["currency"] = df["price"].astype(str).str.extract(r"^(\w+)\$")[0]
            df["currency"] = df["currency"].fillna("SGD")
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace(r"^[A-Z]+\$\s*", "", regex=True)
                .str.replace(",", "", regex=False)
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            usd_to_sgd_rate = (
                1.34  # Current USD→SGD rate TODO: Replace with actual live rate
            )
            usd_mask = df["currency"].str.upper() == "USD"
            if usd_mask.any():
                df.loc[usd_mask, "price"] = df.loc[usd_mask, "price"] * usd_to_sgd_rate
            df.loc[usd_mask, "currency"] = "SGD"

        if "first_time" in df.columns:
            df["first_time"] = (
                pd.to_numeric(df["first_time"], errors="coerce").fillna(0).astype(int)
            )

        # Clamp negative checkout_day (data entry errors)
        if "checkout_day" in df.columns:
            df["checkout_day"] = df["checkout_day"].abs()

        text_to_num_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        # Convert text guest counts ("two" → 2)
        for col in ["num_adults", "num_children"]:
            df[col] = df[col].apply(
                lambda x: text_to_num_map.get(str(x).lower(), x) if pd.notna(x) else x
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Impute missing rooms: mode by (branch, total_guests), then branch mode, then global
        if "room" in df.columns and "total_guests" in df.columns:
            group_room_mode = (
                df.groupby(["branch", "total_guests"])["room"].apply(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
                )
                if "branch" in df.columns
                else pd.Series(dtype=object)
            )
            branch_room_mode = (
                df.groupby("branch")["room"].apply(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
                )
                if "branch" in df.columns
                else pd.Series(dtype=object)
            )

            def _impute_room(row):
                if pd.notna(row["room"]):
                    return row["room"]
                key = (row.get("branch"), row["total_guests"])
                if key in group_room_mode.index and pd.notna(group_room_mode[key]):
                    return group_room_mode[key]
                branch = row.get("branch")
                if branch in branch_room_mode.index and pd.notna(
                    branch_room_mode[branch]
                ):
                    return branch_room_mode[branch]
                return np.nan

            df["room"] = df.apply(_impute_room, axis=1)
            if df["room"].isna().any():
                global_mode = df["room"].mode()
                if len(global_mode) > 0:
                    df["room"] = df["room"].fillna(global_mode.iloc[0])

        # Impute missing prices: median by (branch, room), then global median
        if "price" in df.columns:
            if "branch" in df.columns and "room" in df.columns:
                group_median = df.groupby(["branch", "room"])["price"].transform(
                    "median"
                )
                missing_mask = df["price"].isna()
                df.loc[missing_mask, "price"] = group_median[missing_mask]
            if df["price"].isna().any():
                df["price"] = df["price"].fillna(df["price"].median())

        DAYS_IN_MONTH = {
            1: 31,
            2: 29,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }

        def days_in_month(m):
            return DAYS_IN_MONTH.get(m, 30)

        def calc_stay_length(row):
            if row["checkout_month"] == row["arrival_month"]:
                return max(1, row["checkout_day"] - row["arrival_day"])
            else:
                # span across months
                m_diff = int((row["checkout_month"] - row["arrival_month"]) % 12)
                days = days_in_month(int(row["arrival_month"])) - row["arrival_day"]
                for i in range(1, m_diff):
                    m = int((row["arrival_month"] + i - 1) % 12 + 1)
                    days += days_in_month(m)
                days += row["checkout_day"]
                return max(1, days)

        df["stay_length"] = df.apply(calc_stay_length, axis=1)

        df["lead_time_months"] = (df["arrival_month"] - df["booking_month"]) % 12

        df["total_guests"] = df["num_adults"] + df["num_children"]
        df["price_per_guest"] = df["price"] / df["total_guests"].replace(0, np.nan)

        # cyclical month encoding
        for col in ["arrival_month", "booking_month"]:
            if col in df.columns:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / 12)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / 12)

        # cyclical day encoding (normalised by days in month)
        days_in_arr_month = df["arrival_month"].apply(days_in_month)
        df["arrival_day_sin"] = np.sin(
            2 * np.pi * df["arrival_day"] / days_in_arr_month
        )
        df["arrival_day_cos"] = np.cos(
            2 * np.pi * df["arrival_day"] / days_in_arr_month
        )
        days_in_chk_month = df["checkout_month"].apply(days_in_month)
        df["checkout_day_sin"] = np.sin(
            2 * np.pi * df["checkout_day"] / days_in_chk_month
        )
        df["checkout_day_cos"] = np.cos(
            2 * np.pi * df["checkout_day"] / days_in_chk_month
        )

        return df


class DataPreprocessor:
    """Preprocessing pipeline including feature
    engineering, imputation, encoding, and scaling.
    """

    def __init__(self, config: Dict):
        """Initialize preprocessor from configuration.

        Args:
            config: Configuration dictionary with feature definitions.
        """
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

        self.numerical_features = config["features"]["numerical"]
        self.categorical_features = config["features"]["categorical"]
        self.binary_features = config["features"]["binary"]
        self.target_col = config["data"]["target_col"]

    def _build_transformer(self) -> ColumnTransformer:
        """Build ColumnTransformer for feature preprocessing.

        Returns:
            Configured ColumnTransformer instance.
        """

        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformers = [
            ("num", numerical_pipeline, self.numerical_features),
            ("cat", categorical_pipeline, self.categorical_features),
            ("bin", "passthrough", self.binary_features),
        ]

        return ColumnTransformer(
            transformers=transformers, remainder="drop", verbose_feature_names_out=False
        )

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Execute complete preprocessing pipeline.

        Args:
            df: Input DataFrame with raw booking data.
            fit: Whether to fit transformers (True for training data,
                False for validation/test data).

        Returns:
            Tuple containing:
                - X: Preprocessed feature DataFrame
                - y: Target variable Series (or None if target not in df)
        """
        df = self.feature_engineer.transform(df)
        transform_cols = (
            self.numerical_features + self.categorical_features + self.binary_features
        )

        if fit:
            self.transformer = self._build_transformer()
            transformed = self.transformer.fit_transform(df[transform_cols])
            self.feature_names = self.transformer.get_feature_names_out().tolist()
        else:
            transformed = self.transformer.transform(df[transform_cols])

        X = pd.DataFrame(transformed, columns=self.feature_names, index=df.index)
        y = df[self.target_col]
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor.

        Args:
            df: Input DataFrame with same schema as training data.

        Returns:
            Preprocessed feature DataFrame.
        """
        X, _ = self.prepare_features(df, fit=False)
        return X

    def save(self, filepath: str) -> None:
        """Save fitted preprocessor to disk using joblib.

        Args:
            filepath: Path where preprocessor will be saved.
        """
        state = {
            "config": self.config,
            "transformer": self.transformer,
            "feature_names": self.feature_names,
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Load fitted preprocessor from disk.

        Args:
            filepath: Path to saved preprocessor file.

        Returns:
            Loaded DataPreprocessor instance with fitted transformers.
        """
        state = joblib.load(filepath)
        instance = cls(state["config"])
        instance.transformer = state["transformer"]
        instance.feature_names = state["feature_names"]
        logger.info(f"Loaded preprocessor from {filepath}")
        return instance


def preprocess_splits(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor_path: str,
    fit: bool = True,
) -> Tuple[
    "DataPreprocessor",
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Preprocess train/val/test splits.

    Args:
        config: Configuration dictionary.
        train_df: Training dataframe.
        val_df: Validation dataframe.
        test_df: Test dataframe.
        preprocessor_path: Path to save/load preprocessor.
        fit: Whether to fit the preprocessor on training data.

    Returns:
        Tuple of (preprocessor, X_train, y_train, X_val, y_val, X_test, y_test).
    """
    logger.info("Preprocessing data.")
    if fit:
        preprocessor = DataPreprocessor(config)
    else:
        preprocessor = DataPreprocessor.load(preprocessor_path)
    X_train, y_train = preprocessor.prepare_features(train_df, fit=fit)
    X_val, y_val = preprocessor.prepare_features(val_df, fit=False)
    X_test, y_test = preprocessor.prepare_features(test_df, fit=False)
    logger.info(
        f"Preprocessed: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    if fit:
        preprocessor.save(preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")

    return preprocessor, X_train, y_train, X_val, y_val, X_test, y_test
