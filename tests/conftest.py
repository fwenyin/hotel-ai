"""Pytest fixtures and configuration."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    from src.utils.config import load_config

    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="session")
def minimal_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Config with reduced epochs/iterations for faster tests."""
    test_config = config.copy()
    for model_name in test_config.get("models", {}).keys():
        if "epochs" in test_config["models"][model_name]:
            test_config["models"][model_name]["epochs"] = 5
        if "n_estimators" in test_config["models"][model_name]:
            test_config["models"][model_name]["n_estimators"] = 10
        if "max_iter" in test_config["models"][model_name]:
            test_config["models"][model_name]["max_iter"] = 100

    return test_config


@pytest.fixture(scope="session")
def sample_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n_samples = 500

    return pd.DataFrame(
        {
            "booking_id": range(1, n_samples + 1),
            "branch": np.random.choice(["NYC", "LA", "CHI", "MIA"], n_samples),
            "country": np.random.choice(["US", "UK", "CA", "DE", "FR"], n_samples),
            "room": np.random.choice(["standard", "deluxe", "suite"], n_samples),
            "platform": np.random.choice(["direct", "ota", "corporate"], n_samples),
            "price": np.random.uniform(100, 500, n_samples),
            "num_adults": np.random.randint(1, 4, n_samples),
            "num_children": np.random.randint(0, 3, n_samples),
            "first_time": np.random.randint(0, 2, n_samples),
            "booking_month": np.random.randint(1, 13, n_samples),
            "arrival_month": np.random.randint(1, 13, n_samples),
            "arrival_day": np.random.randint(1, 29, n_samples),
            "checkout_month": np.random.randint(1, 13, n_samples),
            "checkout_day": np.random.randint(1, 29, n_samples),
            "no_show": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )


@pytest.fixture(scope="function")
def small_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "booking_id": range(1, n_samples + 1),
            "branch": np.random.choice(["NYC", "LA"], n_samples),
            "country": np.random.choice(["US", "UK"], n_samples),
            "room": np.random.choice(["standard", "deluxe"], n_samples),
            "platform": np.random.choice(["direct", "ota"], n_samples),
            "price": np.random.uniform(100, 300, n_samples),
            "num_adults": np.random.randint(1, 3, n_samples),
            "num_children": np.random.randint(0, 2, n_samples),
            "first_time": np.random.randint(0, 2, n_samples),
            "booking_month": np.random.randint(1, 13, n_samples),
            "arrival_month": np.random.randint(1, 13, n_samples),
            "arrival_day": np.random.randint(1, 29, n_samples),
            "checkout_month": np.random.randint(1, 13, n_samples),
            "checkout_day": np.random.randint(1, 29, n_samples),
            "no_show": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )


@pytest.fixture(scope="function")
def train_val_data(small_dataframe: pd.DataFrame, minimal_config: Dict) -> Tuple:
    from sklearn.model_selection import train_test_split

    from src.data.preprocessor import DataPreprocessor

    train_df, val_df = train_test_split(
        small_dataframe,
        test_size=0.2,
        random_state=42,
        stratify=small_dataframe["no_show"],
    )

    preprocessor = DataPreprocessor(minimal_config)
    X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
    X_val, y_val = preprocessor.prepare_features(val_df, fit=False)

    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="function")
def temp_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_mlruns(temp_dir):
    mlruns_dir = os.path.join(temp_dir, "mlruns")
    os.makedirs(mlruns_dir)
    return mlruns_dir


@pytest.fixture(scope="session")
def database_path(project_root) -> Path:
    return project_root / "data" / "raw_data" / "noshow.db"


@pytest.fixture(scope="session")
def database_exists(database_path) -> bool:
    return database_path.exists()


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
