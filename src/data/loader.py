"""Data loading utilities."""

import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy.engine import Engine

from src.utils.clients import get_db_engine

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading from SQLite database using centralized engine."""

    def __init__(self, db_path: str):
        """Initialize database connection using centralized engine.

        Args:
            db_path: Path to SQLite database"""
        self.db_path = db_path
        self.engine: Engine = get_db_engine(db_path)

    def load_data(self, query: str) -> pd.DataFrame:
        """Load data from database using provided SQL query.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df


def load_and_split_data(
    config: dict, query_file: Optional[str] = "config/queries.sql"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from database and split into train/val/test sets.

    Args:
        config: Configuration dictionary.
        query_file: Path to SQL file containing the data loading query.

    Returns:
        Tuple of (full_df, train_df, val_df, test_df).
    """
    with open(query_file, "r") as f:
        query = f.read()

    db_path = config["data"]["database_path"]
    data_loader = DataLoader(db_path)
    df = data_loader.load_data(query)
    target_col = config["data"]["target_col"]
    logger.info(f"Loaded {len(df)} records from database")

    train_df, test_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=42,
        stratify=df[target_col],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=config["data"]["validation_size"],
        random_state=42,
        stratify=train_df[target_col],
    )
    logger.info(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return df, train_df, val_df, test_df
