"""Batch prediction script using the production model and preprocessor."""
import argparse
import logging
from typing import Dict, Any
import pandas as pd
import joblib

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

setup_logging()
logger = logging.getLogger(__name__)


def run_batch_inference(
    config: Dict[str, Any],
    db_path: str,
    query_path: str,
    output_path: str
) -> pd.DataFrame:
    """Run batch inference on new data loaded from a SQLite DB using an SQL query.

    Args:
        config: Configuration dictionary
        db_path: Path to input SQLite database file
        query_path: Path to .sql file
        output_path: Path to save predictions

    Returns:
        pandas.DataFrame: DataFrame containing input rows with prediction columns
            appended (``noshow_prob``, ``prediction``).
    """
    preprocessor_path = config['paths']['production']['preprocessor']
    model_path = config['paths']['production']['model']

    logger.info("Loading model and preprocessor")
    preprocessor = DataPreprocessor.load(preprocessor_path)
    model = joblib.load(model_path)
    if preprocessor is None or model is None:
        raise FileNotFoundError("Production artifacts not found. Run retrain.py first.")

    with open(query_path, 'r') as f:
        query = f.read()
    
    logger.info(f"Loading data from {db_path}")
    loader = DataLoader(db_path)
    df = loader.load_data(query)
    logger.info(f"Loaded {len(df)} records from {db_path}")

    logger.info("Preprocessing data")
    X = preprocessor.transform(df)

    logger.info("Generating predictions")
    probs = model.predict_proba(X)
    preds = model.predict(X)
    results_df = df.copy()
    results_df['noshow_prob'] = probs
    results_df['prediction'] = preds

    results_df.to_parquet(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"  - Predicted no-shows: {preds.sum()} ({preds.mean():.2%})")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('--db', type=str, required=True, 
                       help='Path to input SQLite DB file')
    parser.add_argument('--query', type=str, default='config/queries.sql',
                       help='Path to .sql file')
    parser.add_argument('--output', type=str, default='outputs/predictions.parquet',
                       help='Path to save predictions')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    results_df = run_batch_inference(
        config, 
        args.db, 
        args.query, 
        args.output
    )
    print("\nSample Predictions:")
    print(results_df[['noshow_prob', 'prediction']].head())


if __name__ == "__main__":
    main()

