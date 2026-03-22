"""Production model retraining script."""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import joblib
import mlflow
from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.models.model_trainer import ModelTrainer
from src.data.loader import load_and_split_data
from src.data.preprocessor import preprocess_splits
from src.mlops.mlflow_tracker import MLflowTracker

setup_logging()
logger = logging.getLogger(__name__)


def run_champion_retraining(config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrain the champion model for production deployment.

    Args:
        config: Configuration dictionary.

    Returns:
        dict: Dictionary with training results and model info.
    """
    champion_model_name = config['champion_model']
    if not champion_model_name:
        raise ValueError("No champion model found in config. Run tune.py first.")
    logger.info(f"Retraining champion model: {champion_model_name}")
    
    Path(config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading and splitting data")
    _, train_df, val_df, test_df = load_and_split_data(config)
    preprocessor_path = config['paths']['production']['preprocessor']
    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test = preprocess_splits(
        config, train_df, val_df, test_df,
        fit=True,
        preprocessor_path=preprocessor_path
    )
    
    experiment_name = config['mlflow']['experiments']['production']
    mlflow_dir = config['mlflow']['tracking_uri']
    tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=mlflow_dir)
    
    trainer = ModelTrainer(config)
    
    with tracker.start_run(run_name=f"retrain_{champion_model_name}") as run:
        model_params = config['models'][champion_model_name]
        tracker.log_params(model_params)
        model = trainer.factory.create(champion_model_name, model_params)
        model.fit(X_train, y_train)
        results = model.evaluate(X_val, y_val)
    
        tracker.log_metrics(results)
        logger.info(f"Champion model retrained")
        logger.info(f"  - ROC-AUC: {results['roc_auc']:.3f}")
        logger.info(f"  - Accuracy: {results['accuracy']:.3f}")
        logger.info(f"  - F1 Score: {results['f1_score']:.3f}")
        
        sklearn_model = getattr(model, 'model', None)
        if sklearn_model:
            if champion_model_name == 'tabnet':
                tracker.log_model(model, "model", model_type="pytorch", X_sample=X_val.head(5))
            else:
                mlflow.sklearn.log_model(sklearn_model, "model")
        logger.info(f"Logged {champion_model_name} model to MLflow")
        
        run_id = tracker.active_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, "hotel_noshow_champion")
        logger.info(f"Registered model to Model Registry as 'hotel_noshow_champion' (version {registered_model.version})")
        
        model_path = config['paths']['production']['model']
        preprocessor.save(preprocessor_path)
        joblib.dump(model, model_path)
        logger.info(f"Artifacts saved: {preprocessor_path}, {model_path}")
        
        os.makedirs('reports', exist_ok=True)
        metrics_path = 'reports/performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return {
            'model_name': champion_model_name,
            'results': results,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'registered_model_version': registered_model.version,
            'metrics_path': metrics_path
        }


def main():
    parser = argparse.ArgumentParser(description='Retrain Champion Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    
    results = run_champion_retraining(config)
    logger.info(f"Model artifacts saved:")
    logger.info(f"  - Model: {results['model_path']}")
    logger.info(f"  - Preprocessor: {results['preprocessor_path']}")


if __name__ == "__main__":
    main()

