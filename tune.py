"""Hyperparameter tuning script using Optuna."""
import argparse
import logging
import yaml
import mlflow
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.models.model_trainer import ModelTrainer
from src.data.loader import load_and_split_data
from src.data.preprocessor import preprocess_splits
from src.mlops.mlflow_tracker import MLflowTracker

setup_logging()
logger = logging.getLogger(__name__)


def run_hyperparameter_tuning(
    config: Dict[str, Any],
    n_trials: int = 20,
    model_types: Optional[List[str]] = ['random_forest', 'xgboost', 'lightgbm'],
) -> Dict[str, Any]:
    """Run hyperparameter tuning for specified models.

    Args:
        config: Configuration dictionary.
        n_trials: Number of Optuna trials per model.
        model_types: List of model types to tune.

    Returns:
        Dictionary with best model name, score, and parameters.
    """
    Path(config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)

    logger.info("Loading and splitting data.")
    _, train_df, val_df, test_df = load_and_split_data(config)

    preprocessor_path = config['paths']['tuning']['preprocessor']
    fit_preprocessor = not Path(preprocessor_path).exists()
    if fit_preprocessor:
        logger.info("No existing preprocessor found — fitting from scratch.")
    else:
        logger.info(f"Reusing existing preprocessor from {preprocessor_path}.")
    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test = preprocess_splits(
        config, train_df, val_df, test_df,
        fit=fit_preprocessor,
        preprocessor_path=preprocessor_path
    )

    experiment_name = config['mlflow']['experiments']['tuning']
    mlflow_dir = config['mlflow']['tracking_uri']
    tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=mlflow_dir)
    trainer = ModelTrainer(config)

    best_overall_score = -1.0
    best_overall_model = None
    best_overall_params = None
    best_model_name: str = ""
    best_run_id: str = ""

    logger.info(f"Starting tuning for models: {model_types}")

    for model_name in model_types:
        with tracker.start_run(run_name=f"tune_{model_name}"):
            tracker.log_params({"model_type": model_name, "n_trials": n_trials})
            results = trainer.tune_and_train(
                model_name,
                X_train, y_train,
                X_val, y_val,
                n_trials=n_trials
            )
            
            run = mlflow.active_run()
            tracker.log_params(results['tuned_params'])
            tracker.log_metrics({
                "roc_auc": results['roc_auc'],
                "accuracy": results['accuracy'],
                "f1_score": results['f1_score'],
                "cv_score": results['tuning_best_cv_score']
            })
            model_instance = trainer.models[model_name]
            mlflow.sklearn.log_model(model_instance, "model")

            if results['roc_auc'] > best_overall_score:
                best_overall_score = results['roc_auc']
                best_overall_model = model_instance
                best_overall_params = results['tuned_params']
                best_model_name = model_name
                best_run_id = run.info.run_id
                
            logger.info(f"Finished tuning {model_name}. Best ROC-AUC: {results['roc_auc']:.3f}")

    logger.info(f"Best model overall: {best_model_name} with ROC-AUC: {best_overall_score:.3f}")
    
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "hotel_noshow_champion")
    logger.info(f"Registered {best_model_name} to Model Registry as 'hotel_noshow_champion' (version {registered_model.version})")

    return {
        'best_model_name': best_model_name,
        'best_score': best_overall_score,
        'best_params': best_overall_params,
        'best_model': best_overall_model,
        'registered_model_version': registered_model.version
    }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials per model')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'xgboost', 'lightgbm'], help='Models to tune (default: random_forest, xgboost, lightgbm)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    results = run_hyperparameter_tuning(
        config,
        n_trials=args.n_trials,
        model_types=args.models
    )
    best_model_name = results['best_model_name']
    best_params = results['best_params']
    best_score = results['best_score']
    config['models'][best_model_name].update(best_params)

    current_champion = config.get('champion_model')
    current_champion_score = config.get('champion_roc_auc', 0.0)
    if best_score > current_champion_score:
        config['champion_model'] = best_model_name
        config['champion_roc_auc'] = round(best_score, 6)
        logger.info(f"New champion: {best_model_name} (ROC-AUC: {best_score:.4f}) — beat previous {current_champion} ({current_champion_score:.4f})")
    else:
        logger.info(f"{best_model_name} (ROC-AUC: {best_score:.4f}) did not beat current champion {current_champion} ({current_champion_score:.4f}), champion unchanged")

    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Best parameters saved to {args.config}")


if __name__ == "__main__":
    main()

