"""End-to-end ML pipeline orchestrator for hotel no-show prediction."""
import logging
import argparse
import yaml
from pathlib import Path
from typing import List

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import get_champion_model
from src.genai.interpreter import GenAIInterpreter
from src.genai.agent import DataScienceAgent
from src.data.loader import load_and_split_data
from src.data.preprocessor import preprocess_splits
from tune import run_hyperparameter_tuning
from retrain import run_champion_retraining

setup_logging()
logger = logging.getLogger(__name__)


class HotelNoShowPipeline:
    """Unified pipeline orchestrating tuning, retraining, and assessment."""
    
    def __init__(
        self, 
        config_path: str = 'config/config.yaml', 
        model_types: List[str] = ['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'tabnet']
    ) -> None:
        """Initialize the pipeline.

        Args:
            config_path: Path to YAML config file.
            model_types: List of model type keys to operate on (default: all 5 models).
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.model_types = model_types 
        self.trainer = None
        self.genai = None
        self.agent = None
        
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.preprocessor = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def _load_and_preprocess(self, preprocessor_path: str) -> None:
        """Load and preprocess data (common for all modes).

        Args:
            preprocessor_path: Path where preprocessor will be saved/loaded.
        """
        logger.info("Loading and splitting data.")
        
        self.df, self.train_df, self.val_df, self.test_df = load_and_split_data(self.config)
        self.preprocessor, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = preprocess_splits(
            self.config, self.train_df, self.val_df, self.test_df,
            fit=True,
            preprocessor_path=preprocessor_path
        )

    def run_tuning(self, n_trials: int = 20) -> None:
        """Run hyperparameter tuning.

        Args:
            n_trials: Number of Optuna trials to perform per model.
        """
        logger.info("="*70)
        logger.info(f"starting hyperparameter tuning (n_trials={n_trials})")
        logger.info("="*70)
        results = run_hyperparameter_tuning(self.config, n_trials, self.model_types)
        best_model_name = results['best_model_name']
        best_params = results['best_params']
        best_score = results['best_score']
        self.config['models'][best_model_name].update(best_params)

        current_champion = self.config.get('champion_model')
        current_champion_score = self.config.get('champion_roc_auc', 0.0)
        if best_score > current_champion_score:
            self.config['champion_model'] = best_model_name
            self.config['champion_roc_auc'] = round(best_score, 6)
            logger.info(f"New champion: {best_model_name} (ROC-AUC: {best_score:.4f}) — beat previous {current_champion} ({current_champion_score:.4f})")
        else:
            logger.info(f"{best_model_name} (ROC-AUC: {best_score:.4f}) did not beat current champion {current_champion} ({current_champion_score:.4f}), champion unchanged")

        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def run_retrain(self) -> None:
        """Retrain champion model."""
        logger.info("="*70)
        logger.info("retraining champion model")
        logger.info("="*70)
        
        results = run_champion_retraining(self.config)
        logger.info(f"Model artifacts saved:")
        logger.info(f"  - Model: {results['model_path']}")
        logger.info(f"  - Preprocessor: {results['preprocessor_path']}")

    def run_full_assessment(self) -> None:
        """Run the full assessment pipeline (Train All, GenAI, Agent).

        This will ensure models directory exists, preprocess data, run
        training for configured models, run GenAI interpretation, and execute
        the agent workflow.
        """
        logger.info("="*70)
        logger.info("running full assessment pipeline")
        logger.info("="*70)
        
        Path(self.config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
        preprocessor_path = self.config['paths']['tuning']['preprocessor']
        self._load_and_preprocess(preprocessor_path)
        logger.info(f"Training & comparing models: {self.model_types}")
        self.trainer = ModelTrainer(self.config)
        
        for model_name in self.model_types:
            logger.info(f"Training {model_name}.")
            self.trainer.train_model(
                model_name,
                self.X_train, self.y_train,
                self.X_val, self.y_val
            )
        
        comparison = self.trainer.compare_models()
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
        logger.info("Saving models and results...")
        self.trainer.save_models(output_dir='models')
        logger.info("Models and results.json saved to models/")
        
        logger.info("Running GenAI interpretation")
        try:
            self.genai = GenAIInterpreter()
            
            champion_name, champion_results = get_champion_model(self.trainer.results)
            logger.info(f"Generating AI insights for champion model: {champion_name}")
            fi = champion_results.get('feature_importance', {})
            print("\n--- GenAI Feature Analysis ---")
            print(self.genai.interpret_feature_importance(fi))
            
            metrics = {
                'roc_auc': champion_results.get('roc_auc', 0),
                'accuracy': champion_results.get('accuracy', 0),
                'f1_score': champion_results.get('f1_score', 0),
                'precision': champion_results.get('precision', 0),
                'recall': champion_results.get('recall', 0),
                'net_value': champion_results.get('net_value', 0)
            }
            print("\n--- GenAI Performance Analysis ---")
            print(self.genai.interpret_model_performance(metrics, champion_name))
            
        except Exception as e:
            logger.warning(f"GenAI skipped: {e}")

        logger.info("Running agentic AI workflow")
        try:
            self.agent = DataScienceAgent(
                config=self.config,
                max_iterations=self.config['agent']['max_iterations']
            )

            print("\n--- Agent Task: General Analysis ---")
            res = self.agent.execute_task("Analyze drivers of no-shows and provide recommendations")
            print("Agent analysis completed. See results in agent output.")

        except Exception as e:
            logger.warning(f"Agent skipped: {e}")
            
        logger.info("Assessment pipeline done")


def main():
    parser = argparse.ArgumentParser(description='Hotel No-Show ML Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'tune', 'retrain'],
                        help='Pipeline mode')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of tuning trials')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to train/tune')
    
    args = parser.parse_args()
    
    pipeline = HotelNoShowPipeline(args.config, model_types=args.models)
    
    if args.mode == 'tune':
        pipeline.run_tuning(n_trials=args.n_trials)
    elif args.mode == 'retrain':
        pipeline.run_retrain()
    else:
        pipeline.run_full_assessment()


if __name__ == "__main__":
    main()
