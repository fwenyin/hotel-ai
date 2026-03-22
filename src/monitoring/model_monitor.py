"""Model performance monitoring with Evidently AI."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.presets import ClassificationPreset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and detect degradation."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str = "models/preprocessor.joblib",
        baseline_metrics_path: Optional[str] = "models/results.json",
    ):

        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        self.baseline_metrics = None
        if baseline_metrics_path and Path(baseline_metrics_path).exists():
            with open(baseline_metrics_path, "r") as f:
                self.baseline_metrics = json.load(f)
            logger.info(f"Loaded baseline metrics from {baseline_metrics_path}")

    def generate_performance_report(
        self,
        test_data: pd.DataFrame,
        target_column: str = "no_show",
        output_path: str = "reports/performance_report.html",
    ) -> Report:
        logger.info("Generating performance report...")

        X_test = test_data.drop(columns=[target_column])

        y_pred = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            raw_proba = self.model.predict_proba(X_test)
            y_pred_proba = raw_proba[:, 1] if raw_proba.ndim == 2 else raw_proba
        else:
            y_pred_proba = None

        results_df = test_data.copy()
        results_df[target_column] = results_df[target_column].astype(int)
        results_df["prediction"] = y_pred.astype(int)
        if y_pred_proba is not None:
            results_df["prediction_proba"] = y_pred_proba

        # Evidently 0.7+ Dataset setup
        data_def = DataDefinition(
            classification=[
                BinaryClassification(
                    target=target_column,
                    prediction_labels="prediction",
                    prediction_probas=(
                        "prediction_proba" if y_pred_proba is not None else None
                    ),
                )
            ]
        )
        dataset = Dataset.from_pandas(results_df, data_definition=data_def)

        report = Report(
            metrics=[
                ClassificationPreset(),
            ]
        )

        snapshot = report.run(
            current_data=dataset,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))

        logger.info(f"Performance report saved to {output_path}")
        return snapshot

    def calculate_performance_metrics(
        self,
        test_data: pd.DataFrame,
        target_column: str = "no_show",
    ) -> Dict:
        logger.info("Calculating performance metrics...")

        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        y_pred = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            raw_proba = self.model.predict_proba(X_test)
            y_pred_proba = raw_proba[:, 1] if raw_proba.ndim == 2 else raw_proba
        else:
            y_pred_proba = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))

        metrics_with_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "n_samples": len(test_data),
            "metrics": metrics,
        }

        return metrics_with_metadata

    def check_performance_degradation(
        self,
        current_metrics: Dict,
        threshold: float = 0.05,
    ) -> bool:
        if self.baseline_metrics is None:
            logger.warning("No baseline metrics available for comparison")
            return False

        logger.info("Checking for performance degradation...")

        degraded = False
        degradation_report = []

        key_metrics = ["roc_auc", "f1", "precision", "recall"]

        for metric in key_metrics:
            if metric in self.baseline_metrics and metric in current_metrics["metrics"]:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics["metrics"][metric]

                degradation = (current_value - baseline_value) / baseline_value

                if degradation < -threshold:
                    degraded = True
                    degradation_report.append(
                        {
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "degradation_pct": degradation * 100,
                        }
                    )

        if degraded:
            logger.warning("Performance degradation detected:")
            for item in degradation_report:
                logger.warning(
                    f"  {item['metric']}: {item['baseline']:.3f} → {item['current']:.3f} "
                    f"({item['degradation_pct']:.2f}%)"
                )
        else:
            logger.info("No significant performance degradation")

        return degraded

    def save_performance_metrics(
        self,
        metrics: Dict,
        output_path: str = "reports/performance_metrics.json",
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Performance metrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Monitor model performance")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/champion_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default="models/preprocessor.joblib",
        help="Path to preprocessor",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed_data/test.csv",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/performance_report.html",
        help="Path to save performance report",
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        return

    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return

    logger.info(f"Loading test data from {args.test_data}")
    test_data = pd.read_csv(args.test_data)

    monitor = ModelMonitor(
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
    )

    monitor.generate_performance_report(
        test_data=test_data,
        output_path=args.output,
    )

    metrics = monitor.calculate_performance_metrics(test_data)

    monitor.save_performance_metrics(metrics)

    monitor.check_performance_degradation(metrics)

    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    for metric, value in metrics["metrics"].items():
        logger.info(f"{metric}: {value:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
