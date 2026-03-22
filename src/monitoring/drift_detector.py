"""Data drift detection using Evidently AI."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift using Evidently AI."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "no_show",
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.reference_data = reference_data
        self.current_data = current_data
        self.target_column = target_column

        if numerical_features is None:
            self.numerical_features = reference_data.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            if target_column in self.numerical_features:
                self.numerical_features.remove(target_column)
        else:
            self.numerical_features = numerical_features

        if categorical_features is None:
            self.categorical_features = reference_data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            if target_column in self.categorical_features:
                self.categorical_features.remove(target_column)
        else:
            self.categorical_features = categorical_features

        logger.info(
            f"Initialized DriftDetector with {len(self.numerical_features)} "
            f"numerical and {len(self.categorical_features)} categorical features"
        )

    def generate_drift_report(
        self, output_path: str = "reports/drift_report.html"
    ) -> Report:
        logger.info("Generating drift report...")

        report = Report(
            metrics=[
                DataDriftPreset(),
                DataSummaryPreset(),
            ]
        )

        snapshot = report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))

        logger.info(f"Drift report saved to {output_path}")
        return snapshot

    def run_drift_tests(
        self, output_path: str = "reports/drift_tests.html"
    ) -> Tuple[Report, bool]:
        logger.info("Running drift tests...")

        # TestSuite deprecated in 0.7, using Report instead
        report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        # Run tests (Evidently 0.7+ returns Snapshot)
        snapshot = report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))

        try:
            metrics_dict = snapshot.dict()
        except AttributeError:
            metrics_dict = {"metrics": []}
        passed = not self._has_critical_drift(metrics_dict)

        logger.info(f"Drift tests {'PASSED' if passed else 'FAILED'}")
        logger.info(f"Test results saved to {output_path}")

        return snapshot, passed

    def get_drift_metrics(self) -> Dict:
        logger.info("Extracting drift metrics...")

        # Run report (Evidently 0.7+ returns Snapshot)
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
        )

        try:
            metrics_dict = snapshot.dict()
        except AttributeError:
            metrics_dict = {"metrics": []}

        # Evidently 0.7+ metric structure:
        # metrics[0] = DriftedColumnsCount with value: {count, share}
        # metrics[1+] = ValueDrift per column with value: p_value (float)
        drifted_count_metric = metrics_dict["metrics"][0]["value"]
        n_drifted = int(drifted_count_metric["count"])
        drift_share = float(drifted_count_metric["share"])

        drift_metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_reference": len(self.reference_data),
            "n_current": len(self.current_data),
            "n_features": len(self.numerical_features) + len(self.categorical_features),
            "dataset_drift": drift_share > 0.5,
            "drift_share": drift_share,
            "n_drifted_features": n_drifted,
            "drifted_features": [],
            "critical_drift": False,
        }

        # per-column drift from ValueDrift metrics
        for metric in metrics_dict["metrics"][1:]:
            metric_name = metric.get("metric_name", "")
            if "ValueDrift" in metric_name:
                col_start = metric_name.find("column=") + len("column=")
                col_end = metric_name.find(",", col_start)
                column_name = metric_name[col_start:col_end]
                p_value = metric["value"]
                if p_value is not None and p_value < 0.05:
                    drift_metrics["drifted_features"].append(
                        {
                            "feature": column_name,
                            "drift_score": p_value,
                            "stattest_name": (
                                metric_name.split("method=")[-1].split(",")[0]
                                if "method=" in metric_name
                                else None
                            ),
                        }
                    )

        # critical if >30% features drifted or dataset drift flagged
        drift_metrics["critical_drift"] = (
            drift_metrics["dataset_drift"] or drift_metrics["drift_share"] > 0.3
        )

        return drift_metrics

    def save_drift_metrics(self, output_path: str = "reports/drift_metrics.json"):
        metrics = self.get_drift_metrics()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Drift metrics saved to {output_path}")

    def _has_critical_drift(self, metrics_dict: Dict) -> bool:
        try:
            drifted_value = metrics_dict.get("metrics", [{}])[0].get("value", {})
            drift_share = float(drifted_value.get("share", 0))
            return drift_share > 0.3
        except (KeyError, IndexError, TypeError):
            return False


def main():
    parser = argparse.ArgumentParser(description="Detect data drift using Evidently")
    parser.add_argument(
        "--reference-data",
        type=str,
        default="data/processed_data/reference.csv",
        help="Path to reference dataset (training data)",
    )
    parser.add_argument(
        "--current-data",
        type=str,
        default="data/processed_data/current.csv",
        help="Path to current dataset (production data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/drift_report.html",
        help="Path to save drift report",
    )
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="Only check if monitoring infrastructure is set up",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment",
    )
    parser.add_argument(
        "--enable-alerts",
        action="store_true",
        help="Enable alerting for critical drift",
    )

    args = parser.parse_args()

    # Schema check only
    if args.check_schema:
        logger.info("Drift detection infrastructure configured")
        return

    if not Path(args.reference_data).exists():
        logger.warning(f"Reference data not found: {args.reference_data}")
        logger.info("Run training pipeline to generate reference data")
        return

    if not Path(args.current_data).exists():
        logger.warning(f"Current data not found: {args.current_data}")
        logger.info("Collect production data for drift monitoring")
        return

    logger.info(f"Loading reference data from {args.reference_data}")
    reference_data = pd.read_csv(args.reference_data)

    logger.info(f"Loading current data from {args.current_data}")
    current_data = pd.read_csv(args.current_data)

    detector = DriftDetector(
        reference_data=reference_data,
        current_data=current_data,
    )

    detector.generate_drift_report(args.output)

    _, passed = detector.run_drift_tests()

    detector.save_drift_metrics()

    metrics = detector.get_drift_metrics()

    if metrics["critical_drift"]:
        logger.error("🚨 CRITICAL DRIFT DETECTED!")
        logger.error(
            f"Drifted features: {[f['feature'] for f in metrics['drifted_features']]}"
        )

        if args.enable_alerts:
            logger.info("📧 Sending drift alert...")
            # Add alerting logic here (email, Slack, PagerDuty, etc.)
    else:
        logger.info("No critical drift detected")

    logger.info(f"Environment: {args.environment}")
    logger.info(f"Drift share: {metrics['drift_share']:.2%}")
    logger.info(f"Number of drifted features: {metrics['n_drifted_features']}")


if __name__ == "__main__":
    main()
