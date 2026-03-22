"""Performance checker for CI/CD pipeline validation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceChecker:
    """Validate model metrics against minimum thresholds."""

    THRESHOLDS = {
        "roc_auc": 0.70,  # Minimum AUC
        "f1": 0.60,  # Minimum F1 score
        "precision": 0.60,  # Minimum precision
        "recall": 0.60,  # Minimum recall
    }

    def __init__(
        self,
        metrics_path: str = "reports/performance_metrics.json",
        baseline_path: str = "models/results.json",
        custom_thresholds: Dict = None,
    ):
        self.metrics_path = Path(metrics_path)
        self.baseline_path = Path(baseline_path)

        if custom_thresholds:
            self.thresholds = {**self.THRESHOLDS, **custom_thresholds}
        else:
            self.thresholds = self.THRESHOLDS

        self.current_metrics = self._load_metrics(self.metrics_path)
        self.baseline_metrics = (
            self._load_metrics(self.baseline_path) if baseline_path else None
        )

    def _load_metrics(self, path: Path) -> Dict:
        if not path.exists():
            logger.warning(f"Metrics file not found: {path}")
            return {}

        with open(path, "r") as f:
            data = json.load(f)

        if "metrics" in data:
            return data["metrics"]
        return data

    def check_minimum_thresholds(self, metrics: Optional[Dict] = None) -> bool:
        check_metrics = metrics if metrics is not None else self.current_metrics

        if not check_metrics:
            logger.error("No metrics available for checking")
            return False

        logger.info("Checking minimum performance thresholds...")

        all_passed = True
        results = []

        for metric, threshold in self.thresholds.items():
            if metric in self.current_metrics:
                current_value = self.current_metrics[metric]
                passed = current_value >= threshold
                all_passed = all_passed and passed

                status = "PASS" if passed else "FAIL"
                results.append(
                    {
                        "metric": metric,
                        "threshold": threshold,
                        "current": current_value,
                        "passed": passed,
                        "status": status,
                    }
                )

                logger.info(
                    f"{status} | {metric}: {current_value:.3f} "
                    f"(threshold: {threshold:.3f})"
                )
            else:
                logger.warning(f"Metric '{metric}' not found in current metrics")

        return all_passed

    def check_degradation(self, threshold: float = 0.05) -> bool:
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available for comparison")
            return True  # Pass if no baseline

        if not self.current_metrics:
            logger.error("No current metrics available")
            return False

        logger.info(f"Checking for degradation (threshold: {threshold*100:.1f}%)...")

        no_degradation = True

        for metric in self.thresholds.keys():
            if metric in self.baseline_metrics and metric in self.current_metrics:
                baseline = self.baseline_metrics[metric]
                current = self.current_metrics[metric]

                degradation = (current - baseline) / baseline

                if degradation < -threshold:
                    no_degradation = False
                    logger.warning(
                        f"DEGRADATION | {metric}: {baseline:.3f} -> {current:.3f} "
                        f"({degradation*100:.2f}%)"
                    )
                else:
                    logger.info(
                        f"OK | {metric}: {baseline:.3f} -> {current:.3f} "
                        f"({degradation*100:.2f}%)"
                    )

        return no_degradation


def main():
    parser = argparse.ArgumentParser(description="Check model performance")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="reports/performance_metrics.json",
        help="Path to current performance metrics",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="models/results.json",
        help="Path to baseline metrics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Degradation threshold (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--alert-on-degradation",
        action="store_true",
        help="Exit with error code if degradation detected",
    )

    args = parser.parse_args()

    checker = PerformanceChecker(
        metrics_path=args.metrics_path,
        baseline_path=args.baseline_path,
    )

    logger.info("=" * 70)
    logger.info("MINIMUM THRESHOLD CHECK")
    logger.info("=" * 70)
    thresholds_passed = checker.check_minimum_thresholds()

    logger.info("")
    logger.info("=" * 70)
    logger.info("DEGRADATION CHECK")
    logger.info("=" * 70)
    no_degradation = checker.check_degradation(threshold=args.threshold)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if thresholds_passed and no_degradation:
        logger.info("All checks passed - model performance is acceptable")
        sys.exit(0)
    elif not thresholds_passed:
        logger.error("❌ Model does not meet minimum performance thresholds")
        sys.exit(1)
    elif not no_degradation and args.alert_on_degradation:
        logger.error("❌ Significant performance degradation detected")
        sys.exit(1)
    else:
        logger.warning("Performance degradation detected but within tolerance")
        sys.exit(0)


if __name__ == "__main__":
    main()
