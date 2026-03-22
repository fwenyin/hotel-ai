"""Integration tests for the monitoring modules.

Tests DriftDetector, ModelMonitor, and PerformanceChecker
end-to-end using real data from noshow.db and trained model artifacts.
"""

import json
from pathlib import Path

import pytest
from sklearn.model_selection import train_test_split

from src.data.loader import load_and_split_data
from src.data.preprocessor import preprocess_splits
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.performance_checker import PerformanceChecker


@pytest.fixture(scope="module")
def raw_splits(sample_dataframe):
    """Split sample data into train/test for drift detection."""
    train_df, test_df = train_test_split(
        sample_dataframe,
        test_size=0.2,
        random_state=42,
        stratify=sample_dataframe["no_show"],
    )
    return train_df, test_df


@pytest.fixture(scope="module")
def real_preprocessed_test_data(project_root, config, database_exists):
    """Preprocessed test data from real DB, compatible with the production model."""
    if not database_exists:
        pytest.skip("Database not found")
    preprocessor_path = project_root / "models" / "preprocessor.joblib"
    if not preprocessor_path.exists():
        pytest.skip("Fitted preprocessor not found")

    _, train_df, val_df, test_df = load_and_split_data(config)
    preprocessor, _, _, _, _, X_test, y_test = preprocess_splits(
        config,
        train_df,
        val_df,
        test_df,
        preprocessor_path=str(preprocessor_path),
        fit=False,
    )
    test_data = X_test.copy()
    test_data["no_show"] = y_test.astype(int).values
    return test_data


@pytest.mark.integration
class TestDriftDetector:
    """Integration tests for data drift detection."""

    def test_drift_detector_initializes(self, raw_splits):
        """Test that DriftDetector initializes with train/test data."""
        train_df, test_df = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=test_df,
            target_column="no_show",
        )
        assert detector.reference_data is not None
        assert detector.current_data is not None
        assert (
            len(detector.numerical_features) > 0
            or len(detector.categorical_features) > 0
        )

    def test_generate_drift_report(self, raw_splits, temp_dir):
        """Test that drift report HTML is generated."""
        train_df, test_df = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=test_df,
            target_column="no_show",
        )
        output_path = str(Path(temp_dir) / "drift_report.html")
        snapshot = detector.generate_drift_report(output_path)

        assert snapshot is not None
        assert Path(output_path).exists(), "Drift report HTML should be created"
        assert Path(output_path).stat().st_size > 0, "Drift report should not be empty"

    def test_run_drift_tests(self, raw_splits, temp_dir):
        """Test that drift tests run and return a pass/fail result."""
        train_df, test_df = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=test_df,
            target_column="no_show",
        )
        output_path = str(Path(temp_dir) / "drift_tests.html")
        snapshot, passed = detector.run_drift_tests(output_path)

        assert snapshot is not None
        assert isinstance(passed, bool)
        assert Path(output_path).exists(), "Drift test report should be created"

    def test_get_drift_metrics(self, raw_splits):
        """Test that drift metrics contain all required fields."""
        train_df, test_df = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=test_df,
            target_column="no_show",
        )
        metrics = detector.get_drift_metrics()

        required_keys = [
            "timestamp",
            "n_reference",
            "n_current",
            "n_features",
            "dataset_drift",
            "drift_share",
            "n_drifted_features",
            "drifted_features",
            "critical_drift",
        ]
        for key in required_keys:
            assert key in metrics, f"Drift metrics should contain '{key}'"

        assert metrics["n_reference"] == len(train_df)
        assert metrics["n_current"] == len(test_df)
        assert 0.0 <= metrics["drift_share"] <= 1.0
        assert isinstance(metrics["drifted_features"], list)

    def test_save_drift_metrics(self, raw_splits, temp_dir):
        """Test that drift metrics are saved as valid JSON."""
        train_df, test_df = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=test_df,
            target_column="no_show",
        )
        output_path = str(Path(temp_dir) / "drift_metrics.json")
        detector.save_drift_metrics(output_path)

        assert Path(output_path).exists(), "Drift metrics JSON should be created"
        with open(output_path) as f:
            saved = json.load(f)
        assert "drift_share" in saved
        assert "critical_drift" in saved

    def test_same_data_no_drift(self, raw_splits):
        """Test that comparing data to itself shows no critical drift."""
        train_df, _ = raw_splits
        detector = DriftDetector(
            reference_data=train_df,
            current_data=train_df,
            target_column="no_show",
        )
        metrics = detector.get_drift_metrics()

        assert not metrics[
            "critical_drift"
        ], "Identical data should not trigger critical drift"


@pytest.mark.integration
class TestModelMonitor:
    """Integration tests for model performance monitoring."""

    def test_model_monitor_initializes(self, project_root):
        """Test that ModelMonitor loads model and preprocessor."""
        model_path = project_root / "models" / "champion_model.joblib"
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Trained model artifacts not found")

        monitor = ModelMonitor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
        )
        assert monitor.model is not None
        assert monitor.preprocessor is not None

    def test_generate_performance_report(
        self, project_root, real_preprocessed_test_data, temp_dir
    ):
        """Test that performance report HTML is generated."""
        model_path = project_root / "models" / "champion_model.joblib"
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Trained model artifacts not found")

        monitor = ModelMonitor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
        )
        output_path = str(Path(temp_dir) / "performance_report.html")
        snapshot = monitor.generate_performance_report(
            test_data=real_preprocessed_test_data,
            output_path=output_path,
        )

        assert snapshot is not None
        assert Path(output_path).exists(), "Performance report HTML should be created"
        assert (
            Path(output_path).stat().st_size > 0
        ), "Performance report should not be empty"

    def test_calculate_performance_metrics(
        self, project_root, real_preprocessed_test_data
    ):
        """Test that performance metrics are calculated with expected keys."""
        model_path = project_root / "models" / "champion_model.joblib"
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Trained model artifacts not found")

        monitor = ModelMonitor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
        )
        metrics = monitor.calculate_performance_metrics(real_preprocessed_test_data)

        assert "timestamp" in metrics
        assert "n_samples" in metrics
        assert "metrics" in metrics
        assert metrics["n_samples"] == len(real_preprocessed_test_data)

        for key in ["accuracy", "precision", "recall", "f1_score"]:
            assert key in metrics["metrics"], f"Metrics should contain '{key}'"
            assert (
                0.0 <= metrics["metrics"][key] <= 1.0
            ), f"{key} should be between 0 and 1"

    def test_save_performance_metrics(
        self, project_root, real_preprocessed_test_data, temp_dir
    ):
        """Test that performance metrics are saved as valid JSON."""
        model_path = project_root / "models" / "champion_model.joblib"
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        if not model_path.exists() or not preprocessor_path.exists():
            pytest.skip("Trained model artifacts not found")

        monitor = ModelMonitor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
        )
        metrics = monitor.calculate_performance_metrics(real_preprocessed_test_data)
        output_path = str(Path(temp_dir) / "performance_metrics.json")
        monitor.save_performance_metrics(metrics, output_path)

        assert Path(output_path).exists(), "Performance metrics JSON should be created"
        with open(output_path) as f:
            saved = json.load(f)
        assert "metrics" in saved

    def test_check_performance_degradation(
        self, project_root, real_preprocessed_test_data
    ):
        """Test degradation check against baseline metrics."""
        model_path = project_root / "models" / "champion_model.joblib"
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        baseline_path = project_root / "models" / "results.json"
        if not all(p.exists() for p in [model_path, preprocessor_path, baseline_path]):
            pytest.skip("Trained model artifacts or baseline not found")

        monitor = ModelMonitor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            baseline_metrics_path=str(baseline_path),
        )
        metrics = monitor.calculate_performance_metrics(real_preprocessed_test_data)
        degraded = monitor.check_performance_degradation(metrics, threshold=0.05)

        assert isinstance(degraded, bool)


@pytest.mark.integration
class TestPerformanceChecker:
    """Integration tests for CI/CD performance threshold checking."""

    def test_checker_initializes_with_metrics(self, project_root):
        """Test that PerformanceChecker loads metrics files."""
        metrics_path = project_root / "reports" / "performance_metrics.json"
        baseline_path = project_root / "models" / "results.json"
        if not metrics_path.exists() or not baseline_path.exists():
            pytest.skip("Metrics files not found")

        checker = PerformanceChecker(
            metrics_path=str(metrics_path),
            baseline_path=str(baseline_path),
        )
        assert checker.current_metrics, "Current metrics should be loaded"

    def test_check_minimum_thresholds(self, project_root):
        """Test that model meets minimum performance thresholds."""
        metrics_path = project_root / "reports" / "performance_metrics.json"
        if not metrics_path.exists():
            pytest.skip("Performance metrics not found")

        checker = PerformanceChecker(metrics_path=str(metrics_path))
        passed = checker.check_minimum_thresholds()

        assert isinstance(passed, bool)

    def test_check_degradation(self, project_root):
        """Test degradation check compares current vs baseline."""
        metrics_path = project_root / "reports" / "performance_metrics.json"
        baseline_path = project_root / "models" / "results.json"
        if not metrics_path.exists() or not baseline_path.exists():
            pytest.skip("Metrics or baseline files not found")

        checker = PerformanceChecker(
            metrics_path=str(metrics_path),
            baseline_path=str(baseline_path),
        )
        no_degradation = checker.check_degradation(threshold=0.05)

        assert isinstance(no_degradation, bool)

    def test_custom_thresholds(self, temp_dir):
        """Test that custom thresholds override defaults."""
        metrics = {
            "metrics": {"roc_auc": 0.85, "f1": 0.75, "precision": 0.80, "recall": 0.70}
        }
        metrics_path = str(Path(temp_dir) / "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        checker = PerformanceChecker(
            metrics_path=metrics_path,
            baseline_path=str(Path(temp_dir) / "nonexistent_baseline.json"),
            custom_thresholds={"roc_auc": 0.90},
        )
        passed = checker.check_minimum_thresholds()

        assert not passed, "Should fail when ROC-AUC 0.85 < custom threshold 0.90"

    def test_all_thresholds_pass(self, temp_dir):
        """Test that good metrics pass all default thresholds."""
        metrics = {
            "metrics": {"roc_auc": 0.85, "f1": 0.75, "precision": 0.80, "recall": 0.70}
        }
        metrics_path = str(Path(temp_dir) / "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        checker = PerformanceChecker(
            metrics_path=metrics_path,
            baseline_path=str(Path(temp_dir) / "nonexistent_baseline.json"),
        )
        passed = checker.check_minimum_thresholds()

        assert passed, "Metrics above all default thresholds should pass"
