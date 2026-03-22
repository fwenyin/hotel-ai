"""Tests for MLOps module."""

import mlflow
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.mlops.mlflow_pipeline import MLflowModelPipeline, run_mlflow_experiment
from src.mlops.mlflow_tracker import MLflowTracker, get_or_create_experiment


class TestMLflowTracker:
    """Tests for the MLflowTracker class."""

    def test_tracker_start_run_context_manager(self, temp_mlruns):
        """Test that start_run works as context manager."""
        tracker = MLflowTracker(
            experiment_name="test_context", tracking_uri=temp_mlruns
        )

        with tracker.start_run(run_name="test_run"):
            run_id = tracker.get_run_id()
            assert run_id is not None

    def test_tracker_log_params(self, temp_mlruns):
        """Test logging parameters."""
        tracker = MLflowTracker(experiment_name="test_params", tracking_uri=temp_mlruns)

        with tracker.start_run(run_name="param_test"):
            tracker.log_params(
                {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 5}
            )

    def test_tracker_log_metrics(self, temp_mlruns):
        """Test logging metrics."""
        tracker = MLflowTracker(
            experiment_name="test_metrics", tracking_uri=temp_mlruns
        )

        with tracker.start_run(run_name="metric_test"):
            tracker.log_metrics({"accuracy": 0.95, "roc_auc": 0.87, "f1_score": 0.91})

    def test_tracker_nested_runs(self, temp_mlruns):
        """Test nested run support."""
        tracker = MLflowTracker(experiment_name="test_nested", tracking_uri=temp_mlruns)

        with tracker.start_run(run_name="parent_run"):
            parent_id = tracker.get_run_id()

            with tracker.start_run(run_name="child_run", nested=True):
                child_id = tracker.get_run_id()
                assert child_id != parent_id

    def test_tracker_flatten_dict(self):
        """Test dictionary flattening utility."""
        nested_dict = {
            "model": {"learning_rate": 0.01, "layers": {"hidden": 256, "output": 1}},
            "data": {"train_size": 1000},
        }

        flattened = MLflowTracker._flatten_dict(nested_dict)

        assert "model.learning_rate" in flattened
        assert "model.layers.hidden" in flattened
        assert "data.train_size" in flattened
        assert flattened["model.learning_rate"] == 0.01


class TestGetOrCreateExperiment:
    """Tests for get_or_create_experiment function."""

    def test_creates_new_experiment(self, temp_mlruns):
        """Test creating a new experiment."""
        mlflow.set_tracking_uri(temp_mlruns)

        exp_id = get_or_create_experiment(
            experiment_name="new_test_experiment", tags={"project": "test"}
        )

        assert exp_id is not None
        assert isinstance(exp_id, str)

    def test_returns_existing_experiment(self, temp_mlruns):
        """Test returning existing experiment."""
        mlflow.set_tracking_uri(temp_mlruns)

        exp_id1 = get_or_create_experiment("existing_experiment")

        exp_id2 = get_or_create_experiment("existing_experiment")

        assert exp_id1 == exp_id2


class TestLogFeatureImportancePlot:
    """Tests for feature importance plot logging."""

    def test_creates_plot_artifact(self, temp_mlruns):
        """Test that feature importance logging completes without error."""
        tracker = MLflowTracker(
            experiment_name="test_feature_plot", tracking_uri=temp_mlruns
        )

        with tracker.start_run(run_name="feature_plot_test"):
            feature_names = ["feature1", "feature2", "feature3"]
            importances = np.array([0.5, 0.3, 0.2])
            tracker.log_feature_importance(feature_names, importances)

            run_id = tracker.get_run_id()
            assert run_id is not None

    def test_handles_large_feature_set(self, temp_mlruns):
        """Test plotting with many features."""
        tracker = MLflowTracker(
            experiment_name="test_large_features", tracking_uri=temp_mlruns
        )

        with tracker.start_run(run_name="large_feature_test"):
            n_features = 50
            feature_names = [f"feature_{i}" for i in range(n_features)]
            importances = np.random.rand(n_features)
            tracker.log_feature_importance(feature_names, importances)

            run_id = tracker.get_run_id()
            assert run_id is not None


class TestMLflowModelPipeline:
    """Tests for the MLflowModelPipeline class."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_pipeline_run_experiment(self, train_val_data, minimal_config, temp_mlruns):
        """Test running a full experiment."""
        X_train, y_train, X_val, y_val = train_val_data

        pipeline = MLflowModelPipeline(
            config=minimal_config,
            experiment_name="test_full_experiment",
            tracking_uri=temp_mlruns,
        )

        results = pipeline.run_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            model_types=["random_forest"],  # Just one for speed
            tune_hyperparameters=False,
        )

        assert "random_forest" in results
        assert "roc_auc" in results["random_forest"]


class TestRunMlflowExperiment:
    """Tests for the run_mlflow_experiment convenience function."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_run_experiment_returns_pipeline_and_results(
        self, train_val_data, minimal_config, temp_mlruns
    ):
        """Test that run_mlflow_experiment returns expected types."""
        X_train, y_train, X_val, y_val = train_val_data

        import mlflow

        mlflow.set_tracking_uri(temp_mlruns)

        pipeline, results = run_mlflow_experiment(
            config=minimal_config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_name="test_convenience",
            model_types=["random_forest"],
            tune=False,
            artifact_location=temp_mlruns,
        )

        assert isinstance(pipeline, MLflowModelPipeline)
        assert isinstance(results, dict)


class TestModelLogging:
    """Tests for model artifact logging."""

    def test_log_model(self, train_val_data, temp_mlruns):
        """Test logging sklearn model."""
        X_train, y_train, X_val, _ = train_val_data

        tracker = MLflowTracker(
            experiment_name="test_model_log", tracking_uri=temp_mlruns
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        with tracker.start_run(run_name="model_test"):
            tracker.log_model(
                model, artifact_path="model", model_type="sklearn", X_sample=X_val
            )
