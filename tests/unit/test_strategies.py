"""Tests for metrics calculations."""

import numpy as np


class TestMetricsCalculations:
    """Tests for the calculate_all_metrics function."""

    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        from src.models.metrics import calculate_all_metrics

        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.6, 0.85, 0.15, 0.95, 0.05])

        result = calculate_all_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "roc_auc" in result
        assert "pr_auc" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_correct_predictions(self):
        """Test metrics when all predictions are correct."""
        from src.models.metrics import calculate_all_metrics

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])  # All correct
        y_pred_proba = np.array([0.9, 0.1, 0.85, 0.15])

        result = calculate_all_metrics(y_true, y_pred, y_pred_proba)

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics when all predictions are wrong."""
        from src.models.metrics import calculate_all_metrics

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])  # All wrong
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = calculate_all_metrics(y_true, y_pred, y_pred_proba)

        assert result["accuracy"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_zero_division_handling(self):
        """Test that zero division is handled gracefully."""
        from src.models.metrics import calculate_all_metrics

        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.15, 0.25])

        result = calculate_all_metrics(y_true, y_pred, y_pred_proba)

        assert "precision" in result
        assert "recall" in result
