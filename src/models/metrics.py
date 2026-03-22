"""Evaluation metrics for no-show prediction.

Calculates ML metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive ML metrics including PR-AUC.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary of all evaluation metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }
