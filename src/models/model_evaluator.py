"""
Model evaluation utilities.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_cm


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = ["Show", "No-Show"],
) -> Dict:
    """Compute confusion matrix and related metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional class labels

    Returns:
        Dictionary with confusion matrix and component counts
    """
    cm = sklearn_cm(y_true, y_pred)

    return {
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }


def format_feature_importance(
    feature_names: List[str], importances: np.ndarray, top_n: Optional[int] = None
) -> pd.DataFrame:
    """Format feature importance as DataFrame.

    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Optional number of top features to return

    Returns:
        DataFrame with features sorted by importance
    """
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False, key=abs)
    if top_n:
        df = df.head(top_n)
    return df.reset_index(drop=True)


def get_champion_model(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """Get champion model (best ROC-AUC) from results.

    Args:
        results: Dictionary of model_name -> results

    Returns:
        Tuple of (champion_model_name, champion_results)
    """
    if not results:
        raise ValueError("No model results provided")

    champion = max(results.items(), key=lambda x: x[1].get("roc_auc", 0))

    return champion[0], champion[1]
