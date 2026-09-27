"""
Model Evaluation and Metrics Module.

Computes comprehensive classification metrics (Accuracy, F1, Precision,
Recall, ROC-AUC, Confusion Matrix) and visual artifacts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    artifact_dir: str | Path | None = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate a trained model against test data.

    Args:
        model: Trained scikit-learn compatible estimator.
        X_test: Test features.
        y_test: True test labels.
        artifact_dir: Optional path to save visual evaluation plots.

    Returns:
        Tuple of (metrics_dict, artifacts_dict).
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    # Compute key metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
    precision = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="binary", zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_test, y_pred)

    metrics: Dict[str, float] = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
    }

    logger.info("Evaluation metrics: Acc=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f, AUC=%.4f",
                accuracy, f1, precision, recall, roc_auc)

    artifacts: Dict[str, Any] = {
        "confusion_matrix": cm.tolist()
    }

    if artifact_dir:
        art_path = Path(artifact_dir)
        art_path.mkdir(parents=True, exist_ok=True)
        cm_file = art_path / "confusion_matrix.png"

        # Generate Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="large")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_file, dpi=120)
        plt.close(fig)

        artifacts["confusion_matrix_path"] = str(cm_file)
        logger.info("Saved confusion matrix plot to: %s", cm_file)

    return metrics, artifacts
