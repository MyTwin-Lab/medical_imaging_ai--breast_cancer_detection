"""Evaluation metrics for BI-RADS classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
)


def compute_metrics(labels: list, preds: list, probs: np.ndarray,
                    class_names: list, mode: str = "binary") -> dict:
    """
    Compute classification metrics.

    Args:
        labels:      ground-truth integer labels
        preds:       predicted integer labels
        probs:       softmax probabilities, shape (N, C)
        class_names: list of class name strings
        mode:        "binary" or "multiclass"

    Returns:
        dict with accuracy, auc, f1, report, confusion_matrix
    """
    labels = np.array(labels)
    preds  = np.array(preds)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)

    if mode == "binary":
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr",
                                average="weighted")
        except ValueError:
            auc = float("nan")

    report = classification_report(labels, preds,
                                   target_names=class_names,
                                   zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
    }
