import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix, balanced_accuracy_score
)


def choose_threshold(y_true, probabilities, beta: float = 2.0):
    """Choose a threshold maximizing F-beta on validation data only."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def calculate_metrics(y_true, probabilities, threshold: float):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision_pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
