import numpy as np
from fraudguard.metrics import choose_threshold, calculate_metrics


def test_choose_threshold_returns_valid_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.7, 0.8, 0.9])
    threshold, score = choose_threshold(y, p)
    assert 0.01 <= threshold <= 0.99
    assert 0 <= score <= 1


def test_calculate_metrics_counts_confusion_matrix():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.9, 0.2])
    result = calculate_metrics(y, p, 0.5)
    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 1
    assert 0 <= result["average_precision_pr_auc"] <= 1
