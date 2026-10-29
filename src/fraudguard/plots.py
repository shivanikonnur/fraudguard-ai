from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_evaluation_plots(y_true, probabilities, threshold, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, predictions, labels=[0, 1],
        display_labels=["Genuine", "Fraud"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("FraudGuard AI — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Precision–Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)


def save_feature_importance(model, feature_names, output_path, top_n=20):
    """Save feature importance for supported tree models or linear coefficients."""
    import numpy as np
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return

    pairs = sorted(zip(feature_names, values), key=lambda x: x[1], reverse=True)[:top_n]
    names, scores = zip(*pairs)
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.28)))
    sns.barplot(x=list(scores), y=list(names), ax=ax)
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance (absolute coefficient for Logistic Regression)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
