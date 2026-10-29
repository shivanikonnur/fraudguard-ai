from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from fraudguard.data import load_dataset, TARGET
from fraudguard.metrics import choose_threshold, calculate_metrics
from fraudguard.plots import save_evaluation_plots, save_feature_importance

RANDOM_STATE = 42


def make_models():
    """Return reproducible baselines and tree-based models."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1500,
                solver="liblinear", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def train(dataset_path, artifacts_dir="artifacts", reports_dir="reports"):
    df = load_dataset(dataset_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified splits preserve rare fraud examples in every partition.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison = {}
    candidates = {}

    for name, model in make_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        valid_probabilities = model.predict_proba(X_valid)[:, 1]
        threshold, validation_f2 = choose_threshold(y_valid, valid_probabilities)

        test_probabilities = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, test_probabilities, threshold)
        metrics["validation_f2_at_selected_threshold"] = validation_f2
        comparison[name] = metrics
        candidates[name] = {
            "model": model,
            "threshold": threshold,
            "validation_f2": validation_f2,
            "test_probabilities": test_probabilities,
        }

        print(
            f"{name}: PR-AUC={metrics['average_precision_pr_auc']:.4f}, "
            f"Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}, "
            f"F2={metrics['f2']:.4f}"
        )

    # Select using validation F2 only; test metrics are reported after selection.
    best_name = max(candidates, key=lambda n: candidates[n]["validation_f2"])
    best = candidates[best_name]
    bundle = {
        "model": best["model"],
        "threshold": best["threshold"],
        "model_name": best_name,
        "feature_names": list(X.columns),
        "target_name": TARGET,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "dataset_fraud_count": int(y.sum()),
    }
    joblib.dump(bundle, artifacts_dir / "fraudguard_model.joblib")

    selected_metrics = comparison[best_name]
    save_evaluation_plots(
        y_test, best["test_probabilities"], best["threshold"], figures_dir
    )
    final_model = best["model"]
    importance_model = (
        final_model.named_steps["model"]
        if hasattr(final_model, "named_steps") else final_model
    )
    save_feature_importance(
        importance_model, list(X.columns), figures_dir / "feature_importance.png"
    )

    report = {
        "project": "FraudGuard AI",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dataset": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "fraud_count": int(y.sum()),
            "fraud_rate": float(y.mean()),
        },
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_valid)),
            "test": int(len(X_test)),
        },
        "selected_model": best_name,
        "selection_rule": "Highest validation F2 score; test split not used for selection",
        "models": comparison,
        "selected_model_test_metrics": selected_metrics,
        "limitations": [
            "Historical public dataset; not representative of every current payment system.",
            "Probability scores are not guaranteed to be calibrated.",
            "Threshold depends on validation data and operational costs.",
            "Model output is a risk signal, not proof of fraud.",
        ],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([
        {"model": name, **values} for name, values in comparison.items()
    ]).to_csv(reports_dir / "model_comparison.csv", index=False)

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {artifacts_dir / 'fraudguard_model.joblib'}")
    print(f"Saved metrics: {reports_dir / 'metrics.json'}")
    print(f"Saved plots: {figures_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train FraudGuard AI models.")
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    train(args.data, args.artifacts, args.reports)


if __name__ == "__main__":
    main()
