import argparse
from pathlib import Path
import joblib
import pandas as pd

from fraudguard.data import validate_feature_frame


def score_csv(input_path, model_path="artifacts/fraudguard_model.joblib",
              output_path="reports/predictions.csv"):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_names = bundle["feature_names"]

    df = pd.read_csv(input_path)
    # If the input file contains a label, exclude it from inference features.
    df = df.drop(columns=["Class"], errors="ignore")
    X = validate_feature_frame(df, feature_names)

    probabilities = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["fraud_risk_score"] = probabilities
    result["predicted_fraud"] = (probabilities >= threshold).astype(int)
    result["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.000001, 0.25, 0.60, 1.000001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Scored {len(result)} transactions -> {output}")
    print(f"Model: {bundle['model_name']} | threshold: {threshold:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Score new transactions from a CSV.")
    parser.add_argument("--input", required=True, help="CSV with model feature columns.")
    parser.add_argument("--model", default="artifacts/fraudguard_model.joblib")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    score_csv(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
