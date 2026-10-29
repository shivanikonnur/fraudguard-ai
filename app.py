from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from fraudguard.data import validate_feature_frame

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudGuard AI")
st.caption("Explainable machine-learning prototype for credit-card fraud risk analysis.")

MODEL_PATH = Path("artifacts/fraudguard_model.joblib")
METRICS_PATH = Path("reports/metrics.json")

st.warning(
    "Educational demo only. A high score is not proof of fraud. "
    "Do not use this prototype to make real financial decisions."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Follow README.md to download the dataset and train the models.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle["threshold"])
features = bundle["feature_names"]

with st.sidebar:
    st.header("Model information")
    st.write(f"**Selected model:** {bundle['model_name']}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.write(f"**Expected features:** {len(features)}")
    st.caption("Model and threshold are trained from your local dataset.")

tab1, tab2, tab3 = st.tabs(["Batch scoring", "Single transaction", "Evaluation"])

with tab1:
    st.subheader("Upload a CSV to score transactions")
    st.write("CSV must contain the same numeric feature columns used during training.")
    upload = st.file_uploader("Transaction CSV", type=["csv"], key="batch")
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            frame = frame.drop(columns=["Class"], errors="ignore")
            X = validate_feature_frame(frame, features)
            probabilities = model.predict_proba(X)[:, 1]
            result = frame.copy()
            result["fraud_risk_score"] = probabilities
            result["predicted_fraud"] = (probabilities >= threshold).astype(int)
            st.dataframe(result.head(100), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="fraudguard_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score file: {exc}")

with tab2:
    st.subheader("Score one transaction")
    st.write("Enter numeric feature values. The default values are zero placeholders.")
    with st.form("single_transaction"):
        values = {}
        cols = st.columns(3)
        for index, feature in enumerate(features):
            with cols[index % 3]:
                default = 0.0
                values[feature] = st.number_input(
                    feature, value=default, format="%.6f", key=f"single_{feature}"
                )
        submitted = st.form_submit_button("Estimate risk")
    if submitted:
        row = pd.DataFrame([values], columns=features)
        probability = float(model.predict_proba(row)[:, 1][0])
        prediction = int(probability >= threshold)
        c1, c2 = st.columns(2)
        c1.metric("Estimated fraud risk score", f"{probability:.2%}")
        c2.metric("Model classification", "Flag for review" if prediction else "Not flagged")
        st.caption("Synthetic placeholder inputs may not represent a realistic transaction.")

with tab3:
    st.subheader("Held-out test evaluation")
    if METRICS_PATH.exists():
        import json
        report = json.loads(METRICS_PATH.read_text())
        metrics = report.get("selected_model_test_metrics", {})
        cols = st.columns(4)
        cols[0].metric("PR-AUC", f"{metrics.get('average_precision_pr_auc', 0):.4f}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
        cols[3].metric("F2 score", f"{metrics.get('f2', 0):.4f}")
        st.json(report.get("models", {}))
        for filename in [
            "confusion_matrix.png",
            "precision_recall_curve.png",
            "roc_curve.png",
            "feature_importance.png",
        ]:
            path = Path("reports/figures") / filename
            if path.exists():
                st.image(str(path), caption=filename.replace("_", " ").title())
    else:
        st.info("Evaluation report not found. Train the models first.")
