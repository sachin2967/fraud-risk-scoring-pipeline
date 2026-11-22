"""Streamlit cockpit for the real-data card-fraud benchmark."""

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", ROOT / "models" / "credit_card_fraud_model.joblib"))
REPORT_PATH = Path(os.environ.get("FRAUD_REPORT_PATH", ROOT / "real_data_benchmark_report.json"))

st.set_page_config(page_title="Card Fraud Risk Lab", page_icon="💳", layout="wide")
st.title("💳 Real-Data Card Fraud Risk Lab")
st.caption("ULB / Worldline public benchmark · anonymized PCA features · 2013 data")

try:
    artifact = joblib.load(MODEL_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    warmup = pd.DataFrame(
        [[0.0] * len(artifact["feature_names"])], columns=artifact["feature_names"]
    )
    artifact["model"].predict_proba(warmup)
except (FileNotFoundError, OSError, EOFError, KeyError, json.JSONDecodeError):
    st.error("Real-data model artifacts are missing. Run `python scripts/evaluate_openml_benchmark.py` first.")
    st.stop()

metrics = report["test_metrics"]
summary = st.columns(4)
summary[0].metric("Held-out PR-AUC", f"{metrics['average_precision']:.4f}")
summary[1].metric("Fraud prevalence", f"{metrics['prevalence_baseline_average_precision']:.3%}")
summary[2].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
summary[3].metric("Model threshold", f"{artifact['threshold']:.5f}")

st.info("This model is trained on real labeled historical card transactions. The dataset is from 2013 and is not UPI data or evidence of current Indian payment performance. The source does not document Amount's currency, so the dashboard does not relabel it as INR.")
st.subheader("Score a transaction feature vector")
st.write("Enter the dataset's anonymized PCA features V1–V28. These values have no direct business interpretation.")

with st.form("score_transaction"):
    transaction_id = st.text_input("Transaction ID", "TXN_DEMO_001")
    amount = st.number_input("Transaction amount (source unit)", min_value=0.0, value=1499.0, step=100.0, format="%.2f")
    raw = st.text_area("V1–V28, comma separated", ",".join(["0"] * 28), height=80)
    submitted = st.form_submit_button("Score risk", type="primary")

if submitted:
    try:
        values = np.asarray([float(value.strip()) for value in raw.split(",")], dtype=np.float64)
        if len(values) != 28 or not np.isfinite(values).all():
            raise ValueError("Enter exactly 28 finite numeric values.")
        features = pd.DataFrame(
            [np.concatenate([values, [amount]])], columns=artifact["feature_names"]
        )
        started = time.perf_counter()
        probability = float(artifact["model"].predict_proba(features)[0, 1])
        latency_ms = (time.perf_counter() - started) * 1000
        threshold = float(artifact["threshold"])
        decision = "REVIEW_OR_STEP_UP" if probability >= threshold else "APPROVE_DEMO_ONLY"
        left, right = st.columns(2)
        left.metric("Uncalibrated risk score (0–1)", f"{probability:.4f}")
        right.metric("Live model latency", f"{latency_ms:.2f} ms", help="In-process prediction only; excludes HTTP and infrastructure.")
        st.metric("Recommended action", decision)
        st.caption(f"Transaction {transaction_id} · {amount:,.2f} source units · currency not documented · validation-selected cost threshold")
    except ValueError as exc:
        st.error(str(exc))

with st.expander("Evaluation protocol and limitations"):
    st.json({"dataset": report["dataset"], "split": report["split"], "threshold_policy": report["model"], "test_metrics": metrics, "limitations": report["limitations"]})
