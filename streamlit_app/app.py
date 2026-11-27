"""Streamlit cockpit for the real-data card-fraud benchmark.

All scoring goes through FastAPI so dashboard requests share authentication,
idempotency, durable storage and delayed-label handling with other API clients.
"""

import numpy as np
import streamlit as st

from streamlit_app.api_client import (
    APIClientError,
    get_benchmark,
    get_readiness,
    score_transaction,
)

st.set_page_config(page_title="Card Fraud Risk Lab", page_icon="💳", layout="wide")
st.title("💳 Real-Data Card Fraud Risk Lab")
st.caption("ULB / Worldline public benchmark · anonymized PCA features · 2013 data")

try:
    readiness = get_readiness()
    report = get_benchmark()
except APIClientError as exc:
    st.error(f"Scoring service unavailable: {exc}")
    st.info(
        "Start FastAPI first and configure FRAUD_API_URL. In Docker Compose, "
        "the dashboard uses the internal fraud-api service address."
    )
    st.stop()

metrics = report["test_metrics"]
summary = st.columns(4)
summary[0].metric("Held-out PR-AUC", f"{metrics['average_precision']:.4f}")
summary[1].metric(
    "Fraud prevalence", f"{metrics['prevalence_baseline_average_precision']:.3%}"
)
summary[2].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
summary[3].metric("Model threshold", f"{report['model']['threshold']:.5f}")

st.success(
    f"Connected to scoring API · model {readiness['model_version'][:12]} · "
    f"feature schema {readiness['feature_version']}"
)
st.info(
    "This model is trained on real labeled historical card transactions. The "
    "dataset is from 2013 and is not UPI data or evidence of current Indian "
    "payment performance. The source does not document Amount's currency, so "
    "the dashboard does not relabel it as INR."
)
st.subheader("Score a transaction feature vector")
st.write(
    "Enter the dataset's anonymized PCA features V1–V28. These values have no "
    "direct business interpretation. Scoring is performed by FastAPI and "
    "successfully committed predictions are stored in its SQLite ledger."
)

with st.form("score_transaction"):
    transaction_id = st.text_input("Transaction ID", "TXN_DEMO_001")
    amount = st.number_input(
        "Transaction amount (source unit)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=1499.0,
        step=100.0,
        format="%.2f",
    )
    raw = st.text_area(
        "V1–V28, comma separated", ",".join(["0"] * 28), height=80
    )
    submitted = st.form_submit_button("Score risk", type="primary")

if submitted:
    try:
        if not transaction_id or not transaction_id.strip():
            raise ValueError("Transaction ID is required.")
        values = np.asarray(
            [float(value.strip()) for value in raw.split(",")], dtype=np.float64
        )
        if len(values) != 28 or not np.isfinite(values).all():
            raise ValueError("Enter exactly 28 finite numeric values.")
        result = score_transaction(
            {
                "transaction_id": transaction_id.strip(),
                "amount": amount,
                "pca_features": values.tolist(),
            }
        )
        left, right = st.columns(2)
        left.metric(
            "Uncalibrated risk score (0–1)",
            f"{result['fraud_risk_score']:.4f}",
        )
        right.metric(
            "Model inference latency",
            f"{result['model_latency_ms']:.2f} ms",
            help="In-process model prediction only; excludes HTTP and infrastructure.",
        )
        st.metric("Recommended action", result["decision"])
        st.caption(
            f"Transaction {result['transaction_id']} · prediction "
            f"{result['prediction_id']} · {amount:,.2f} source units · "
            "currency not documented"
        )
        st.caption(
            "The score is uncalibrated. The demo action is not a payment "
            "authorization. The prediction has been persisted by the API."
        )
    except ValueError as exc:
        st.error(str(exc))
    except APIClientError as exc:
        st.error(f"Scoring request failed: {exc}")

with st.expander("Evaluation protocol and limitations"):
    st.json(
        {
            "dataset": report["dataset"],
            "split": report["split"],
            "threshold_policy": report["model"],
            "test_metrics": metrics,
            "limitations": report["limitations"],
        }
    )
