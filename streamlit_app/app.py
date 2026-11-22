"""Interactive dashboard for the synthetic fraud-risk scoring demo."""

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data.indian_upi_adapter import IndianUPIPaymentsAdapter
from src.models.fraud_detector import HybridFraudDetector

st.set_page_config(
    page_title="Fraud Risk Scoring Prototype",
    page_icon="💳",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", PROJECT_ROOT / "models"))
DATA_PATH = Path(os.environ.get("FRAUD_DATA_DIR", PROJECT_ROOT / "data")) / "transactions.csv"


@st.cache_resource
def load_artifacts():
    det = HybridFraudDetector.load(str(MODELS_DIR / "hybrid_fraud_detector.joblib"))
    pipe = joblib.load(str(MODELS_DIR / "feature_pipeline.joblib"))
    return det, pipe


try:
    detector, pipeline = load_artifacts()
    df_sample = pd.read_csv(DATA_PATH)
except (FileNotFoundError, OSError, EOFError, KeyError, pd.errors.EmptyDataError) as e:
    st.error(
        f"Error loading artifacts: {e}. Please run `scripts/train_and_evaluate.py` first."
    )
    st.stop()

# --- HEADER ---
st.title("💳 Fraud Risk Scoring Prototype")
st.markdown(
    "**Educational demo using synthetic data.** Cost-weighted LightGBM plus a heuristic Isolation Forest anomaly flag. "
    "The score and routing policy are not validated for payment decisions."
)

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "⚡ Live Transaction Scoring",
        "💰 Cost-Sensitive Threshold Optimizer",
        "📊 Model Benchmarks & Latency",
    ]
)

with tab1:
    st.subheader("Simulate Incoming Checkout Transaction")

    def apply_quick_preset():
        selected = st.session_state["quick_test_preset"]
        if selected.startswith("Normal Chai"):
            values = {
                "txn_amount_input": 150.0, "merchant_category_input": "Grocery",
                "hour_input": 14, "velocity_input": 1, "time_since_input": 3600.0,
                "geo_dist_input": 0.0, "device_trust_input": 0.95,
                "is_foreign_input": 0, "amount_ratio_input": 0.1,
            }
        elif selected.startswith("Midnight"):
            values = {
                "txn_amount_input": 45000.0, "merchant_category_input": "Digital_Wallet",
                "hour_input": 3, "velocity_input": 5, "time_since_input": 5.0,
                "geo_dist_input": 250.0, "device_trust_input": 0.05,
                "is_foreign_input": 0, "amount_ratio_input": 30.0,
            }
        else:
            values = {
                "txn_amount_input": 8500.0, "merchant_category_input": "Gaming",
                "hour_input": 2, "velocity_input": 5, "time_since_input": 35.0,
                "geo_dist_input": 450.0, "device_trust_input": 0.15,
                "is_foreign_input": 1, "amount_ratio_input": 6.2,
            }
        st.session_state.update(values)

    initial_values = {
        "txn_amount_input": 8500.0, "merchant_category_input": "Gaming",
        "hour_input": 2, "velocity_input": 5, "time_since_input": 35.0,
        "geo_dist_input": 450.0, "device_trust_input": 0.15,
        "is_foreign_input": 1, "amount_ratio_input": 6.2,
    }
    for key, value in initial_values.items():
        st.session_state.setdefault(key, value)

    preset = st.selectbox(
        "Quick test preset:",
        ["Custom transaction", "Normal Chai/Grocery P2M QR Payment (₹150)",
         "Midnight High-Value UPI Account Draining Attack (₹45,000)"],
        key="quick_test_preset",
        on_change=apply_quick_preset,
    )
    is_drain_preset = preset.startswith("Midnight")
    is_chai_preset = preset.startswith("Normal Chai")

    col1, col2, col3 = st.columns(3)
    with col1:
        txn_amount = st.number_input(
            "Transaction Amount (₹):", min_value=10.0, max_value=500000.0,
            format="%.2f", key="txn_amount_input"
        )
        category = st.selectbox(
            "Merchant Category:",
            options=[
                "Grocery",
                "Dining",
                "Electronics",
                "Travel",
                "Gaming",
                "Luxury_Jewelry",
                "Digital_Wallet",
            ],
            key="merchant_category_input",
        )
        hour = st.slider("Transaction Hour of Day (0-23):", 0, 23, key="hour_input")

    with col2:
        velocity = st.number_input(
            "Transactions in Last 1 Hour:", min_value=0, max_value=20,
            key="velocity_input"
        )
        time_since = st.number_input(
            "Time Since Last Txn (seconds):",
            min_value=5.0,
            max_value=86400.0,
            key="time_since_input",
        )
        geo_dist = st.number_input(
            "Geo Distance from Prev Txn (km):",
            min_value=0.0,
            max_value=5000.0,
            key="geo_dist_input",
        )

    with col3:
        device_trust = st.slider(
            "Device Trust Score (0=New/Suspicious, 1=Known):", 0.0, 1.0,
            key="device_trust_input",
        )
        is_foreign = st.selectbox(
            "Is International Transaction?", options=[0, 1],
            key="is_foreign_input",
        )
        amount_ratio = st.number_input(
            "Amount / Historical Average Ratio:",
            min_value=0.1,
            max_value=50.0,
            key="amount_ratio_input",
        )

    if st.button("🚀 Score Transaction Risk", type="primary"):
        test_dict = {
            "transaction_id": "TXN_LIVE_SIM",
            "account_id": "ACC_SIM",
            "amount": txn_amount,
            "amount_inr": txn_amount,
            "txn_type": "UPI_P2M" if is_chai_preset else "UPI_P2P" if is_drain_preset else "UPI_P2M",
            "upi_id": "merchant_store@paytm" if is_chai_preset else "sachin@okaxis",
            "old_balance": 45000.0 if is_drain_preset else 5000.0,
            "new_balance": 0.0 if is_drain_preset else (4850.0 if is_chai_preset else 5000.0 - txn_amount),
            "merchant_category": category,
            "hour_of_day": hour,
            "time_since_last_sec": time_since,
            "velocity_1h": velocity,
            "geo_distance_km": geo_dist,
            "device_trust_score": device_trust,
            "is_foreign": is_foreign,
            "amount_deviation_ratio": amount_ratio,
        }
        test_df = IndianUPIPaymentsAdapter().transform(pd.DataFrame([test_dict]))
        X_mat = pipeline.transform(test_df).values
        started = time.perf_counter()
        res = detector.predict_risk(X_mat, upi_context=test_df.iloc[0].to_dict())
        latency_ms = (time.perf_counter() - started) * 1000

        st.divider()

        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            st.metric(
                "Fraud Risk Probability", f"{res['fraud_probability'] * 100:.1f}%"
            )
            st.metric(
                "Cost-Optimal Threshold", f"{res['optimal_threshold'] * 100:.1f}%"
            )
            st.metric("Live scoring latency", f"{latency_ms:.2f} ms", help="In-process model scoring only; target under 20 ms.")
            st.metric("Transaction Amount", f"₹{txn_amount:,.2f}")

            if res["anomaly_flagged"]:
                st.warning(
                    "⚠️ Unusual feature pattern flagged by Isolation Forest (not validated as zero-day fraud)"
                )
            else:
                st.info("ℹ️ Distribution Pattern Consistent with Training History")

        with res_col2:
            st.markdown("### Operational Gateway Decision")
            dec = res["decision"]
            if dec in {"BLOCK_TRANSACTION", "BLOCK_AND_FREEZE_UPI"}:
                st.error(f"🚨 **{dec}**")
            elif dec in {"REQUIRE_STEP_UP_AUTH", "REQUIRE_UPI_PIN_OR_OTP_STEP_UP"}:
                st.warning(f"⚠️ **{dec}**")
            else:
                st.success(f"✅ **{dec}**")

            st.write(f"**Gateway Action:** {res['recommended_action']}")

with tab2:
    st.subheader("Illustrative Cost Policy")
    st.markdown("""
    Threshold selection used the validation split when the benchmark was generated. The sliders below only illustrate
    how assumed costs affect decisions; they do not re-evaluate the model or measure operational ROI.
    """)

    chargeback = st.slider(
        "Chargeback Loss per Missed Fraud (₹):", 1000, 10000, 2500, step=500
    )
    friction = st.slider("Friction Cost per False Positive (₹):", 20, 500, 120, step=10)

    st.metric("Selected validation threshold", f"{detector.optimal_threshold:.4f}")
    st.write(
        f"Illustrative assumed cost ratio: missed fraud ₹{chargeback:,} / false alert ₹{friction:,}."
    )

with tab3:
    st.subheader("Held-Out Synthetic Benchmark (Illustrative)")

    rep_path = Path(
        os.environ.get("FRAUD_REPORT_PATH", PROJECT_ROOT / "benchmark_report.json")
    )
    if rep_path.exists():
        with open(rep_path) as f:
            b_data = json.load(f)

        m_data = b_data.get("test_performance", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC (Precision-Recall)", f"{m_data.get('pr_auc', 1.0):.4f}")
        m2.metric("ROC-AUC", f"{m_data.get('roc_auc', 1.0):.4f}")
        m3.metric(
            "F1 at Validation Threshold",
            f"{m_data.get('f1_at_validation_threshold', 0.0):.4f}",
        )
        m4.metric(
            "Model Scoring P95",
            f"{m_data.get('p95_model_scoring_latency_ms', 0.0):.2f} ms",
        )
        st.warning(b_data.get("warning", "Synthetic demonstration only."))
        st.caption(
            f"Dataset: {b_data.get('dataset', {}).get('source', 'unknown')}. {m_data.get('latency_method', '')}"
        )

        st.markdown("### Top Predictive Risk Signals")
        imps = b_data.get("training", {}).get("feature_importances", {})
        sorted_imps = (
            pd.DataFrame(list(imps.items()), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=False)
            .head(8)
        )
        st.table(sorted_imps)
