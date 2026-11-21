"""Interactive dashboard for the synthetic fraud-risk scoring demo."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud Risk Scoring Prototype",
    page_icon="💳",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"


@st.cache_resource
def load_artifacts():
    det = joblib.load(str(MODELS_DIR / "hybrid_fraud_detector.joblib"))
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

    col1, col2, col3 = st.columns(3)
    with col1:
        txn_amount = st.number_input(
            "Transaction Amount (₹):", min_value=10.0, max_value=500000.0, value=8500.0
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
            index=4,
        )
        hour = st.slider("Transaction Hour of Day (0-23):", 0, 23, 2)

    with col2:
        velocity = st.number_input(
            "Transactions in Last 1 Hour:", min_value=0, max_value=20, value=5
        )
        time_since = st.number_input(
            "Time Since Last Txn (seconds):",
            min_value=5.0,
            max_value=86400.0,
            value=35.0,
        )
        geo_dist = st.number_input(
            "Geo Distance from Prev Txn (km):",
            min_value=0.0,
            max_value=5000.0,
            value=450.0,
        )

    with col3:
        device_trust = st.slider(
            "Device Trust Score (0=New/Suspicious, 1=Known):", 0.0, 1.0, 0.15
        )
        is_foreign = st.selectbox(
            "Is International Transaction?", options=[0, 1], index=1
        )
        amount_ratio = st.number_input(
            "Amount / Historical Average Ratio:",
            min_value=0.1,
            max_value=50.0,
            value=6.2,
        )

    if st.button("🚀 Score Transaction Risk", type="primary"):
        test_dict = {
            "transaction_id": "TXN_LIVE_SIM",
            "account_id": "ACC_SIM",
            "amount": txn_amount,
            "merchant_category": category,
            "hour_of_day": hour,
            "time_since_last_sec": time_since,
            "velocity_1h": velocity,
            "geo_distance_km": geo_dist,
            "device_trust_score": device_trust,
            "is_foreign": is_foreign,
            "amount_deviation_ratio": amount_ratio,
        }
        test_df = pd.DataFrame([test_dict])
        X_mat = pipeline.transform(test_df).values
        res = detector.predict_risk(X_mat)

        st.divider()

        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            st.metric(
                "Fraud Risk Probability", f"{res['fraud_probability'] * 100:.1f}%"
            )
            st.metric(
                "Cost-Optimal Threshold", f"{res['optimal_threshold'] * 100:.1f}%"
            )

            if res["anomaly_flagged"]:
                st.warning(
                    "⚠️ Unusual feature pattern flagged by Isolation Forest (not validated as zero-day fraud)"
                )
            else:
                st.info("ℹ️ Distribution Pattern Consistent with Training History")

        with res_col2:
            st.markdown("### Operational Gateway Decision")
            dec = res["decision"]
            if dec == "BLOCK_TRANSACTION":
                st.error(f"🚨 **{dec}**")
            elif dec == "REQUIRE_STEP_UP_AUTH":
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

    rep_path = PROJECT_ROOT / "benchmark_report.json"
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
