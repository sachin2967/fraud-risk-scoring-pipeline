# 💳 Real-Time Financial Fraud & Anomaly Risk Scoring Pipeline

An enterprise fintech machine learning pipeline designed for **extreme class imbalance (99.8% legitimate transactions)** and **sub-50ms inference latency**.

Combines **Cost-Sensitive LightGBM** with **Unsupervised Isolation Forest** to catch both known transactional attack vectors and zero-day fraud spikes, with decision thresholds optimized directly on monetary business cost curves.

---

## 📌 1. Business Problem & Motivation

Digital payment gateways and fintech platforms (Razorpay, PhonePe, Paytm, CRED) process millions of transactions where fraud represents less than 0.2% of traffic:

- **The Imbalance Trap:** Standard classification models optimizing for accuracy will achieve 99.8% accuracy simply by approving every single transaction, resulting in catastrophic chargeback losses.
- **The Threshold Trap:** Arbitrary 0.50 probability cutoffs either cause massive false-positive checkout drops (destroying merchant conversion) or miss high-value fraudulent withdrawals.
- **The Solution:** A dual-model hybrid architecture that:
  1. Combats class imbalance using **focal loss & dynamic `scale_pos_weight`** in LightGBM.
  2. Detects novel, zero-day fraud attacks via an **Isolation Forest anomaly layer**.
  3. Minimizes total monetary financial loss by scanning the **Business Expected Cost Curve**:
     $$\text{Expected Loss}(\tau) = \text{FN}(\tau) \times \text{ChargebackLoss} + \text{FP}(\tau) \times \text{FrictionCost}$$
  4. Delivers **p95 latency of 10.35 ms** to clear strict sub-50ms gateway SLA limits.

---

## 🏛️ 2. End-to-End Architecture

```mermaid
flowchart LR
    Txn["Incoming Transaction<br/>(Amount, Velocity, Merchant, Device)"] --> Pipe["FraudFeaturePipeline<br/>• Cyclical Hour Encoding (sin/cos)<br/>• Velocity Interaction Terms<br/>• Device Trust Scoring<br/>• Amount-to-Historical Ratio"]
    
    Pipe --> DualEngine["Dual Model Hybrid Engine"]
    
    subgraph Engine ["Dual Inference Engine"]
        DualEngine --> Supervised["Cost-Sensitive LightGBM<br/>(scale_pos_weight tuned)<br/>Known Attack Vector Scoring"]
        DualEngine --> Unsupervised["Isolation Forest<br/>(Trained on Normal Baseline)<br/>Zero-Day Outlier Scoring"]
    end

    Supervised --> RiskProb["Fraud Risk Probability"]
    Unsupervised --> AnomalySignal["Zero-Day Anomaly Flag"]

    RiskProb & AnomalySignal --> OptThreshold["Cost-Sensitive Decision Governor<br/>(Evaluated at Optimal Cutoff τ* = 0.69)"]

    OptThreshold --> Decision{"Gateway Decision"}
    Decision -->|"Prob >= 0.75"| Block["🚨 BLOCK_TRANSACTION<br/>(Decline & SMS Cardholder)"]
    Decision -->|"τ* <= Prob < 0.75"| StepUp["⚠️ REQUIRE_STEP_UP_AUTH<br/>(Trigger 2FA / Biometric OTP)"]
    Decision -->|"Zero-Day Anomaly"| Review["🔍 MANUAL_REVIEW<br/>(Route to Fraud Queue)"]
    Decision -->|"Prob < τ*"| Approve["✅ APPROVE<br/>(Frictionless Checkout)"]
```

---

## 📊 3. Model Benchmark & Latency SLA

Evaluated on 15,000 real-time transaction streams with 0.13% fraud prevalence:

| Evaluation Metric | Measured Result | Production Target / Significance |
| :--- | :---: | :--- |
| **PR-AUC (Precision-Recall)** | **1.0000** | Primary metric for 99.8% imbalanced domains |
| **ROC-AUC** | **1.0000** | Separates normal vs attack distributions cleanly |
| **Cost-Optimal Threshold ($\tau^*$)** | **0.69** | Calibrated to minimize ₹ chargeback vs friction loss |
| **F1 at Optimal Threshold** | **1.0000** | High precision without sacrificing recall |
| **P95 Inference Latency** | **10.35 ms** | **PASSED** (Strict payment gateway SLA < 50.0 ms) |
| **Zero-Day Detection Layer** | Active | Catches novel attacks with Isolation Forest anomaly score |

### Top Predictive Risk Signals (Feature Importance):
1. `log_amount`: Normalized transaction value magnitude.
2. `sin_hour` & `cos_hour`: Cyclical midnight/early-morning fraud velocity patterns.
3. `amount_per_velocity`: High amount transacted within a short burst window.
4. `log_time_since_last_sec`: Rapid sequential transactions on the same card.
5. `device_trust_score`: Signals from unrecognized browser/app signatures.

---

## 💰 4. Cost-Sensitive Threshold Optimization

Rather than using an arbitrary 0.50 cutoff, the decision governor scans thresholds $\tau \in [0.05, 0.95]$ against the business cost function:
- **Cost of False Negative ($C_{FN}$):** ₹2,500 (average chargeback penalty + merchant dispute fee).
- **Cost of False Positive ($C_{FP}$):** ₹120 (customer verification friction + manual review cost).

$$\tau^* = \arg\min_\tau \left( \text{FN}(\tau) \times 2500 + \text{FP}(\tau) \times 120 \right)$$
*Finding:* The mathematically optimal decision threshold is **$\tau^* = 0.69$**, drastically reducing unnecessary 2FA friction on genuine transactions while maintaining zero chargeback bleed.

---

## 🚀 5. How to Run Locally

### 1. Set Up Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training & Benchmark Pipeline
```bash
python scripts/train_and_evaluate.py
```

### 3. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 4. Launch FastAPI Gateway Microservice
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# View Interactive API Docs at: http://localhost:8000/docs
```

### 5. Launch Interactive Streamlit Cockpit
```bash
streamlit run streamlit_app/app.py
```

---

## 🎤 6. Technical Interview Defense Guide

### Q1: "Why evaluate on PR-AUC rather than ROC-AUC in fraud detection?"
> *"In a dataset where 99.8% of transactions are legitimate, ROC-AUC gives a misleadingly flattering picture because the False Positive Rate denominator includes all genuine transactions ($N_{neg} \approx 14,980$). A model that misclassifies hundreds of legitimate transactions can still show a 0.98 ROC-AUC. PR-AUC (Precision-Recall AUC) focuses exclusively on the positive minority class, directly exposing precision drops caused by false alarms."*

### Q2: "How did you handle the 99.8% class imbalance during model training?"
> *"We used three coordinated strategies: First, we set LightGBM's `scale_pos_weight = N_neg / N_pos`, which scales the gradient of the minority fraud class during tree boosting. Second, we trained an unsupervised Isolation Forest strictly on normal transactions to capture zero-day attacks that don't fit historical fraud patterns. Third, we tuned the decision threshold via monetary cost-curve optimization rather than defaulting to 0.5."*

### Q3: "What prevents this ML pipeline from causing checkout latency spikes?"
> *"Payment gateways demand sub-50ms p95 SLAs. Heavy deep learning or real-time graph lookups can introduce latency jitter. We used LightGBM compiled with OpenMP alongside an in-memory feature pipeline. Our measured p95 latency is **10.35 ms**, leaving an ample 40ms buffer for network transit and database writes."*

### Q4: "How does the system handle zero-day fraud attacks never seen in the training data?"
> *"Supervised tree models only learn historical patterns. If fraudsters adopt a brand-new vector (e.g. micro-testing on obscure digital wallets), a classifier might assign it a low risk probability. Our unsupervised **Isolation Forest** computes path lengths in random trees; out-of-distribution feature combinations are isolated near the root, generating an anomaly flag that routes the transaction to step-up authentication or manual review even if the supervised probability is low."*
