# Synthetic Fraud Risk Scoring Prototype

A reproducible educational project for imbalanced classification, threshold selection, model serving, and evaluation under temporal and account shift. The bundled dataset is **synthetic and locally generated**. It is not evidence about real payment fraud, production business value, zero-day detection, or a payment-gateway SLA.

## What it demonstrates

- A seeded generator with overlapping probabilistic risk signals, repeated transactions per account, and a partially observable late-window attack shift. Labels are sampled rather than assigned by a single deterministic rule. Amount-deviation features use only the account's earlier transactions, with a fixed cold-start prior; the latent account-spend variable is not exposed to the model.
- A LightGBM classifier with `scale_pos_weight`; the Isolation Forest is fitted on legitimate training rows and its fixed anomaly rule is reported as a separate overlay.
- A chronological train / validation / test evaluation with non-overlapping account cohorts. The validation partition alone selects the supervised decision threshold; test results are reported once.
- Account-cluster bootstrap intervals for test ROC-AUC and average precision. These quantify sampling variation among synthetic accounts only; they do not capture generator uncertainty or external validity.
- A FastAPI scoring endpoint and Streamlit demo.

## Reproduce

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python -c "from pathlib import Path; from data.generate_dataset import generate_fraud_dataset; generate_fraud_dataset(Path('data/transactions.csv'))"
python scripts/train_and_evaluate.py
pytest -q
```

The explicit generator command replaces any previous local dataset; training only generates data automatically when the CSV is absent. `benchmark_report.json` records the dataset recipe, time windows, class counts, account overlap checks, validation-selected threshold, test results, and clustered intervals. The checked-in report is generated from seed 42 and 60,000 synthetic transactions.

### Current synthetic benchmark

The latest run produced 60,000 events with 1,481 synthetic positive labels (2.47%). The future test window contained 12,061 events from 2,342 accounts and 304 positives. It was account-disjoint from train and validation (zero overlapping account IDs). At the validation-selected threshold, held-out ROC-AUC was **0.672**, average precision **0.0542** against test prevalence **0.0252**, and recall **0.329**. The account-cluster bootstrap 95% intervals were **[0.0432, 0.0737]** for average precision and **[0.6368, 0.7022]** for ROC-AUC.

Under illustrative costs of ₹2,500 per missed positive and ₹120 per false positive, the chosen supervised rule reduced modeled test cost by **7.19%** against the all-approve baseline for this generated sample. The fixed Isolation Forest overlay flagged one event and did not change the supervised decisions or modeled cost. These comparisons are assumption-dependent and must not be presented as realized savings or evidence of zero-day detection.

These numbers are neither a target nor an estimate of real-world performance. They are generator-dependent results from one seeded run. A competitive project claim requires real data with documented rights, a properly designed evaluation protocol, and replication across time periods.

## API

After generating model artifacts, start the service:

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the schema. `POST /api/score` returns a supervised probability, configured threshold, anomaly indicator, and prototype routing decision. `GET /api/benchmark` serves the local benchmark report. The `BLOCK_TRANSACTION` and `REQUIRE_STEP_UP_AUTH` cutoffs are demo policy constants, not calibrated business rules.

## Limitations and next steps

- This project has no legitimate external transaction dataset. Obtain an openly licensed dataset with documented provenance before making claims about real payment risk.
- Synthetic labels are generated from known risk factors; even with account-disjoint future windows, performance says only how well this model recovers this simulation's process.
- Account bootstrap intervals reflect clustered sampling uncertainty within the held-out synthetic window, not temporal variability, model-selection uncertainty, or data-generation uncertainty.
- Cost assumptions are illustrative. Probability calibration, policy constraints, and threshold selection need representative labels and operational cost estimates.
- The anomaly rule is a fixed heuristic and has no evidence of zero-day detection value; compare its reported overlay against the supervised-only policy.
- Latency is 100 in-process single-row scoring calls, excluding HTTP, network, serialization, load, and infrastructure. It is not an end-to-end SLA.
- Docker image builds and realistic concurrent API load remain unverified in this environment.
