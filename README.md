# Synthetic Fraud Risk Scoring Prototype

A reproducible educational project for imbalanced classification, threshold selection, model serving, and evaluation under temporal and account shift. The bundled dataset is **synthetic and locally generated**. It is not evidence about real payment fraud, production business value, zero-day detection, or a payment-gateway SLA.

The API and dashboard continue to use the synthetic schema and artifacts. A separate evaluation script now benchmarks a model on the public, anonymized ULB / Worldline credit-card dataset. That second benchmark is not connected to the API and does not make the demo production ready.

## What it demonstrates

- A seeded generator with overlapping probabilistic risk signals, repeated transactions per account, and a partially observable late-window attack shift. Labels are sampled rather than assigned by a single deterministic rule. Amount-deviation features use only the account's earlier transactions, with a fixed cold-start prior; the latent account-spend variable is not exposed to the model.
- A LightGBM classifier with `scale_pos_weight`; the Isolation Forest is fitted on legitimate training rows and its fixed anomaly rule is reported as a separate overlay.
- A chronological train / validation / test evaluation with non-overlapping account cohorts. The validation partition alone selects the supervised decision threshold; test results are reported once.
- Account-cluster bootstrap intervals for test ROC-AUC and average precision. These quantify sampling variation among synthetic accounts only; they do not capture generator uncertainty or external validity.
- A FastAPI scoring endpoint and Streamlit demo.
- An Indian UPI and wallet adapter for `UPI_P2P`, `UPI_P2M`, wallet cash-out, and IMPS records. It derives INR, balance-drain, zero-balance, recent UPI velocity, recipient-handle heuristic, and cyclical-hour features.

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

### Separate public-data benchmark

```bash
python scripts/evaluate_openml_benchmark.py
```

This downloads OpenML dataset 1597 (version 1) to ignored `data/raw/`, checks the file against the checksum published in OpenML metadata, sorts by elapsed `Time`, and creates 70%/15%/15% chronological train/validation/test partitions. The validation partition selects an F1 threshold; the test partition is evaluated once. The script writes `real_data_benchmark_report.json`. The source reports 284,807 transactions over two days in September 2013, 492 labeled fraud cases, and anonymized PCA features; this is real historical data but not a representative sample of current payment traffic. See the [OpenML dataset record](https://www.openml.org/d/1597) and retain its source attribution. The source data is not committed or redistributed here.

This dataset has no account identifier, covers only two days, and has no operating-cost or review-capacity labels. Consequently this benchmark does not estimate account generalization, business cost, present-day fraud detection, or deployment readiness. The public-data experiment is intentionally independent of the synthetic scoring API because their feature schemas do not match.

### Current synthetic benchmark

The latest run produced 60,000 events with 1,481 synthetic positive labels (2.47%). The future test window contained 12,061 events from 2,342 accounts and 304 positives. It was account-disjoint from train and validation (zero overlapping account IDs). At the validation-selected threshold, held-out ROC-AUC was **0.672**, average precision **0.0542** against test prevalence **0.0252**, and recall **0.329**. The account-cluster bootstrap 95% intervals were **[0.0432, 0.0737]** for average precision and **[0.6368, 0.7022]** for ROC-AUC.

Under illustrative costs of ₹2,500 per missed positive and ₹120 per false positive, the chosen supervised rule reduced modeled test cost by **7.19%** against the all-approve baseline for this generated sample. The fixed Isolation Forest overlay flagged one event and did not change the supervised decisions or modeled cost. These comparisons are assumption-dependent and must not be presented as realized savings or evidence of zero-day detection.

These numbers are neither a target nor an estimate of real-world performance. They are generator-dependent results from one seeded run. A competitive project claim requires real data with documented rights, a properly designed evaluation protocol, and replication across time periods.

## API

For a local API, first run the synthetic training pipeline. To build and start the complete local Docker demo, including its one-shot training service:

```bash
docker compose up --build
```

For direct local execution, use `python scripts/train_and_evaluate.py` first, then `uvicorn api.app:app --host 127.0.0.1 --port 8000`. Compose publishes only on loopback (API `8010`, dashboard `8510` by default; override with `FRAUD_API_PORT` / `FRAUD_DASHBOARD_PORT`) and stores generated data/models in a named Docker volume. Open `/docs` for the schema. `POST /api/score` accepts either `amount` or `amount_inr`, plus optional UPI transaction context. UPI balance-drain rules can raise the risk score and route to `REQUIRE_UPI_PIN_OR_OTP_STEP_UP` or `BLOCK_AND_FREEZE_UPI`; the Streamlit cockpit includes a ₹150 QR payment and ₹45,000 midnight drain preset. The checked-in synthetic training artifacts retain their original feature schema, so the UPI drain overlay is an explicit policy heuristic rather than a learned signal from UPI-labelled training records. `GET /api/benchmark` serves the synthetic report.

## Limitations and next steps

- The real-data benchmark uses a single two-day 2013 public dataset and does not provide current or representative payment-risk validation.
- Synthetic labels are generated from known risk factors; even with account-disjoint future windows, performance says only how well this model recovers this simulation's process.
- Account bootstrap intervals reflect clustered sampling uncertainty within the held-out synthetic window, not temporal variability, model-selection uncertainty, or data-generation uncertainty.
- Cost assumptions are illustrative. Probability calibration, policy constraints, and threshold selection need representative labels and operational cost estimates.
- UPI recipient-risk values and balance-drain routing are prototype heuristics. The UPI rule score is an operational override, not a calibrated probability. Validate with representative, consented payment data and account for gateway-specific authorization requirements before any real transaction use.
- The anomaly rule is a fixed heuristic and has no evidence of zero-day detection value; compare its reported overlay against the supervised-only policy.
- Latency is 100 in-process single-row scoring calls, excluding HTTP, network, serialization, load, and infrastructure. It is not an end-to-end SLA.
- Docker Compose is a local educational demo only. It has no authentication, request throttling, durable audit trail, secret management, external observability, or realistic concurrent API load test. Do not expose it to a network or use its routing decisions for transactions.
