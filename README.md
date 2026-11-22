# Real-Data Card Fraud Risk Lab

A reproducible fraud-modeling portfolio project trained on the public ULB / Worldline credit-card fraud dataset through OpenML (dataset 1597, version 1). It includes a checksum-verified OpenML Parquet download, time-ordered holdout, model comparison, validation-only threshold selection, a persisted trained model, FastAPI scoring, and a Streamlit cockpit.

**This is a real-data research benchmark, not a production fraud service.** The dataset contains 284,807 anonymized transactions collected over two days in September 2013, with 492 labeled fraud cases. Its features are anonymized PCA components; it has no account identifiers, no UPI or wallet events, and its `Amount` currency is not documented. The benchmark therefore says nothing about current payment behavior in India, account-level generalization, actual INR losses, or gateway decisions. [OpenML dataset record](https://www.openml.org/d/1597)

## What the project does

- Downloads a pinned OpenML Parquet object and validates its MD5 ETag before training.
- Orders observations by elapsed `Time` and creates sequential 70% train, 15% validation, and 15% final test windows. `Time` is used only for ordering and excluded from the model features.
- Compares class-weighted LightGBM and standardized class-weighted logistic regression. It selects the model using validation average precision; the final test window is not used for model selection.
- Selects the action threshold on validation data using a declared 20:1 false-negative to false-positive *relative* cost weight. This is a sensitivity assumption, not a measured cost model or INR ROI claim.
- Reports average precision against fraud prevalence, ROC-AUC, Brier score, precision/recall/F1, confusion counts, and modeled relative costs on the held-out window.
- Saves a local model artifact used by the FastAPI and Streamlit apps. Raw dataset and model files are excluded from Git.
- Exposes an uncalibrated model score, not a calibrated probability; action labels are demonstration outputs only.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/evaluate_openml_benchmark.py
```

The first training run downloads the public dataset. To use a manually downloaded copy, put the checksum-matching file at `data/raw/openml_1597_creditcard.parquet`. The report is written to `real_data_benchmark_report.json` and the locally ignored model artifact to `models/credit_card_fraud_model.joblib`.

## Reproduced benchmark snapshot

With the pinned OpenML Parquet checksum and seed 42, the latest run selected standardized logistic regression over LightGBM using validation average precision (0.840 vs 0.821). On the untouched later test window, average precision was **0.708** versus **0.00122** fraud prevalence, ROC-AUC was **0.977**, and recall was **0.750** at the validation-selected score threshold. That operating point produced 39 true positives, 13 false negatives, and 46 false positives among 42,721 test transactions. The modeled relative cost was 306 units versus 1,040 for approving everything, under the illustrative 20:1 FN/FP weight. These are historical benchmark results, not expected production outcomes or realized savings.

The 100-call warmed, in-process single-row model latency p95 was **1.43 ms** on the machine used for this run. This does not establish an API or deployment latency SLA.

Start the API and dashboard in separate terminals:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
streamlit run streamlit_app/app.py
```

The API docs are at `/docs`. `POST /api/score` accepts the 28 anonymized values in `pca_features` plus the source `amount` value. The uncalibrated score is not a transaction authorization; action labels are illustrative. Latency shown by the API and dashboard is the in-process model call only and is not an end-to-end SLA.

## Docker demo

```powershell
docker compose up --build
```

The one-shot training service downloads and evaluates the benchmark, then persists the report, model, and raw data in a named volume. The API and dashboard are published on loopback (`8010` and `8510` by default).

## Validation

```powershell
python -m pytest tests/ -v
```

The test suite checks split boundaries, validation-only threshold selection, API input contracts, and the retained synthetic-data utilities. Synthetic generator tests are not evidence about the real-data model. The CI job does not download third-party training data; a local benchmark run is required to produce the real model artifact.

## Limitations and responsible use

- The dataset is historical and narrow. Its 2013 results are not “latest data” and cannot establish current performance.
- PCA features prevent useful transaction-level explanation. No account identity, device history, merchant context, geography, or network graph is available.
- The benchmark has only one short temporal split. It cannot show stability across merchants, countries, years, attack campaigns, or policy changes.
- The relative cost weights are illustrative. They must be replaced with measured costs and operational capacity before selecting any real review threshold.
- Do not use the score or action labels to approve, challenge, or block real customers. A production system requires licensed representative data, privacy and security controls, online feature parity, calibration and drift monitoring, human review, resilience, load testing, and formal risk governance.
- No UPI data or UPI fraud rules are included. PhonePe Pulse publishes aggregate statistics, not per-transaction fraud labels suitable for this supervised task.
