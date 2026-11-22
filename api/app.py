"""API serving the locally trained public card-fraud benchmark model."""

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", ROOT / "models" / "credit_card_fraud_model.joblib"))
REPORT_PATH = Path(os.environ.get("FRAUD_REPORT_PATH", ROOT / "real_data_benchmark_report.json"))

app = FastAPI(
    title="Real-Data Card Fraud Risk Scoring API",
    description="Research demo trained on the public ULB / Worldline anonymized card-fraud dataset (2013).",
    version="2.0.0",
)

try:
    artifact = joblib.load(MODEL_PATH)
    warmup = pd.DataFrame(
        [[0.0] * len(artifact["feature_names"])], columns=artifact["feature_names"]
    )
    artifact["model"].predict_proba(warmup)
except (FileNotFoundError, OSError, EOFError, KeyError):
    artifact = None


class TransactionInput(BaseModel):
    """Provide V1 through V28 in the dataset's anonymized PCA feature order."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(..., min_length=1, max_length=128, examples=["TXN_DEMO_001"])
    amount: FiniteFloat = Field(..., ge=0, le=1_000_000, examples=[1499.0], description="Dataset Amount value; source currency is undocumented.")
    pca_features: list[FiniteFloat] = Field(..., min_length=28, max_length=28)


@app.get("/health")
def health():
    if artifact is None:
        raise HTTPException(status_code=503, detail="Real-data model is not trained. Run python scripts/evaluate_openml_benchmark.py.")
    return {"status": "healthy", "model_loaded": True, "dataset_id": artifact["dataset_id"], "dataset_version": artifact["dataset_version"]}


@app.post("/api/score")
def score_transaction(data: TransactionInput):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Real-data model artifact unavailable. Train the benchmark first.")
    features = pd.DataFrame(
        [data.pca_features + [data.amount]], columns=artifact["feature_names"], dtype=float
    )
    started = time.perf_counter()
    probability = float(artifact["model"].predict_proba(features)[0, 1])
    latency_ms = (time.perf_counter() - started) * 1000
    threshold = float(artifact["threshold"])
    decision = "REVIEW_OR_STEP_UP" if probability >= threshold else "APPROVE_DEMO_ONLY"
    return {
        "transaction_id": data.transaction_id,
        "fraud_risk_score": round(probability, 6),
        "score_interpretation": "uncalibrated model score; not a calibrated fraud probability",
        "threshold": threshold,
        "decision": decision,
        "amount_value": data.amount,
        "amount_unit": "source dataset unit (currency not documented)",
        "model_latency_ms": round(latency_ms, 3),
        "latency_scope": "in-process model call only; excludes HTTP and infrastructure",
    }


@app.get("/api/benchmark")
def get_benchmark():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark report unavailable. Train the real-data model first.")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
