"""FastAPI demo endpoint for synthetic transaction risk scoring."""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.fraud_detector import HybridFraudDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="Fraud Risk Scoring Prototype API",
    description="Educational synthetic-data demo with cost-sensitive LightGBM scoring and an Isolation Forest anomaly flag.",
    version="1.0.0",
)

try:
    detector = HybridFraudDetector.load(
        str(MODELS_DIR / "hybrid_fraud_detector.joblib")
    )
    pipeline = joblib.load(str(MODELS_DIR / "feature_pipeline.joblib"))
    print("✅ Fraud scoring models and feature pipeline loaded.")
except (FileNotFoundError, OSError, EOFError, KeyError) as e:
    print(f"⚠️ Error loading artifacts: {e}")
    detector = None
    pipeline = None


class TransactionInput(BaseModel):
    transaction_id: str = Field(..., json_schema_extra={"example": "TXN_0001234"})
    account_id: str = Field(..., json_schema_extra={"example": "ACC_00512"})
    amount: float = Field(..., json_schema_extra={"example": 8450.0})
    merchant_category: str = Field(..., json_schema_extra={"example": "Electronics"})
    hour_of_day: int = Field(..., ge=0, le=23, json_schema_extra={"example": 3})
    time_since_last_sec: float = Field(..., json_schema_extra={"example": 45.0})
    velocity_1h: int = Field(..., json_schema_extra={"example": 4})
    geo_distance_km: float = Field(..., json_schema_extra={"example": 120.5})
    device_trust_score: float = Field(
        ..., ge=0.0, le=1.0, json_schema_extra={"example": 0.15}
    )
    is_foreign: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    amount_deviation_ratio: float = Field(..., json_schema_extra={"example": 4.5})


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "optimal_threshold": detector.optimal_threshold if detector else None,
        "latency_sla_claimed": False,
    }


@app.post("/api/score")
def score_transaction(data: TransactionInput):
    """Scores one transaction; no end-to-end latency guarantee is made."""
    if detector is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    input_df = pd.DataFrame([data.model_dump()])
    X_mat = pipeline.transform(input_df).values

    result = detector.predict_risk(X_mat)
    result["transaction_id"] = data.transaction_id
    result["amount"] = data.amount
    return result


@app.get("/api/benchmark")
def get_benchmark():
    """Returns the frozen benchmark report."""
    rep_file = PROJECT_ROOT / "benchmark_report.json"
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found.")

    with open(rep_file) as f:
        return json.load(f)
