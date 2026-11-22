"""FastAPI demo endpoint for synthetic transaction risk scoring."""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from src.data.indian_upi_adapter import IndianUPIPaymentsAdapter
from src.models.fraud_detector import HybridFraudDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", PROJECT_ROOT / "models"))

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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(..., min_length=1, max_length=128, json_schema_extra={"example": "TXN_0001234"})
    account_id: str = Field(..., min_length=1, max_length=128, json_schema_extra={"example": "ACC_00512"})
    amount: FiniteFloat | None = Field(None, ge=0, le=1_000_000_000, json_schema_extra={"example": 8500.0})
    amount_inr: FiniteFloat | None = Field(None, ge=0, le=1_000_000_000, json_schema_extra={"example": 8500.0})
    upi_id: str | None = Field(None, max_length=256, json_schema_extra={"examples": ["sachin@okaxis", "merchant_store@paytm"]})
    txn_type: Literal["UPI_P2P", "UPI_P2M", "WALLET_CASH_OUT", "NET_BANKING_IMPS"] | None = Field(None, json_schema_extra={"examples": ["UPI_P2P", "UPI_P2M", "WALLET_CASH_OUT"]})
    old_balance: FiniteFloat | None = Field(None, ge=0, le=1_000_000_000)
    new_balance: FiniteFloat | None = Field(None, ge=0, le=1_000_000_000)
    upi_velocity_1h: int | None = Field(None, ge=0, le=100_000)
    dest_mule_risk_score: FiniteFloat | None = Field(None, ge=0, le=1)
    merchant_category: str = Field(..., min_length=1, max_length=80, json_schema_extra={"example": "Electronics"})
    hour_of_day: int = Field(..., ge=0, le=23, json_schema_extra={"example": 3})
    time_since_last_sec: FiniteFloat = Field(..., ge=0, le=31_536_000, json_schema_extra={"example": 45.0})
    velocity_1h: int = Field(..., ge=0, le=100_000, json_schema_extra={"example": 4})
    geo_distance_km: FiniteFloat = Field(..., ge=0, le=25_000, json_schema_extra={"example": 120.5})
    device_trust_score: float = Field(
        ..., ge=0.0, le=1.0, json_schema_extra={"example": 0.15}
    )
    is_foreign: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    amount_deviation_ratio: FiniteFloat = Field(..., ge=0, le=1_000_000, json_schema_extra={"example": 4.5})

    @model_validator(mode="after")
    def require_transaction_amount(self):
        if self.amount is None and self.amount_inr is None:
            raise ValueError("amount or amount_inr is required")
        return self


@app.get("/health")
def health():
    ready = detector is not None and pipeline is not None
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run the training pipeline before serving.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "optimal_threshold": detector.optimal_threshold,
        "latency_sla_claimed": False,
    }


@app.post("/api/score")
def score_transaction(data: TransactionInput):
    """Scores one transaction; no end-to-end latency guarantee is made."""
    if detector is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    input_df = pd.DataFrame([data.model_dump()])
    enriched = IndianUPIPaymentsAdapter().transform(input_df)
    X_mat = pipeline.transform(enriched).values

    context = enriched.iloc[0].to_dict() if data.txn_type else None
    result = detector.predict_risk(X_mat, upi_context=context)
    result["transaction_id"] = data.transaction_id
    result["amount_inr"] = float(enriched.iloc[0]["amount_inr"])
    return result


@app.get("/api/benchmark")
def get_benchmark():
    """Returns the frozen benchmark report."""
    rep_file = Path(
        os.environ.get("FRAUD_REPORT_PATH", PROJECT_ROOT / "benchmark_report.json")
    )
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found.")

    with open(rep_file) as f:
        return json.load(f)
