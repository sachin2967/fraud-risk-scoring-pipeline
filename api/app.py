"""API serving the locally trained public card-fraud benchmark model."""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from src.serving.http import ServiceBoundary, configuration_error
from src.serving.store import ConflictError, DecisionStore

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get(
        "FRAUD_MODEL_PATH", ROOT / "models" / "credit_card_fraud_model.joblib"
    )
)
REPORT_PATH = Path(
    os.environ.get("FRAUD_REPORT_PATH", ROOT / "real_data_benchmark_report.json")
)

app = FastAPI(
    title="Real-Data Card Fraud Risk Scoring API",
    description="Research demo trained on the public ULB / Worldline anonymized card-fraud dataset (2013).",
    version="2.0.0",
)
app.add_middleware(ServiceBoundary)
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
FEATURE_VERSION = "ulb-pca-v1"
MODEL_VERSION = "unavailable"

try:
    MODEL_VERSION = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    expected_hash = os.getenv("FRAUD_MODEL_SHA256")
    if expected_hash and MODEL_VERSION != expected_hash:
        raise ValueError("Model checksum mismatch")
    artifact = joblib.load(MODEL_PATH)
    if (
        artifact["feature_names"] != FEATURE_NAMES
        or not 0 <= float(artifact["threshold"]) <= 2
    ):
        raise ValueError("Invalid model feature order or threshold")
    warmup = pd.DataFrame(
        [[0.0] * len(artifact["feature_names"])], columns=artifact["feature_names"]
    )
    artifact["model"].predict_proba(warmup)
except (
    OSError,
    EOFError,
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    ImportError,
    RuntimeError,
    pickle.UnpicklingError,
) as exc:
    # Only load operator-controlled joblib artifacts. The hash verifies identity,
    # not trustworthiness of arbitrary pickle input. Never accept model uploads.
    logging.getLogger(__name__).warning("Model unavailable (%s)", type(exc).__name__)
    artifact = None


@lru_cache(maxsize=8)
def store_for_path(path: Path) -> DecisionStore:
    return DecisionStore(path)


def get_store() -> DecisionStore:
    return store_for_path(
        Path(os.getenv("FRAUD_STATE_PATH", ROOT / "var" / "decisions.sqlite3"))
    )


class TransactionInput(BaseModel):
    """Provide V1 through V28 in the dataset's anonymized PCA feature order."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(
        ..., pattern=r"^[A-Za-z0-9_-]{1,128}$", examples=["TXN_DEMO_001"]
    )
    amount: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000,
        examples=[1499.0],
        description="Dataset Amount value; source currency is undocumented.",
    )
    pca_features: list[FiniteFloat] = Field(..., min_length=28, max_length=28)


class LabelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: int = Field(ge=0, le=1)
    source: str = Field(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")


@app.get("/live")
def liveness():
    return {"status": "alive"}


@app.get("/health")
@app.get("/ready")
def health():
    error = configuration_error()
    if error:
        raise HTTPException(status_code=503, detail=error)
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Real-data model is not trained. Run python scripts/evaluate_openml_benchmark.py.",
        )
    try:
        get_store().ready()
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Decision storage unavailable.") from exc
    return {
        "status": "healthy",
        "model_loaded": True,
        "storage_ready": True,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset_id": artifact["dataset_id"],
        "dataset_version": artifact["dataset_version"],
    }


@app.post("/api/score")
def score_transaction(data: TransactionInput):
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Real-data model artifact unavailable. Train the benchmark first.",
        )

    def predict():
        return predict_transaction(data)

    try:
        return get_store().decide(data.model_dump(), predict)
    except ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(
            503, "Decision storage unavailable; no decision acknowledged."
        ) from exc


def predict_transaction(data: TransactionInput):
    features = pd.DataFrame(
        [data.pca_features + [data.amount]],
        columns=artifact["feature_names"],
        dtype=float,
    )
    started = time.perf_counter()
    try:
        probability = float(artifact["model"].predict_proba(features)[0, 1])
        if not 0 <= probability <= 1:
            raise ValueError("Non-finite or out-of-range model output")
    except Exception as exc:
        raise HTTPException(
            503, "Model inference failed; no decision recorded."
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    threshold = float(artifact["threshold"])
    decision = "REVIEW_OR_STEP_UP" if probability >= threshold else "APPROVE_DEMO_ONLY"
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "transaction_id": data.transaction_id,
        "fraud_risk_score": probability,
        "score_interpretation": "uncalibrated model score; not a calibrated fraud probability",
        "threshold": threshold,
        "decision": decision,
        "amount_value": data.amount,
        "amount_unit": "source dataset unit (currency not documented)",
        "model_latency_ms": round(latency_ms, 3),
        "latency_scope": "in-process model call only; excludes HTTP and infrastructure",
    }


@app.post("/api/predictions/{prediction_id}/label")
def record_label(prediction_id: str, data: LabelInput):
    try:
        return get_store().label(prediction_id, data.label, data.source)
    except KeyError as exc:
        raise HTTPException(404, "Prediction not found.") from exc
    except ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Decision storage unavailable.") from exc


@app.get("/api/predictions")
def export_predictions(
    after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=5000)
):
    try:
        return get_store().export(after=after, limit=limit)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(503, "Decision storage unavailable.") from exc


@app.get("/api/benchmark")
def get_benchmark():
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark report unavailable. Train the real-data model first.",
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
