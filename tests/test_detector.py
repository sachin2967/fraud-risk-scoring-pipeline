"""Unit tests for the Hybrid Fraud Detector model and latency SLA."""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import pytest
from src.features.pipeline import FraudFeaturePipeline
from src.models.fraud_detector import HybridFraudDetector


@pytest.fixture
def artifacts_dir():
    return Path(__file__).resolve().parent.parent / "models"


def test_model_loading_and_prediction(artifacts_dir):
    det_path = artifacts_dir / "hybrid_fraud_detector.joblib"
    pipe_path = artifacts_dir / "feature_pipeline.joblib"
    assert det_path.exists()
    assert pipe_path.exists()

    detector = HybridFraudDetector.load(str(det_path))
    pipeline = joblib.load(str(pipe_path))

    assert detector.lgbm_model is not None
    assert detector.isolation_forest is not None
    assert 0.0 < detector.optimal_threshold < 1.0

    sample = pd.DataFrame([{
        "transaction_id": "TXN_TEST",
        "account_id": "ACC_TEST",
        "amount": 25000.0,
        "merchant_category": "Luxury_Jewelry",
        "hour_of_day": 2,
        "time_since_last_sec": 30.0,
        "velocity_1h": 5,
        "geo_distance_km": 300.0,
        "device_trust_score": 0.05,
        "is_foreign": 1,
        "amount_deviation_ratio": 9.0,
    }])

    X_mat = pipeline.transform(sample).values

    # Test scoring latency SLA (< 50ms)
    t0 = time.perf_counter()
    res = detector.predict_risk(X_mat)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert latency_ms < 50.0  # Must be fast
    assert "fraud_probability" in res
    assert "decision" in res
    assert res["decision"] in ["BLOCK_TRANSACTION", "REQUIRE_STEP_UP_AUTH", "MANUAL_REVIEW", "APPROVE"]
