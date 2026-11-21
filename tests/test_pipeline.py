"""Unit tests for the fraud feature pipeline."""

import pandas as pd
import pytest
from src.features.pipeline import FraudFeaturePipeline


@pytest.fixture
def sample_transactions():
    return pd.DataFrame([
        {
            "transaction_id": "TXN_001",
            "account_id": "ACC_001",
            "amount": 2500.0,
            "merchant_category": "Electronics",
            "hour_of_day": 14,
            "time_since_last_sec": 300.0,
            "velocity_1h": 2,
            "geo_distance_km": 15.0,
            "device_trust_score": 0.85,
            "is_foreign": 0,
            "amount_deviation_ratio": 1.2,
            "is_fraud": 0,
        },
        {
            "transaction_id": "TXN_002",
            "account_id": "ACC_002",
            "amount": 18500.0,
            "merchant_category": "Gaming",
            "hour_of_day": 3,
            "time_since_last_sec": 45.0,
            "velocity_1h": 6,
            "geo_distance_km": 450.0,
            "device_trust_score": 0.10,
            "is_foreign": 1,
            "amount_deviation_ratio": 8.5,
            "is_fraud": 1,
        },
    ])


def test_feature_pipeline_fit_transform(sample_transactions):
    pipe = FraudFeaturePipeline()
    X, y = pipe.fit_transform(sample_transactions)

    assert len(X) == 2
    assert "sin_hour" in X.columns
    assert "cos_hour" in X.columns
    assert "log_amount" in X.columns
    assert "amount_per_velocity" in X.columns
    assert not X.isnull().values.any()
    assert len(pipe.feature_names) > 10


def test_feature_pipeline_production_transform(sample_transactions):
    pipe = FraudFeaturePipeline()
    pipe.fit_transform(sample_transactions)

    new_tx = sample_transactions.iloc[:1].copy()
    X_new = pipe.transform(new_tx)

    assert len(X_new) == 1
    assert list(X_new.columns) == pipe.feature_names
