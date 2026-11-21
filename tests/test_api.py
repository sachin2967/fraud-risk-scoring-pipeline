"""Unit tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["latency_sla_claimed"] is False


def test_benchmark_endpoint():
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "test_performance" in data
    assert "dataset" in data
    assert "warning" in data


def test_score_legitimate_transaction():
    payload = {
        "transaction_id": "TXN_TEST_001",
        "account_id": "ACC_0001",
        "amount": 250.0,
        "merchant_category": "Grocery",
        "hour_of_day": 14,
        "time_since_last_sec": 3600.0,
        "velocity_1h": 1,
        "geo_distance_km": 5.0,
        "device_trust_score": 0.95,
        "is_foreign": 0,
        "amount_deviation_ratio": 0.8,
    }
    response = client.post("/api/score", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "decision" in res
    assert "fraud_probability" in res
    assert "anomaly_flagged" in res
    assert "recommended_action" in res
    assert res["transaction_id"] == "TXN_TEST_001"
    assert res["decision"] in ["APPROVE", "REQUIRE_STEP_UP_AUTH", "BLOCK_TRANSACTION", "MANUAL_REVIEW"]


def test_score_invalid_payload():
    # Missing required field 'amount'
    payload = {
        "transaction_id": "TXN_INVALID",
        "account_id": "ACC_0001",
    }
    response = client.post("/api/score", json=payload)
    assert response.status_code == 422  # Pydantic validation error
