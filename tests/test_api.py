"""Contract tests for the real-data scoring API."""

import numpy as np
from fastapi.testclient import TestClient

import api.app as api_module

client = TestClient(api_module.app)


class FixedProbabilityModel:
    def predict_proba(self, features):
        assert features.shape == (1, 29)
        return np.array([[0.01, 0.99]])


def valid_payload():
    return {"transaction_id": "TXN_DEMO_001", "amount": 1499.0, "pca_features": [0.0] * 28}


def test_health_requires_real_model_artifact(monkeypatch):
    monkeypatch.setattr(api_module, "artifact", None)
    assert client.get("/health").status_code == 503


def test_real_data_score_contract(monkeypatch):
    monkeypatch.setattr(api_module, "artifact", {
        "model": FixedProbabilityModel(), "threshold": 0.5,
        "dataset_id": 1597, "dataset_version": 1,
        "feature_names": [f"V{i}" for i in range(1, 29)] + ["Amount"],
    })
    response = client.post("/api/score", json=valid_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "REVIEW_OR_STEP_UP"
    assert result["fraud_risk_score"] == 0.99
    assert "uncalibrated" in result["score_interpretation"]
    assert result["amount_value"] == 1499.0
    assert "currency not documented" in result["amount_unit"]
    assert result["model_latency_ms"] >= 0


def test_score_rejects_wrong_feature_width_and_unknown_fields():
    payload = valid_payload()
    payload["pca_features"] = [0.0] * 27
    assert client.post("/api/score", json=payload).status_code == 422
    payload = valid_payload()
    payload["upi_id"] = "someone@bank"
    assert client.post("/api/score", json=payload).status_code == 422
