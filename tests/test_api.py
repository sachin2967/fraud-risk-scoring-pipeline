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
    return {
        "transaction_id": "TXN_DEMO_001",
        "amount": 1499.0,
        "pca_features": [0.0] * 28,
    }


def install_model(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "artifact",
        {
            "model": FixedProbabilityModel(),
            "threshold": 0.5,
            "dataset_id": 1597,
            "dataset_version": 1,
            "feature_names": [f"V{i}" for i in range(1, 29)] + ["Amount"],
        },
    )


def test_health_requires_real_model_artifact(monkeypatch):
    monkeypatch.setattr(api_module, "artifact", None)
    assert client.get("/health").status_code == 503


def test_real_data_score_contract(monkeypatch):
    install_model(monkeypatch)
    response = client.post("/api/score", json=valid_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "REVIEW_OR_STEP_UP"
    assert result["fraud_risk_score"] == 0.99
    assert "uncalibrated" in result["score_interpretation"]
    assert result["amount_value"] == 1499.0
    assert "currency not documented" in result["amount_unit"]
    assert result["model_latency_ms"] >= 0
    assert result["prediction_id"]


def test_dashboard_style_scoring_persists_and_retries_idempotently(monkeypatch):
    install_model(monkeypatch)
    payload = valid_payload()

    first = client.post("/api/score", json=payload)
    retry = client.post("/api/score", json=payload)

    assert first.status_code == retry.status_code == 200
    assert first.json()["prediction_id"] == retry.json()["prediction_id"]
    assert first.json()["created_at"] == retry.json()["created_at"]

    exported = client.get("/api/predictions").json()
    assert len(exported["events"]) == 1
    event = exported["events"][0]
    assert event["transaction_id"] == payload["transaction_id"]
    assert event["prediction_id"] == first.json()["prediction_id"]
    assert event["label"] is None


def test_persisted_prediction_can_receive_delayed_label(monkeypatch):
    install_model(monkeypatch)
    scored = client.post("/api/score", json=valid_payload()).json()
    prediction_id = scored["prediction_id"]

    labelled = client.post(
        f"/api/predictions/{prediction_id}/label",
        json={"label": 1, "source": "verified_outcome"},
    )
    assert labelled.status_code == 200
    assert labelled.json()["label"] == 1

    refreshed = client.get("/api/predictions?after=0").json()
    assert refreshed["events"][0]["label"] == 1
    assert refreshed["events"][0]["label_source"] == "verified_outcome"


def test_reusing_transaction_id_with_different_payload_conflicts(monkeypatch):
    install_model(monkeypatch)
    assert client.post("/api/score", json=valid_payload()).status_code == 200
    changed = valid_payload()
    changed["amount"] = 1500.0
    assert client.post("/api/score", json=changed).status_code == 409


def test_score_rejects_wrong_feature_width_and_unknown_fields():
    payload = valid_payload()
    payload["pca_features"] = [0.0] * 27
    assert client.post("/api/score", json=payload).status_code == 422
    payload = valid_payload()
    payload["upi_id"] = "someone@bank"
    assert client.post("/api/score", json=payload).status_code == 422
