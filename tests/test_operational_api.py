"""Exercise retries, corruption, recovery and fail-closed serving boundaries."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.app as serving
from src.serving.store import DecisionStore


class CountingModel:
    def __init__(self):
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += 1
        return np.array([[0.2, 0.8]])


@pytest.fixture
def model(monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(
        serving,
        "artifact",
        {
            "model": model,
            "threshold": 0.5,
            "dataset_id": 1597,
            "dataset_version": 1,
            "feature_names": serving.FEATURE_NAMES,
        },
    )
    monkeypatch.setattr(serving, "MODEL_VERSION", "a" * 64)
    return model


def payload():
    return {"transaction_id": "retry_1", "amount": 12.0, "pca_features": [0.0] * 28}


def test_concurrent_identical_retries_commit_once(model):
    def submit(_):
        with TestClient(serving.app) as client:
            response = client.post("/api/score", json=payload())
            assert response.status_code == 200
            return response.json()

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(submit, range(12)))
    assert all(result == results[0] for result in results)
    assert model.calls == 1
    assert len(serving.get_store().export()["events"]) == 1


def test_different_payload_same_id_is_conflict(model):
    with TestClient(serving.app) as client:
        assert client.post("/api/score", json=payload()).status_code == 200
        changed = {**payload(), "amount": 14.0}
        assert client.post("/api/score", json=changed).status_code == 409
    assert model.calls == 1


def test_backup_reopen_and_delayed_label(model, tmp_path):
    with TestClient(serving.app) as client:
        response = client.post("/api/score", json=payload()).json()
        route = f"/api/predictions/{response['prediction_id']}/label"
        assert (
            client.post(
                route, json={"label": 1, "source": "historical_benchmark"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                route, json={"label": 1, "source": "historical_benchmark"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                route, json={"label": 0, "source": "historical_benchmark"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                route, json={"label": True, "source": "historical_benchmark"}
            ).status_code
            == 422
        )
    snapshot = tmp_path / "backup.sqlite3"
    serving.get_store().backup(snapshot)
    restored = DecisionStore(snapshot)
    event = restored.export()["events"][0]
    assert event["label"] == 1
    assert event["label_available_at"] >= event["created_at"]
    assert event["features"] == [0.0] * 28 + [12.0]
    assert (
        restored.decide(payload(), lambda: pytest.fail("Must reuse persisted decision"))
        == response
    )


def test_failed_inference_does_not_commit(model, monkeypatch):
    monkeypatch.setattr(
        model, "predict_proba", lambda _: np.array([[0.0, float("nan")]])
    )
    with TestClient(serving.app) as client:
        assert client.post("/api/score", json=payload()).status_code == 503
    assert serving.get_store().export()["events"] == []


def test_storage_failure_returns_unavailable(model, monkeypatch):
    def fail():
        raise sqlite3.OperationalError("private filesystem detail")

    monkeypatch.setattr(serving, "get_store", fail)
    with TestClient(serving.app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/api/score", json=payload())
        assert response.status_code == 503
        assert "private" not in response.text


def test_auth_request_bounds_and_fail_closed_production(model, monkeypatch):
    with TestClient(serving.app) as client:
        monkeypatch.setenv("FRAUD_API_KEY", "x" * 32)
        assert client.get("/api/predictions").status_code == 401
        assert client.get("/docs").status_code == 401
        response = client.post(
            "/api/score",
            json=payload(),
            headers={"X-API-Key": "x" * 32, "X-Request-ID": "load-test_1"},
        )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "load-test_1"
        assert (
            client.post(
                "/api/score", content=b" " * 65537, headers={"X-API-Key": "x" * 32}
            ).status_code
            == 413
        )
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.delenv("FRAUD_MODEL_SHA256", raising=False)
        assert client.get("/ready").status_code == 503
        assert (
            client.post(
                "/api/score", json=payload(), headers={"X-API-Key": "x" * 32}
            ).status_code
            == 503
        )
