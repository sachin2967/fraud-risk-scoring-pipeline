"""Tests for the server-side Streamlit-to-FastAPI client."""

import pytest

from streamlit_app import api_client


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._body


def test_score_request_uses_server_side_api_key(monkeypatch):
    captured = {}
    monkeypatch.setenv("FRAUD_API_KEY", "server-side-test-key")

    def fake_request(session, method, url, **kwargs):
        assert not session.trust_env
        captured.update(method=method, url=url, **kwargs)
        return FakeResponse(
            {
                "prediction_id": "abc",
                "transaction_id": "TXN_1",
                "decision": "approve",
                "fraud_risk_score": 0.2,
                "model_latency_ms": 1.0,
            }
        )

    monkeypatch.setattr(api_client.requests.Session, "request", fake_request)
    result = api_client.score_transaction({"transaction_id": "TXN_1"})

    assert result["prediction_id"] == "abc"
    assert captured["method"] == "POST"
    assert captured["url"] == f"{api_client.API_URL}/api/score"
    assert captured["headers"]["X-API-Key"] == "server-side-test-key"
    assert captured["json"] == {"transaction_id": "TXN_1"}
    assert captured["timeout"] == api_client.API_TIMEOUT_SECONDS
    assert not captured["allow_redirects"]


def test_api_error_detail_is_reported(monkeypatch):
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    monkeypatch.setattr(
        api_client.requests.Session,
        "request",
        lambda *args, **kwargs: FakeResponse(
            {"detail": "Valid X-API-Key required."}, status_code=401
        ),
    )
    with pytest.raises(api_client.APIClientError, match="Valid X-API-Key"):
        api_client.get_readiness()


def test_connection_error_is_safe_and_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise api_client.requests.ConnectionError("private transport detail")

    monkeypatch.setattr(api_client.requests.Session, "request", fail)
    with pytest.raises(api_client.APIClientError, match="Cannot reach the scoring API"):
        api_client.get_benchmark()


def test_success_with_invalid_json_is_a_contract_error(monkeypatch):
    class InvalidJSONResponse(FakeResponse):
        def json(self):
            raise ValueError("invalid JSON")

    monkeypatch.setattr(
        api_client.requests.Session,
        "request",
        lambda *args, **kwargs: InvalidJSONResponse(None),
    )
    with pytest.raises(api_client.APIClientError, match="unexpected response"):
        api_client.get_readiness()


def test_success_without_prediction_fields_is_a_contract_error(monkeypatch):
    monkeypatch.setattr(
        api_client.requests.Session,
        "request",
        lambda *args, **kwargs: FakeResponse({"prediction_id": "abc"}),
    )
    with pytest.raises(api_client.APIClientError, match="incomplete prediction"):
        api_client.score_transaction({"transaction_id": "TXN_1"})
