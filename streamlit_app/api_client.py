"""Server-side HTTP client for the FastAPI scoring service.

The API key stays in the Streamlit server process and is never sent to browser code.
"""

import os

import requests

API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("FRAUD_API_TIMEOUT_SECONDS", "5"))


class APIClientError(RuntimeError):
    """A safe, user-presentable API connectivity or contract error."""


def _request(method: str, path: str, *, payload: dict | None = None) -> dict:
    headers = {}
    api_key = os.getenv("FRAUD_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            json=payload,
            headers=headers,
            timeout=API_TIMEOUT_SECONDS,
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not response.ok:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise APIClientError(
                str(detail or f"Scoring API returned HTTP {response.status_code}.")
            )
        if not isinstance(body, dict):
            raise APIClientError("Scoring API returned an unexpected response.")
        return body
    except requests.RequestException as exc:
        raise APIClientError(
            "Cannot reach the scoring API. Check that FastAPI is running and "
            "FRAUD_API_URL points to the API service."
        ) from exc


def get_readiness() -> dict:
    return _request("GET", "/ready")


def get_benchmark() -> dict:
    return _request("GET", "/api/benchmark")


def score_transaction(payload: dict) -> dict:
    return _request("POST", "/api/score", payload=payload)
