"""Small ASGI boundary: bounded bodies, optional local auth, safe correlation IDs."""

import hmac
import json
import logging
import os
import re
import time
from uuid import uuid4

LOGGER = logging.getLogger("fraud.http")
MAX_BODY_BYTES = 64 * 1024


def configuration_error() -> str | None:
    if os.getenv("APP_ENV", "local") == "production":
        if len(os.getenv("FRAUD_API_KEY", "")) < 32:
            return "Production requires FRAUD_API_KEY with at least 32 characters."
        if not os.getenv("FRAUD_STATE_PATH"):
            return "Production requires an explicit persistent FRAUD_STATE_PATH."
        if not os.getenv("FRAUD_MODEL_SHA256"):
            return "Production requires a trusted FRAUD_MODEL_SHA256."
    return None


class ServiceBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        headers = dict(scope.get("headers", []))
        supplied_id = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = (
            supplied_id
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied_id)
            else uuid4().hex
        )
        status = 500

        async def response_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii")),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        async def reject(code, detail):
            body = json.dumps({"detail": detail}).encode()
            await response_send(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await response_send({"type": "http.response.body", "body": body})

        try:
            public = scope["path"] in {"/live", "/ready", "/health"}
            if not public:
                error = configuration_error()
                if error:
                    return await reject(503, error)
                expected = os.getenv("FRAUD_API_KEY", "")
                actual = headers.get(b"x-api-key", b"")
                if expected and not hmac.compare_digest(actual, expected.encode()):
                    return await reject(401, "Valid X-API-Key required.")
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > MAX_BODY_BYTES:
                    return await reject(413, "Request body exceeds 64 KiB.")
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            await self.app(scope, bounded_receive, response_send)
        finally:
            # Deliberately omit URL queries, credentials, transaction IDs and payloads.
            LOGGER.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
            )
