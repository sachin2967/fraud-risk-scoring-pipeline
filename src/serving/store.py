"""Durable SQLite ledger for a bounded, single-host pilot.

An immediate transaction serializes competing submissions. A successful HTTP result
is returned only after commit. SQLite is not a multi-host feature store or HA service.
"""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

SQLITE_TIMEOUT_SECONDS = 5


class ConflictError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DecisionStore:
    def __init__(self, path: Path):
        self.path = path
        # Queue local writers in Python so they do not each incur SQLite busy
        # retries. BEGIN IMMEDIATE still coordinates writes across processes.
        self._write_lock = Lock()

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=SQLITE_TIMEOUT_SECONDS, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("""CREATE TABLE IF NOT EXISTS decisions (
            transaction_id TEXT PRIMARY KEY, prediction_id TEXT UNIQUE NOT NULL,
            request_hash TEXT NOT NULL, request_json TEXT NOT NULL,
            response_json TEXT NOT NULL, created_at TEXT NOT NULL,
            label INTEGER CHECK(label IN (0,1)), label_available_at TEXT,
            label_source TEXT)""")
        return connection

    @contextmanager
    def _writer_connection(self):
        if not self._write_lock.acquire(timeout=SQLITE_TIMEOUT_SECONDS):
            raise sqlite3.OperationalError("Local writer queue timed out.")
        try:
            with closing(self.connect()) as connection:
                yield connection
        finally:
            self._write_lock.release()

    def ready(self):
        with self._writer_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("ROLLBACK")

    def decide(self, request: dict, predict):
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        with self._writer_connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM decisions WHERE transaction_id = ?",
                    (request["transaction_id"],),
                ).fetchone()
                if row:
                    if row["request_hash"] != digest:
                        raise ConflictError(
                            "Transaction ID already belongs to a different request."
                        )
                    response = json.loads(row["response_json"])
                    connection.execute("COMMIT")
                    return response
                # Keep inference inside the transaction for exact retry semantics in this
                # small pilot; publish the resulting concurrency/throughput limitation.
                response = predict()
                created_at = utc_now()
                response.update(prediction_id=uuid4().hex, created_at=created_at)
                connection.execute(
                    """INSERT INTO decisions
                    (transaction_id,prediction_id,request_hash,request_json,response_json,created_at)
                    VALUES (?,?,?,?,?,?)""",
                    (
                        request["transaction_id"],
                        response["prediction_id"],
                        digest,
                        canonical,
                        json.dumps(response, allow_nan=False),
                        created_at,
                    ),
                )
                connection.execute("COMMIT")
                return response
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def label(self, prediction_id: str, label: int, source: str):
        with self._writer_connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM decisions WHERE prediction_id = ?", (prediction_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(prediction_id)
                if row["label"] is not None:
                    if row["label"] != label or row["label_source"] != source:
                        raise ConflictError(
                            "A conflicting label is already recorded; correction requires audit."
                        )
                else:
                    connection.execute(
                        """UPDATE decisions SET label=?, label_available_at=?,
                        label_source=? WHERE prediction_id=?""",
                        (label, utc_now(), source, prediction_id),
                    )
                connection.execute("COMMIT")
                return {"prediction_id": prediction_id, "label": label, "source": source}
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def export(self, after: int = 0, limit: int = 1000):
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT rowid AS cursor, * FROM decisions WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (after, limit),
            ).fetchall()
            events = []
            for row in rows:
                response = json.loads(row["response_json"])
                request = json.loads(row["request_json"])
                events.append(
                    {
                        "cursor": row["cursor"],
                        "prediction_id": row["prediction_id"],
                        "transaction_id": row["transaction_id"],
                        "created_at": row["created_at"],
                        "model_version": response["model_version"],
                        "feature_version": response["feature_version"],
                        "threshold": response["threshold"],
                        "score": response["fraud_risk_score"],
                        "features": request["pca_features"] + [request["amount"]],
                        "label": row["label"],
                        "label_available_at": row["label_available_at"],
                        "label_source": row["label_source"],
                    }
                )
            return {
                "schema_version": "fraud-events-v1",
                "events": events,
                "next_cursor": rows[-1]["cursor"] if rows else after,
                "label_refresh": "Re-export from cursor 0 to include labels attached to earlier decisions.",
            }
        finally:
            connection.close()

    def backup(self, destination: Path):
        """Use SQLite's online backup API; never copy a live database file blindly."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() == self.path.resolve():
            raise ValueError("Backup destination must differ from the source.")
        if destination.exists():
            raise ValueError(
                "Backup destination already exists; choose a new snapshot name."
            )
        connection = self.connect()
        try:
            with sqlite3.connect(destination) as output:
                connection.backup(output)
        finally:
            connection.close()
