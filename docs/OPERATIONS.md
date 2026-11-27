# Fraud pilot operations

Use Python 3.11 and `pip install -r requirements.txt`; core model/runtime versions are
constrained. Only load operator-trusted joblib files. Train/evaluate with
`python scripts/evaluate_openml_benchmark.py` if no trusted artifact is available.
Run `python scripts/export_monitoring_reference.py` to create a 2,000-row training-only
reference and record the printed model/reference SHA-256 values.

## Serving

Set `APP_ENV=production`, a random `FRAUD_API_KEY` of 32+ characters,
`FRAUD_STATE_PATH` to an absolute persistent SQLite path, and `FRAUD_MODEL_SHA256`
to the trusted model digest. Start `uvicorn api.app:app --host 127.0.0.1 --port 8010
--workers 1`. Keep it behind authenticated TLS access; never expose the operator key
in browser code. `/live` checks the process; `/ready` checks configuration/model/storage.

Send `X-API-Key` on non-health routes. `POST /api/score` takes a stable transaction ID,
28 ordered PCA values and amount. Identical retries reuse the original decision;
different payloads sharing an ID conflict. Successful results are committed first.
The model score stays in [0,1]; a decision cutoff of 2.0 represents the all-approve
baseline. Validation cost selection includes that option even for scores exactly 1.
Failed inference/storage returns 503; callers may retry the same identity/payload.

## Dashboard integration

The Streamlit process is an API client, not a second model-serving process. Locally,
set `FRAUD_API_URL=http://127.0.0.1:8000` before starting Streamlit. If API
authentication is enabled, set the same `FRAUD_API_KEY` in the Streamlit server
environment. The key is added to server-to-server HTTP headers and is never embedded
in browser code, a query string or a client-side request. Set
`FRAUD_API_TIMEOUT_SECONDS` to adjust the bounded request timeout (default 5 seconds).

Docker Compose configures the dashboard to use `http://fraud-api:8000` on the
private Compose network and passes the optional API key to both services. Keep
both published ports loopback-only. A dashboard score now uses the same API input
validation, authentication, idempotency and durable SQLite ledger as any other
caller. Its displayed prediction ID can be used with the delayed-label endpoint.
If the API is unavailable or not ready, the dashboard reports that condition
instead of falling back to local model inference.

## Labels, exports and recovery

Attach labels through `POST /api/predictions/{prediction_id}/label` with integer
`label` and a declared `source`. Server receipt time is the availability time.
Conflicting labels require an audited correction, not silent overwrite.
`GET /api/predictions?after=0&limit=1000` exports events; paginate using `next_cursor`.
Re-export earlier rows for later labels: this cursor is not a label change feed.

`python scripts/backup_decisions.py --source var/decisions.sqlite3 --output var/snapshot-001.sqlite3`
uses the SQLite online backup API and refuses overwrite. Protect backups like the
ledger. Restore while stopped into a new filename, switch `FRAUD_STATE_PATH`, check
readiness and replay a known ID to verify the same decision and label.

Retain previous image/model/report/reference pairs and hashes for rollback. Restart
with the complete previous pair; do not mix model cohorts/references or replace a
live model file expecting workers to reload it. SQLite inference/writes are serialized
for a bounded single-host pilot; this is not HA or horizontal scaling. Keep state on
a persistent VM disk, never an ephemeral Cloud Run container filesystem.

The 2013 dataset does not validate modern card/UPI behavior. Scores, demo actions and
illustrative relative costs must not authorize real payments. Representative licensed
data, genuine delayed labels, external load/security review and operational governance
remain required for actual use.
