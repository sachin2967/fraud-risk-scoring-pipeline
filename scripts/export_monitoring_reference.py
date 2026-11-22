"""Export a bounded, training-only reference paired with the exact serving artifact."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_openml_benchmark import (
    DATA_MD5,
    MODEL_PATH,
    chronological_split,
    load_public_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models" / "fraud_reference.json"
    )
    args = parser.parse_args()
    artifact = joblib.load(args.model)  # Operator-controlled artifact only.
    train, _, _ = chronological_split(load_public_dataset())
    reference = train.sample(n=min(2000, len(train)), random_state=42)[
        artifact["feature_names"]
    ]
    snapshot = {
        "schema_version": "fraud-reference-v1",
        "feature_version": "ulb-pca-v1",
        "model_version": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "feature_names": artifact["feature_names"],
        "threshold": artifact["threshold"],
        "dataset_md5": DATA_MD5,
        "source": "OpenML 1597 v1; training partition only; seed 42",
        "training_time_max": float(train["Time"].max()),
        "records": reference.values.tolist(),
        "scores": artifact["model"].predict_proba(reference)[:, 1].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "model_version": snapshot["model_version"],
                "reference_sha256": hashlib.sha256(
                    args.output.read_bytes()
                ).hexdigest(),
                "reference_rows": len(reference),
            }
        )
    )


if __name__ == "__main__":
    main()
