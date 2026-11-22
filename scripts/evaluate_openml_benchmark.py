"""Train and evaluate the real-data ULB / Worldline card-fraud benchmark."""

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_ID = 1597
DATA_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
DATA_MD5 = "1593844f40edbdabaf5bddec4649e1c4"  # OpenML parquet object's MD5 ETag
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", ROOT / "data"))
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", ROOT / "models" / "credit_card_fraud_model.joblib"))
REPORT_PATH = Path(os.environ.get("FRAUD_REPORT_PATH", ROOT / "real_data_benchmark_report.json"))
# Relative cost units only. The benchmark's Amount field has no documented
# currency, so assigning INR-denominated operational costs would be misleading.
FALSE_NEGATIVE_COST = 20.0
FALSE_POSITIVE_COST = 1.0


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Sort by elapsed transaction time and create fixed 70/15/15 partitions."""
    if "Time" not in frame or "Class" not in frame:
        raise ValueError("Expected OpenML features with Time and target Class.")
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    # Keep all events sharing a second in the same partition. This avoids
    # splitting simultaneous observations across a chronological boundary.
    time_group_ends = np.flatnonzero(
        ordered["Time"].ne(ordered["Time"].shift(-1)).to_numpy()
    ) + 1
    train_end = int(time_group_ends[np.argmin(np.abs(time_group_ends - len(ordered) * 0.70))])
    later_ends = time_group_ends[time_group_ends > train_end]
    validation_end = int(
        later_ends[np.argmin(np.abs(later_ends - len(ordered) * 0.85))]
    )
    parts = (
        ordered.iloc[:train_end],
        ordered.iloc[train_end:validation_end],
        ordered.iloc[validation_end:],
    )
    if any(part.empty or part["Class"].nunique() != 2 for part in parts):
        raise ValueError("Each chronological partition must contain both classes.")
    if not (
        parts[0]["Time"].max() < parts[1]["Time"].min()
        and parts[1]["Time"].max() < parts[2]["Time"].min()
    ):
        raise ValueError("Chronological partition boundaries are invalid.")
    return parts


def select_cost_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Select threshold from validation costs only; return threshold and cost."""
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative_true_positive = np.cumsum(y_true[order] == 1)
    predicted_positive = np.searchsorted(-sorted_scores, -candidates, side="right")
    true_positive = np.zeros(len(candidates), dtype=np.int64)
    has_predictions = predicted_positive > 0
    true_positive[has_predictions] = cumulative_true_positive[
        predicted_positive[has_predictions] - 1
    ]
    positives = int(np.sum(y_true == 1))
    false_negative = positives - true_positive
    false_positive = predicted_positive - true_positive
    costs = false_negative * FALSE_NEGATIVE_COST + false_positive * FALSE_POSITIVE_COST
    best_idx = int(np.argmin(costs))
    return float(candidates[best_idx]), float(costs[best_idx])


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compatibility helper for the original unit tests."""
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    return float(max(candidates, key=lambda threshold: f1_score(
        y_true, scores >= threshold, zero_division=0
    )))


def load_public_dataset() -> pd.DataFrame:
    """Download the pinned OpenML file once and verify its published MD5."""
    path = DATA_DIR / "raw" / "openml_1597_creditcard.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary_path = path.with_suffix(".download")
        metadata = requests.head(DATA_URL, timeout=(10, 30))
        metadata.raise_for_status()
        total_bytes = int(metadata.headers["Content-Length"])
        remote_etag = metadata.headers.get("ETag", "").strip('"')
        if remote_etag != DATA_MD5:
            raise ValueError(f"OpenML Parquet ETag mismatch: {remote_etag}")
        offset = temporary_path.stat().st_size if temporary_path.exists() else 0
        if offset > total_bytes:
            temporary_path.unlink()
            offset = 0
        with temporary_path.open("ab" if offset else "wb") as output:
            # OpenML's object host can stall on larger ranged responses; small
            # verified chunks make downloads resumable on ordinary connections.
            chunk_bytes = 1024 * 1024
            for start in range(offset, total_bytes, chunk_bytes):
                end = min(start + chunk_bytes, total_bytes) - 1
                response = requests.get(
                    DATA_URL,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(10, 120),
                )
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if (
                    response.status_code != 206
                    or len(response.content) != end - start + 1
                    or not content_range.startswith(f"bytes {start}-{end}/")
                    or response.headers.get("ETag", "").strip('"') != DATA_MD5
                ):
                    raise ValueError(f"OpenML returned an invalid byte range at offset {start}.")
                output.write(response.content)
        temporary_path.replace(path)

    digest_builder = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != DATA_MD5:
        path.unlink(missing_ok=True)
        raise ValueError(f"OpenML dataset checksum mismatch: {digest}")
    return pd.read_parquet(path)


def main() -> None:
    frame = load_public_dataset()
    if set(pd.to_numeric(frame["Class"], errors="raise").unique()) != {0, 1}:
        raise ValueError(f"Unexpected target values: {sorted(frame['Class'].unique())}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    train, validation, test = chronological_split(frame)

    # OpenML marks Time as a row identifier; use it only for ordering the split.
    feature_names = [column for column in frame.columns if column not in {"Class", "Time"}]
    lightgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    lightgbm_model.fit(train[feature_names], train["Class"])
    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    logistic_model.fit(train[feature_names], train["Class"])
    validation_scores_by_model = {
        "LightGBM": lightgbm_model.predict_proba(validation[feature_names])[:, 1],
        "LogisticRegression": logistic_model.predict_proba(validation[feature_names])[:, 1],
    }
    validation_ap_by_model = {
        name: float(average_precision_score(validation["Class"], scores))
        for name, scores in validation_scores_by_model.items()
    }
    model_name = max(validation_ap_by_model, key=validation_ap_by_model.get)
    model = lightgbm_model if model_name == "LightGBM" else logistic_model
    validation_scores = validation_scores_by_model[model_name]
    threshold, validation_cost = select_cost_threshold(
        validation["Class"].to_numpy(), validation_scores
    )
    test_scores = model.predict_proba(test[feature_names])[:, 1]
    latency_sample = test[feature_names].iloc[[0]]
    model.predict_proba(latency_sample)  # warm-up outside the timed observations
    latency_ms = []
    for _ in range(100):
        started = time.perf_counter()
        model.predict_proba(latency_sample)
        latency_ms.append((time.perf_counter() - started) * 1000)
    test_pred = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(test["Class"], test_pred, labels=[0, 1]).ravel()
    prevalence = float(test["Class"].mean())

    report = {
        "dataset": {
            "name": "Credit Card Fraud Detection (ULB / Worldline)",
            "source": "OpenML dataset 1597, version 1; originally collected/analyzed by Worldline and ULB MLG",
            "url": "https://www.openml.org/d/1597",
            "openml_data_id": DATA_ID,
            "version": 1,
            "format": "OpenML parquet",
            "download_url": DATA_URL,
            "file_md5": DATA_MD5,
            "observations": len(frame),
            "fraud_cases": int(frame["Class"].sum()),
            "columns": feature_names,
            "known_scope": "anonymized transactions from two days in September 2013; original Amount currency is undocumented; not representative of current payment traffic",
        },
        "split": {
            "method": "stable chronological ordering by Time; first 70% train, next 15% validation, final 15% test",
            "train_rows": len(train),
            "train_fraud_cases": int(train["Class"].sum()),
            "validation_rows": len(validation),
            "validation_fraud_cases": int(validation["Class"].sum()),
            "test_rows": len(test),
            "test_fraud_cases": int(test["Class"].sum()),
            "train_time_max": float(train["Time"].max()),
            "validation_time_min": float(validation["Time"].min()),
            "validation_time_max": float(validation["Time"].max()),
            "test_time_min": float(test["Time"].min()),
            "test_prevalence": prevalence,
        },
        "model": {
            "selected_estimator": model_name,
            "selection_metric": "Validation average precision; test split remains untouched",
            "calibration": "No post-hoc calibration; class-weighted outputs are uncalibrated ranking scores, not calibrated fraud probabilities",
            "validation_average_precision_by_model": validation_ap_by_model,
            "candidates": ["LightGBM (class_weight=balanced)", "StandardScaler + LogisticRegression (class_weight=balanced)"],
            "features": feature_names,
            "time_handling": "Time is used only to order the split and excluded from model features",
            "threshold_selection": "relative-cost-minimizing threshold selected on validation only (20:1 false-negative to false-positive weight); illustrative, not an operating policy",
            "threshold": threshold,
            "assumed_relative_cost_weights": {
                "missed_fraud_fn_relative_units": FALSE_NEGATIVE_COST,
                "false_alert_fp_relative_units": FALSE_POSITIVE_COST,
            },
            "validation_cost_relative_units": validation_cost,
        },
        "test_metrics": {
            "average_precision": float(average_precision_score(test["Class"], test_scores)),
            "prevalence_baseline_average_precision": prevalence,
            "roc_auc": float(roc_auc_score(test["Class"], test_scores)),
            "uncalibrated_score_brier_score": float(brier_score_loss(test["Class"], test_scores)),
            "precision_at_validation_threshold": float(precision_score(test["Class"], test_pred, zero_division=0)),
            "recall_at_validation_threshold": float(recall_score(test["Class"], test_pred, zero_division=0)),
            "f1_at_validation_threshold": float(f1_score(test["Class"], test_pred, zero_division=0)),
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
            "modeled_cost_relative_units": float(fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST),
            "all_approve_cost_relative_units": float(int(test["Class"].sum()) * FALSE_NEGATIVE_COST),
            "single_row_model_latency_p50_ms": float(np.percentile(latency_ms, 50)),
            "single_row_model_latency_p95_ms": float(np.percentile(latency_ms, 95)),
            "latency_scope": "100 warmed in-process single-row predictions; excludes HTTP, serialization, network, and infrastructure",
        },
        "limitations": [
            "The dataset covers only two days in 2013 and has anonymized PCA features; it does not establish generalization to current payment systems.",
            "The dataset contains no account identifier, so account-disjoint evaluation is unavailable.",
            "Threshold selection uses illustrative costs on one validation segment; these are not measured operating costs.",
            "One chronological split does not quantify temporal variability; repeat-period validation is needed for stronger claims.",
            "Relative cost weights are illustrative and not measured operating costs; Amount currency is unspecified, so no INR amount or ROI claim is made.",
        ],
    }
    output = REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "feature_names": feature_names,
         "model_name": model_name,
         "dataset_id": DATA_ID, "dataset_version": 1},
        MODEL_PATH,
        compress=3,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
