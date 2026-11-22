"""Evaluate on the public ULB credit-card dataset without altering the synthetic demo."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from scipy.io import arff
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_ID = 1597
DATA_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
DATA_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Sort by elapsed transaction time and create fixed 70/15/15 partitions."""
    if "Time" not in frame or "Class" not in frame:
        raise ValueError("Expected OpenML features with Time and target Class.")
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * 0.70)
    validation_end = int(len(ordered) * 0.85)
    parts = (
        ordered.iloc[:train_end],
        ordered.iloc[train_end:validation_end],
        ordered.iloc[validation_end:],
    )
    if any(part.empty or part["Class"].nunique() != 2 for part in parts):
        raise ValueError("Each chronological partition must contain both classes.")
    if not (
        parts[0]["Time"].max() <= parts[1]["Time"].min()
        and parts[1]["Time"].max() <= parts[2]["Time"].min()
    ):
        raise ValueError("Chronological partition boundaries are invalid.")
    return parts


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select a threshold on validation data only; do not interpret as policy."""
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    return float(max(candidates, key=lambda threshold: f1_score(
        y_true, scores >= threshold, zero_division=0
    )))


def load_public_dataset() -> pd.DataFrame:
    """Download the pinned OpenML file once and verify its published MD5."""
    path = ROOT / "data" / "raw" / "openml_1597_creditcard.arff"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary_path = path.with_suffix(".download")
        with requests.get(DATA_URL, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        temporary_path.replace(path)

    digest_builder = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != DATA_MD5:
        path.unlink(missing_ok=True)
        raise ValueError(f"OpenML dataset checksum mismatch: {digest}")
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    frame["Class"] = frame["Class"].map(
        lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
    )
    return frame


def main() -> None:
    frame = load_public_dataset()
    if set(frame["Class"].unique()) != {"0", "1"}:
        raise ValueError(f"Unexpected target values: {sorted(frame['Class'].unique())}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    train, validation, test = chronological_split(frame)

    # OpenML marks Time as a row identifier; use it only for ordering the split.
    feature_names = [column for column in frame.columns if column not in {"Class", "Time"}]
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[feature_names], train["Class"])
    validation_scores = model.predict_proba(validation[feature_names])[:, 1]
    threshold = select_f1_threshold(validation["Class"].to_numpy(), validation_scores)
    test_scores = model.predict_proba(test[feature_names])[:, 1]
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
            "observations": len(frame),
            "fraud_cases": int(frame["Class"].sum()),
            "columns": feature_names,
            "known_scope": "anonymized transactions from two days in September 2013; not representative of current payment traffic",
        },
        "split": {
            "method": "stable chronological ordering by Time; first 70% train, next 15% validation, final 15% test",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_time_max": float(train["Time"].max()),
            "validation_time_min": float(validation["Time"].min()),
            "validation_time_max": float(validation["Time"].max()),
            "test_time_min": float(test["Time"].min()),
            "test_prevalence": prevalence,
        },
        "model": {
            "estimator": "LightGBM classifier with class_weight=balanced",
            "features": feature_names,
            "time_handling": "Time is used only to order the split and excluded from model features",
            "threshold_selection": "F1-maximizing threshold selected on validation only; illustrative, not an operating policy",
            "threshold": threshold,
        },
        "test_metrics": {
            "average_precision": float(average_precision_score(test["Class"], test_scores)),
            "prevalence_baseline_average_precision": prevalence,
            "roc_auc": float(roc_auc_score(test["Class"], test_scores)),
            "brier_score": float(brier_score_loss(test["Class"], test_scores)),
            "precision_at_validation_threshold": float(precision_score(test["Class"], test_pred, zero_division=0)),
            "recall_at_validation_threshold": float(recall_score(test["Class"], test_pred, zero_division=0)),
            "f1_at_validation_threshold": float(f1_score(test["Class"], test_pred, zero_division=0)),
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        },
        "limitations": [
            "The dataset covers only two days in 2013 and has anonymized PCA features; it does not establish generalization to current payment systems.",
            "The dataset contains no account identifier, so account-disjoint evaluation is unavailable.",
            "Threshold selection optimizes F1 on one validation segment, not business cost or an operational capacity constraint.",
            "One chronological split does not quantify temporal variability; repeat-period validation is needed for stronger claims.",
            "This benchmark is separate from the synthetic feature pipeline and its API; it does not validate the synthetic model's features or routing policy.",
        ],
    }
    output = ROOT / "real_data_benchmark_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
