"""Bounded training search, calibration and cost sensitivity on historical data.

This is an exploratory extension of an already inspected benchmark, not a new
untouched test set. It never replaces the serving artifact or original report.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_openml_benchmark import (
    chronological_split,
    load_public_dataset,
    select_cost_threshold,
)
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def development_folds(train):
    """Three expanding, nonoverlapping validation windows inside original training."""
    times = train.Time.to_numpy()
    cuts = [
        np.searchsorted(times, np.quantile(times, q), side="right")
        for q in (0.4, 0.6, 0.8)
    ]
    for start, end in zip(cuts, cuts[1:] + [len(train)]):
        before, after = train.iloc[:start], train.iloc[start:end]
        if before.Class.nunique() != 2 or after.Class.nunique() != 2:
            raise ValueError("Development windows require both classes.")
        yield before, after


def estimator(name):
    if name.startswith("lr_"):
        weight = "balanced" if name.endswith("balanced") else None
        c = float(name.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight=weight, max_iter=1000, random_state=42
            ),
        )
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=int(name.split("_")[1]),
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


def logit(scores):
    scores = np.clip(scores, 1e-8, 1 - 1e-8)
    return np.log(scores / (1 - scores)).reshape(-1, 1)


def metrics(y, scores):
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
    }


def main():
    frame = load_public_dataset().apply(pd.to_numeric, errors="raise")
    train, validation, test = chronological_split(frame)
    features = [c for c in frame if c not in {"Time", "Class"}]
    candidates = [
        "lr_0.1_balanced",
        "lr_1_balanced",
        "lr_1_unweighted",
        "lr_10_unweighted",
        "lgbm_7",
        "lgbm_31",
    ]
    comparisons = {}
    for name in candidates:
        values = []
        for earlier, later in development_folds(train):
            model = estimator(name)
            started = time.perf_counter()
            model.fit(earlier[features], earlier.Class)
            score = model.predict_proba(later[features])[:, 1]
            values.append(
                {
                    "train_rows": len(earlier),
                    "validation_rows": len(later),
                    "train_time_max": float(earlier.Time.max()),
                    "validation_time_min": float(later.Time.min()),
                    "fraud_cases": int(later.Class.sum()),
                    **metrics(later.Class, score),
                    "fit_and_score_seconds": time.perf_counter() - started,
                }
            )
        comparisons[name] = {
            "folds": values,
            "mean_ap": float(np.mean([r["average_precision"] for r in values])),
        }
        print(
            f"Completed {name}: development mean AP={comparisons[name]['mean_ap']:.4f}",
            flush=True,
        )
    selected = max(candidates, key=lambda name: comparisons[name]["mean_ap"])
    model = estimator(selected)
    model.fit(train[features], train.Class)
    # Calibration and policy selection use disjoint chronological validation pieces.
    boundary = validation.Time.quantile(0.5)
    calibration = validation[validation.Time <= boundary]
    policy = validation[validation.Time > boundary]
    if calibration.Class.nunique() < 2 or policy.Class.nunique() < 2:
        raise ValueError("Calibration and policy windows require both classes.")
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    calibrator.fit(
        logit(model.predict_proba(calibration[features])[:, 1]), calibration.Class
    )
    raw_policy = model.predict_proba(policy[features])[:, 1]
    raw_test = model.predict_proba(test[features])[:, 1]
    calibrated_policy = calibrator.predict_proba(logit(raw_policy))[:, 1]
    calibrated_test = calibrator.predict_proba(logit(raw_test))[:, 1]
    threshold, val_cost = select_cost_threshold(
        policy.Class.to_numpy(), calibrated_policy
    )
    tn, fp, fn, tp = confusion_matrix(
        test.Class, calibrated_test >= threshold, labels=[0, 1]
    ).ravel()
    sensitivity = []
    for ratio in (5, 10, 20, 50, 100):
        chosen, cost = select_cost_threshold(
            policy.Class.to_numpy(), calibrated_policy, false_negative_cost=float(ratio)
        )
        cm = confusion_matrix(
            test.Class, calibrated_test >= chosen, labels=[0, 1]
        ).ravel()
        sensitivity.append(
            {
                "fn_to_fp_relative_weight": ratio,
                "validation_threshold": chosen,
                "validation_relative_cost": cost,
                "test_false_positive": int(cm[1]),
                "test_false_negative": int(cm[2]),
                "test_relative_cost": int(cm[1] + ratio * cm[2]),
            }
        )
    report = {
        "protocol": "stability-v1; 6 candidates x 3 expanding folds inside original training; selection by mean AP; validation first half sigmoid calibration, second half policy; existing historical test reused once for this report",
        "python": platform.python_version(),
        "seed": 42,
        "development": comparisons,
        "selected": selected,
        "calibration_rows": len(calibration),
        "policy_rows": len(policy),
        "calibration_time_max": float(calibration.Time.max()),
        "policy_time_min": float(policy.Time.min()),
        "test_rows": len(test),
        "test_fraud_cases": int(test.Class.sum()),
        "raw_test": metrics(test.Class, raw_test),
        "calibrated_test": metrics(test.Class, calibrated_test),
        "validation_selected_threshold": threshold,
        "validation_relative_cost": val_cost,
        "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cost_sensitivity": sensitivity,
        "limitations": [
            "Previously inspected two-day 2013 dataset; exploratory reuse, not a fresh external validation.",
            "Small numbers of frauds per window; no account identifiers or longer-period stability.",
            "Calibration is measured on this period only; it does not establish current or deployment calibration.",
            "Costs are illustrative; serving artifact and original benchmark remain unchanged.",
        ],
    }
    (ROOT / "stability_benchmark_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("selected", "raw_test", "calibrated_test", "test_confusion")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
