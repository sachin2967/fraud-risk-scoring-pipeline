"""Train and evaluate on a future, account-disjoint synthetic holdout."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FraudFeaturePipeline  # noqa: E402
from src.models.fraud_detector import HybridFraudDetector  # noqa: E402


def temporal_account_split(df):
    """Return fixed future windows and fail loudly on class or account leakage."""
    train_mask = df["timestamp"] < pd.Timestamp("2026-02-06")
    validation_mask = df["timestamp"].between(pd.Timestamp("2026-02-06"), pd.Timestamp("2026-02-17 23:59:59"))
    test_mask = df["timestamp"] >= pd.Timestamp("2026-02-18")
    splits = {"train": df.loc[train_mask], "validation": df.loc[validation_mask], "test": df.loc[test_mask]}
    if any(part.empty or part["is_fraud"].nunique() != 2 for part in splits.values()):
        raise ValueError("Each temporal window must contain both classes; regenerate with the default seed.")
    names = list(splits)
    account_sets = {name: set(part["account_id"]) for name, part in splits.items()}
    overlap = {f"{a}_{b}": len(account_sets[a] & account_sets[b]) for i, a in enumerate(names) for b in names[i+1:]}
    if any(overlap.values()):
        raise ValueError(f"Account overlap found across chronological partitions: {overlap}")
    return splits, overlap


def _account_bootstrap_intervals(y, scores, accounts, iterations=300, seed=1701):
    """Percentile intervals from resampling test accounts as clusters."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(accounts))).groupby(accounts, sort=False).apply(list)
    group_rows = groups.to_list()
    aps, rocs = [], []
    for _ in range(iterations):
        chosen = rng.integers(0, len(group_rows), size=len(group_rows))
        idx = np.fromiter((i for group_idx in chosen for i in group_rows[group_idx]), dtype=int)
        boot_y = y.iloc[idx].to_numpy()
        if boot_y.min() == boot_y.max():
            continue
        boot_scores = scores[idx]
        aps.append(average_precision_score(boot_y, boot_scores))
        rocs.append(roc_auc_score(boot_y, boot_scores))
    def interval(values):
        return [round(float(x), 4) for x in np.percentile(values, [2.5, 97.5])]
    return {"method": "percentile bootstrap resampling test accounts as clusters", "iterations": len(aps),
            "average_precision_95pct_ci": interval(aps), "roc_auc_95pct_ci": interval(rocs)}


def main():
    data_path = PROJECT_ROOT / "data" / "transactions.csv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        from data.generate_dataset import generate_fraud_dataset
        generate_fraud_dataset(data_path)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    if len(df) < 3_000 or df["is_fraud"].nunique() != 2:
        raise ValueError("Benchmark requires at least 3,000 records and both label classes.")
    # Fixed chronological windows avoid random future leakage. Generator cohorts are
    # assigned to one window only; assertions make this property auditable.
    splits, overlap = temporal_account_split(df)

    pipeline = FraudFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(splits["train"])
    X_validation = pipeline.transform(splits["validation"])
    y_validation = splits["validation"]["is_fraud"]
    X_test = pipeline.transform(splits["test"])
    y_test = splits["test"]["is_fraud"]
    joblib.dump(pipeline, models_dir / "feature_pipeline.joblib")

    detector = HybridFraudDetector(random_state=42)
    validation_result = detector.train(X_train, y_train, X_validation, y_validation,
                                       chargeback_penalty=2500.0, friction_cost=120.0)
    test_result = detector.evaluate(X_test, y_test)
    baseline_cost = test_result["all_approve_baseline_business_cost"]
    test_result["cost_savings_vs_all_approve_pct"] = round(
        100.0 * (baseline_cost - test_result["business_cost"]) / baseline_cost, 2
    ) if baseline_cost else 0.0
    test_result["anomaly_overlay"]["cost_savings_vs_supervised_pct"] = round(
        100.0 * (test_result["business_cost"] - test_result["anomaly_overlay"]["business_cost"])
        / test_result["business_cost"], 2
    ) if test_result["business_cost"] else 0.0
    test_scores = detector.lgbm_model.predict_proba(X_test.values)[:, 1]
    test_result["average_precision_vs_prevalence"] = round(
        test_result["pr_auc"] / max(float(y_test.mean()), 1e-12), 3)
    test_result["account_clustered_confidence_intervals"] = _account_bootstrap_intervals(
        y_test.reset_index(drop=True), test_scores, splits["test"]["account_id"].reset_index(drop=True))
    detector.save(str(models_dir / "hybrid_fraud_detector.joblib"))

    report = {
        "dataset": {
            "source": "locally generated synthetic transactions; no real or third-party transaction data",
            "generator": "seeded overlapping probabilistic risk processes with a partially observable late-window attack shift",
            "total_transactions": len(df), "unique_accounts": int(df["account_id"].nunique()),
            "fraud_cases": int(df["is_fraud"].sum()),
            "fraud_prevalence_pct": round(float(df["is_fraud"].mean() * 100), 4),
            "split_method": "chronological train (before 2026-02-06), validation (2026-02-06 through 2026-02-17), test (from 2026-02-18); account-disjoint by generator cohort",
            "account_overlap": overlap,
            "class_counts": {name: {"transactions": len(part), "accounts": int(part["account_id"].nunique()),
                                     "legitimate": int((part["is_fraud"] == 0).sum()),
                                     "fraud": int(part["is_fraud"].sum()),
                                     "prevalence_pct": round(float(part["is_fraud"].mean() * 100), 4)}
                             for name, part in splits.items()},
            "window_bounds": {name: {"min": str(part["timestamp"].min()), "max": str(part["timestamp"].max())}
                              for name, part in splits.items()},
        },
        "training": {"threshold_selection": "validation only; threshold minimizing the stated illustrative costs",
                     "validation_min_business_cost": validation_result["min_business_loss"],
                     "feature_importances": validation_result["feature_importances"]},
        "test_performance": test_result,
        "cost_assumptions_inr": {"false_negative": 2500.0, "false_positive": 120.0},
        "latency_scope": test_result["latency_method"],
        "warning": "All evidence is synthetic and generator-dependent; confidence intervals quantify account sampling variation only, not real-world or generator uncertainty. No production utility, zero-day detection, or payment SLA is established.",
    }
    (PROJECT_ROOT / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
