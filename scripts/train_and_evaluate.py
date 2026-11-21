"""Training and benchmark execution script for the Fraud Risk Scoring Pipeline."""

import json
import sys
from pathlib import Path
import joblib
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import FraudFeaturePipeline
from src.models.fraud_detector import HybridFraudDetector


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "transactions.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("Dataset not found. Generating transactions...")
        from data.generate_dataset import generate_fraud_dataset
        generate_fraud_dataset(data_path)

    print("==================================================")
    print("🚀 1. LOADING TRANSACTION STREAM BENCHMARK")
    print("==================================================")
    df = pd.read_csv(data_path)
    total_txns = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_pct = (fraud_count / total_txns) * 100
    print(f"Loaded {total_txns:,} transactions across {df['account_id'].nunique():,} accounts.")
    print(f"Class Imbalance: {fraud_count} Fraudulent ({fraud_pct:.2f}%) vs {total_txns - fraud_count} Legitimate (99.8%)")

    print("\n==================================================")
    print("🛠️ 2. TRANSFORMING BEHAVIORAL & VELOCITY FEATURES")
    print("==================================================")
    pipeline = FraudFeaturePipeline()
    X, y = pipeline.fit_transform(df)
    print(f"Feature matrix shape: {X.shape}, features: {list(X.columns)}")

    joblib.dump(pipeline, models_dir / "feature_pipeline.joblib")

    print("\n==================================================")
    print("🏆 3. TRAINING COST-SENSITIVE DUAL-MODEL ENSEMBLE")
    print("==================================================")
    detector = HybridFraudDetector(random_state=42)
    metrics = detector.train(
        X,
        y,
        chargeback_penalty=2500.0,
        friction_cost=120.0,
    )

    detector.save(str(models_dir / "hybrid_fraud_detector.joblib"))

    print(f"Model ROC-AUC:               {metrics['roc_auc']:.4f}")
    print(f"Model PR-AUC (Priority):     {metrics['pr_auc']:.4f}")
    print(f"Cost-Optimal Threshold (τ*): {metrics['optimal_threshold']:.2f}")
    print(f"F1 at Optimal Threshold:     {metrics['f1_at_optimal_threshold']:.4f}")
    print(f"P95 Inference Latency:       {metrics['p95_inference_latency_ms']:.2f} ms (< 50ms SLA)")

    print("\nTop 5 Risk-Predictive Features:")
    sorted_imps = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imps[:5]:
        print(f"  - {feat}: {imp}")

    benchmark_report = {
        "dataset_summary": {
            "total_transactions": total_txns,
            "fraud_cases": fraud_count,
            "fraud_prevalence_pct": round(fraud_pct, 2),
        },
        "model_performance": metrics,
        "business_sla": {
            "p95_latency_ms": metrics["p95_inference_latency_ms"],
            "sla_target_ms": 50.0,
            "sla_met": metrics["p95_inference_latency_ms"] < 50.0,
        },
    }

    out_file = project_root / "benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\n✅ All artifacts and benchmark report saved to: {out_file}")


if __name__ == "__main__":
    main()
