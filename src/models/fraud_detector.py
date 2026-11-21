"""Hybrid Fraud Risk Detection Model combining Cost-Sensitive LightGBM with Isolation Forest."""

import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


class HybridFraudDetector:
    """Production hybrid risk model: Supervised LightGBM + Unsupervised Isolation Forest."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.lgbm_model: Optional[lgb.LGBMClassifier] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.optimal_threshold: float = 0.50
        self.cost_matrix: Dict[str, float] = {
            "chargeback_loss_fn": 2500.0,  # ₹2500 lost per false negative (undetected fraud)
            "friction_cost_fp": 120.0,     # ₹120 lost per false positive (manual review / customer friction)
        }
        self.feature_names: List[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        chargeback_penalty: float = 2500.0,
        friction_cost: float = 120.0,
    ) -> Dict[str, Any]:
        """Trains dual-model ensemble with cost-sensitive threshold optimization."""
        self.feature_names = list(X.columns)
        self.cost_matrix["chargeback_loss_fn"] = chargeback_penalty
        self.cost_matrix["friction_cost_fp"] = friction_cost

        # Calculate class weight ratio for extreme imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        # 1. Train Supervised LightGBM with scale_pos_weight
        lgb_params = {
            "n_estimators": 120,
            "learning_rate": 0.04,
            "num_leaves": 24,
            "max_depth": 5,
            "scale_pos_weight": scale_pos_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "verbose": -1,
        }
        self.lgbm_model = lgb.LGBMClassifier(**lgb_params)
        self.lgbm_model.fit(X.values, y.values)

        # 2. Train Unsupervised Isolation Forest (trained only on normal transactions to capture zero-day attacks)
        X_normal = X[y == 0].values
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_normal)

        # 3. Optimize Decision Threshold on Financial Cost Curve
        probs = self.lgbm_model.predict_proba(X.values)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 91)
        best_cost = float("inf")
        best_t = 0.50

        cost_curve = []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            fn = int(np.sum((preds == 0) & (y.values == 1)))
            fp = int(np.sum((preds == 1) & (y.values == 0)))
            total_loss = (fn * chargeback_penalty) + (fp * friction_cost)
            cost_curve.append({"threshold": round(float(t), 2), "total_cost": total_loss, "fn": fn, "fp": fp})

            if total_loss < best_cost:
                best_cost = total_loss
                best_t = float(t)

        self.optimal_threshold = round(best_t, 2)

        # Evaluate performance metrics at optimal threshold
        opt_preds = (probs >= self.optimal_threshold).astype(int)
        roc_auc = float(roc_auc_score(y, probs))
        pr_auc = float(average_precision_score(y, probs))
        f1_opt = float(f1_score(y, opt_preds, zero_division=0))

        # Measure single-transaction inference latency
        sample_x = X.values[:1]
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = self.predict_risk(sample_x)
            latencies.append((time.perf_counter() - t0) * 1000)

        p95_latency_ms = float(np.percentile(latencies, 95))

        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "optimal_threshold": self.optimal_threshold,
            "f1_at_optimal_threshold": round(f1_opt, 4),
            "min_business_loss": best_cost,
            "p95_inference_latency_ms": round(p95_latency_ms, 3),
            "feature_importances": dict(zip(self.feature_names, self.lgbm_model.feature_importances_.tolist())),
        }

    def predict_risk(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """Scores transaction with dual supervised probability and isolation anomaly signal."""
        # Supervised probability
        lgbm_prob = float(self.lgbm_model.predict_proba(X_sample)[0, 1])

        # Isolation anomaly score (negative values are outliers)
        iso_score = float(self.isolation_forest.score_samples(X_sample)[0])
        is_zero_day_anomaly = bool(iso_score < -0.65)

        # Decision
        is_fraud_predicted = bool(lgbm_prob >= self.optimal_threshold or is_zero_day_anomaly)

        if lgbm_prob >= 0.75:
            decision = "BLOCK_TRANSACTION"
            action = "Immediate decline; notify cardholder via SMS"
        elif lgbm_prob >= self.optimal_threshold:
            decision = "REQUIRE_STEP_UP_AUTH"
            action = "Trigger 2-Factor Biometric or OTP challenge"
        elif is_zero_day_anomaly:
            decision = "MANUAL_REVIEW"
            action = "Approve provisionally, route to fraud analyst queue"
        else:
            decision = "APPROVE"
            action = "Frictionless checkout"

        return {
            "fraud_probability": round(lgbm_prob, 4),
            "optimal_threshold": self.optimal_threshold,
            "is_fraud": is_fraud_predicted,
            "zero_day_anomaly": is_zero_day_anomaly,
            "decision": decision,
            "recommended_action": action,
        }

    def save(self, filepath: str):
        """Persists trained hybrid model artifacts."""
        joblib.dump({
            "lgbm": self.lgbm_model,
            "iso": self.isolation_forest,
            "threshold": self.optimal_threshold,
            "cost_matrix": self.cost_matrix,
            "features": self.feature_names,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HybridFraudDetector":
        """Loads persisted model."""
        data = joblib.load(filepath)
        inst = cls()
        inst.lgbm_model = data["lgbm"]
        inst.isolation_forest = data["iso"]
        inst.optimal_threshold = data["threshold"]
        inst.cost_matrix = data["cost_matrix"]
        inst.feature_names = data["features"]
        return inst
