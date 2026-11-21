"""Hybrid Fraud Risk Detection Model combining Cost-Sensitive LightGBM with Isolation Forest."""

import time
from typing import Any, Dict, List, Optional
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class HybridFraudDetector:
    """Prototype hybrid risk model: supervised LightGBM + Isolation Forest."""

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
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        chargeback_penalty: float = 2500.0,
        friction_cost: float = 120.0,
    ) -> Dict[str, Any]:
        """Fit on training data and select a decision threshold on separate validation data."""
        if len(X) == 0 or len(X_validation) == 0:
            raise ValueError("Training and validation datasets must be non-empty.")
        if set(y.unique()) != {0, 1} or set(y_validation.unique()) != {0, 1}:
            raise ValueError("Training and validation data must each contain both classes.")
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

        # 2. Fit Isolation Forest on legitimate training rows as an unsupervised anomaly heuristic.
        X_normal = X[y == 0].values
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_normal)

        # Select threshold on validation only. Test data is never passed to train().
        probs = self.lgbm_model.predict_proba(X_validation.values)[:, 1]
        ordered = np.unique(probs)
        midpoints = (ordered[:-1] + ordered[1:]) / 2.0
        thresholds = np.unique(np.concatenate(([0.0], midpoints, [1.0])))
        best_cost = float("inf")
        best_t = 0.50

        for t in thresholds:
            preds = (probs >= t).astype(int)
            fn = int(np.sum((preds == 0) & (y_validation.values == 1)))
            fp = int(np.sum((preds == 1) & (y_validation.values == 0)))
            total_loss = (fn * chargeback_penalty) + (fp * friction_cost)
            if total_loss < best_cost:
                best_cost = total_loss
                best_t = float(t)

        self.optimal_threshold = best_t

        return {
            "optimal_threshold": self.optimal_threshold,
            "min_business_loss": best_cost,
            "threshold_selection": "validation",
            "feature_importances": dict(zip(self.feature_names, self.lgbm_model.feature_importances_.tolist())),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate once on held-out data; cost uses the validation-selected threshold."""
        if self.lgbm_model is None or self.isolation_forest is None:
            raise ValueError("Detector must be trained before evaluation.")
        probs = self.lgbm_model.predict_proba(X.values)[:, 1]
        preds = (probs >= self.optimal_threshold).astype(int)
        anomaly_flags = self.isolation_forest.score_samples(X.values) < -0.65
        hybrid_preds = np.logical_or(preds == 1, anomaly_flags).astype(int)
        fn = int(np.sum((preds == 0) & (y.values == 1)))
        fp = int(np.sum((preds == 1) & (y.values == 0)))
        hybrid_fn = int(np.sum((hybrid_preds == 0) & (y.values == 1)))
        hybrid_fp = int(np.sum((hybrid_preds == 1) & (y.values == 0)))
        latencies = []
        sample_x = X.values[:1]
        for _ in range(100):
            started = time.perf_counter()
            self.predict_risk(sample_x)
            latencies.append((time.perf_counter() - started) * 1000)
        return {
            "roc_auc": float(roc_auc_score(y, probs)),
            "pr_auc": float(average_precision_score(y, probs)),
            "f1_at_validation_threshold": float(f1_score(y, preds, zero_division=0)),
            "precision_at_validation_threshold": float(precision_score(y, preds, zero_division=0)),
            "recall_at_validation_threshold": float(recall_score(y, preds, zero_division=0)),
            "threshold": self.optimal_threshold,
            "predicted_positive_count": int(preds.sum()),
            "false_negatives": fn,
            "false_positives": fp,
            "business_cost": float(fn * self.cost_matrix["chargeback_loss_fn"] + fp * self.cost_matrix["friction_cost_fp"]),
            "all_approve_baseline_business_cost": float(int(np.sum(y.values == 1)) * self.cost_matrix["chargeback_loss_fn"]),
            "anomaly_overlay": {
                "rule": "Isolation Forest score_samples < -0.65 OR supervised threshold",
                "flagged_count": int(anomaly_flags.sum()),
                "false_negatives": hybrid_fn,
                "false_positives": hybrid_fp,
                "business_cost": float(hybrid_fn * self.cost_matrix["chargeback_loss_fn"] + hybrid_fp * self.cost_matrix["friction_cost_fp"]),
                "precision": float(precision_score(y, hybrid_preds, zero_division=0)),
                "recall": float(recall_score(y, hybrid_preds, zero_division=0)),
            },
            "p95_model_scoring_latency_ms": float(np.percentile(latencies, 95)),
            "latency_method": "100 in-process single-row model calls; excludes HTTP, network, and infrastructure",
        }

    def predict_risk(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """Scores transaction with dual supervised probability and isolation anomaly signal."""
        # Supervised probability
        lgbm_prob = float(self.lgbm_model.predict_proba(X_sample)[0, 1])

        # Isolation anomaly score (negative values are outliers)
        iso_score = float(self.isolation_forest.score_samples(X_sample)[0])
        anomaly_flagged = bool(iso_score < -0.65)

        # Decision
        risk_triggered = bool(lgbm_prob >= self.optimal_threshold or anomaly_flagged)

        if lgbm_prob >= 0.75:
            decision = "BLOCK_TRANSACTION"
            action = "Immediate decline; notify cardholder via SMS"
        elif lgbm_prob >= self.optimal_threshold:
            decision = "REQUIRE_STEP_UP_AUTH"
            action = "Trigger 2-Factor Biometric or OTP challenge"
        elif anomaly_flagged:
            decision = "MANUAL_REVIEW"
            action = "Hold for review by a fraud analyst"
        else:
            decision = "APPROVE"
            action = "Frictionless checkout"

        return {
            "fraud_probability": round(lgbm_prob, 4),
            "optimal_threshold": self.optimal_threshold,
            "risk_triggered": risk_triggered,
            "anomaly_flagged": anomaly_flagged,
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
