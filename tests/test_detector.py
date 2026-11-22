"""Unit tests for the hybrid fraud detector."""

import numpy as np
import pandas as pd
import pytest

from src.models.fraud_detector import HybridFraudDetector


def test_model_training_and_prediction_do_not_need_local_artifacts():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"signal": rng.normal(size=600), "noise": rng.normal(size=600)})
    y = pd.Series((X.signal > 1.0).astype(int))
    detector = HybridFraudDetector()
    detector.train(X.iloc[:450], y.iloc[:450], X.iloc[450:], y.iloc[450:])
    assert detector.lgbm_model is not None
    assert detector.isolation_forest is not None
    assert np.isfinite(detector.optimal_threshold)
    result = detector.predict_risk(X.iloc[[0]].to_numpy())
    assert "fraud_probability" in result
    assert result["decision"] in {"BLOCK_TRANSACTION", "REQUIRE_STEP_UP_AUTH", "MANUAL_REVIEW", "APPROVE"}


def test_training_requires_both_classes_for_threshold_selection():
    detector = HybridFraudDetector()
    X = pd.DataFrame({"signal": [0.0, 1.0, 0.1, 0.9]})
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValueError, match="both classes"):
        detector.train(X, y, X.iloc[:2], pd.Series([0, 0]))
