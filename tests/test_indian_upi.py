"""Coverage for Indian UPI feature enrichment and drain rules."""

import joblib
import numpy as np
import pandas as pd

from src.data.indian_upi_adapter import IndianUPIPaymentsAdapter
from src.models.fraud_detector import HybridFraudDetector


def test_adapter_parses_transaction_types_and_balance_drain():
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC_1",
                "amount_inr": 8500.0,
                "txn_type": "upi_p2p",
                "old_balance": 10000.0,
                "new_balance": 1500.0,
                "hour_of_day": 3,
                "upi_id": "sachin@okaxis",
                "device_trust_score": 0.2,
            },
            {
                "account_id": "ACC_1",
                "amount_inr": 150.0,
                "txn_type": "UPI_P2M",
                "old_balance": 1500.0,
                "new_balance": 1350.0,
                "hour_of_day": 14,
                "upi_id": "merchant_store@paytm",
                "device_trust_score": 0.9,
            },
        ]
    )
    enriched = IndianUPIPaymentsAdapter().transform(transactions)

    assert enriched.txn_type.tolist() == ["UPI_P2P", "UPI_P2M"]
    assert np.allclose(enriched.balance_drain_ratio, [0.85, 0.1])
    assert enriched.is_zero_balance_after_txn.tolist() == [0, 0]
    assert enriched.upi_velocity_1h.tolist() == [1, 2]
    assert enriched.amount_inr_formatted.tolist() == ["₹8,500.00", "₹150.00"]


def test_pipeline_encodes_upi_features():
    from src.features.pipeline import FraudFeaturePipeline

    records = pd.DataFrame(
        [
            {"amount": 150.0, "amount_inr": 150.0, "merchant_category": "Grocery",
             "hour_of_day": 14, "time_since_last_sec": 60.0, "velocity_1h": 1,
             "geo_distance_km": 1.0, "device_trust_score": 0.9, "is_foreign": 0,
             "amount_deviation_ratio": 0.1, "txn_type": "UPI_P2M", "is_fraud": 0},
            {"amount": 45000.0, "amount_inr": 45000.0, "merchant_category": "Digital_Wallet",
             "hour_of_day": 3, "time_since_last_sec": 5.0, "velocity_1h": 5,
             "geo_distance_km": 200.0, "device_trust_score": 0.05, "is_foreign": 0,
             "amount_deviation_ratio": 10.0, "txn_type": "UPI_P2P", "is_fraud": 1,
             "balance_drain_ratio": 1.0, "is_zero_balance_after_txn": 1,
             "upi_velocity_1h": 5, "dest_mule_risk_score": 0.9},
        ]
    )
    enriched = IndianUPIPaymentsAdapter().transform(records)
    pipeline = FraudFeaturePipeline()
    features, _ = pipeline.fit_transform(enriched)

    for feature in ("balance_drain_ratio", "is_zero_balance_after_txn", "upi_velocity_1h", "txn_type_UPI_P2P"):
        assert feature in features


def test_adapter_counts_only_upi_attempts_in_trailing_hour():
    transactions = pd.DataFrame(
        [
            {"account_id": "ACC_1", "timestamp": "2026-01-01T00:00:00", "txn_type": "UPI_P2P"},
            {"account_id": "ACC_1", "timestamp": "2026-01-01T00:30:00", "txn_type": "NET_BANKING_IMPS"},
            {"account_id": "ACC_1", "timestamp": "2026-01-01T00:45:00", "txn_type": "UPI_P2M"},
            {"account_id": "ACC_1", "timestamp": "2026-01-01T01:01:00", "txn_type": "WALLET_CASH_OUT"},
        ]
    )

    enriched = IndianUPIPaymentsAdapter().transform(transactions)

    assert enriched.upi_velocity_1h.tolist() == [1, 1, 2, 1]


def test_adapter_preserves_supplied_velocity_and_does_not_assume_unknown_balance_is_zero():
    enriched = IndianUPIPaymentsAdapter().transform(
        pd.DataFrame(
            [{
                "account_id": "ACC_1", "amount_inr": 1200.0, "txn_type": "UPI_P2P",
                "hour_of_day": 3, "upi_velocity_1h": 7,
            }]
        )
    )

    assert enriched.loc[0, "upi_velocity_1h"] == 7
    assert enriched.loc[0, "is_zero_balance_after_txn"] == 0


def test_detector_flags_midnight_account_drain():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    detector = HybridFraudDetector.load(str(root / "models" / "hybrid_fraud_detector.joblib"))
    pipeline = joblib.load(str(root / "models" / "feature_pipeline.joblib"))
    attack = IndianUPIPaymentsAdapter().transform(
        pd.DataFrame([{
            "transaction_id": "TXN_DRAIN", "account_id": "ACC_ATTACK",
            "amount": 45000.0, "amount_inr": 45000.0,
            "merchant_category": "Digital_Wallet", "hour_of_day": 3,
            "time_since_last_sec": 5.0, "velocity_1h": 5,
            "geo_distance_km": 250.0, "device_trust_score": 0.05,
            "is_foreign": 0, "amount_deviation_ratio": 30.0,
            "txn_type": "UPI_P2P", "upi_id": "sachin@okaxis",
            "old_balance": 45000.0, "new_balance": 0.0,
            "dest_mule_risk_score": 0.9,
        }])
    )
    scored = detector.predict_risk(pipeline.transform(attack).values, attack.iloc[0].to_dict())

    assert scored["fraud_probability"] >= 0.95
    assert scored["decision"] == "BLOCK_AND_FREEZE_UPI"


def test_api_accepts_inr_only_upi_payload():
    from fastapi.testclient import TestClient

    from api.app import app

    response = TestClient(app).post(
        "/api/score",
        json={
            "transaction_id": "TXN_UPI_DRAIN",
            "account_id": "ACC_UPI_DRAIN",
            "amount_inr": 45000.0,
            "upi_id": "sachin@okaxis",
            "txn_type": "UPI_P2P",
            "old_balance": 45000.0,
            "new_balance": 0.0,
            "merchant_category": "Digital_Wallet",
            "hour_of_day": 3,
            "time_since_last_sec": 5.0,
            "velocity_1h": 5,
            "geo_distance_km": 250.0,
            "device_trust_score": 0.05,
            "is_foreign": 0,
            "amount_deviation_ratio": 30.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["fraud_probability"] >= 0.95
    assert response.json()["decision"] == "BLOCK_AND_FREEZE_UPI"
