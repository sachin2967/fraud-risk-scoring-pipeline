"""Generates a realistic financial transaction stream benchmark with 99.6% class imbalance."""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_fraud_dataset(output_path: Path, n_records: int = 15000, seed: int = 42):
    """Generates synthetic but realistic card transactions with velocity and behavioral patterns."""
    np.random.seed(seed)

    n_accounts = 1200
    accounts = [f"ACC_{i:05d}" for i in range(1, n_accounts + 1)]
    merchants = ["Grocery", "Dining", "Electronics", "Travel", "Gaming", "Luxury_Jewelry", "Digital_Wallet"]

    data = []
    # Account historical spending averages
    account_avg_spend = {acc: np.random.exponential(1200) + 200 for acc in accounts}

    base_time = pd.Timestamp("2026-01-01 00:00:00")

    for i in range(1, n_records + 1):
        txn_id = f"TXN_{i:07d}"
        acc = np.random.choice(accounts)
        merchant = np.random.choice(merchants, p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03])

        # Time simulation (rolling across 30 days)
        minute_offset = int(np.random.uniform(0, 30 * 24 * 60))
        txn_time = base_time + pd.Timedelta(minutes=minute_offset)
        hour_of_day = txn_time.hour

        # Base normal behavior
        avg_spend = account_avg_spend[acc]
        amount = float(np.random.exponential(avg_spend * 0.7) + 50.0)
        time_since_last = float(np.random.exponential(3600 * 12) + 60.0)  # seconds
        velocity_1h = int(np.random.poisson(0.4))
        geo_dist = float(np.random.exponential(8.0))  # km
        device_trust = float(np.clip(np.random.beta(8, 2), 0.05, 1.0))
        is_foreign = int(np.random.choice([0, 1], p=[0.96, 0.04]))

        # Base fraud probability is 0.4% (extreme imbalance)
        is_fraud = 0

        # Inject known realistic fraud attack patterns:
        # Pattern A: High amount at midnight in risky category (Gaming/Crypto/Luxury) with new device
        if (hour_of_day in [1, 2, 3, 4]) and (merchant in ["Gaming", "Luxury_Jewelry", "Digital_Wallet"]) and (device_trust < 0.35):
            if np.random.rand() < 0.65:
                is_fraud = 1
                amount = amount * float(np.random.uniform(3.5, 7.0))
                velocity_1h += int(np.random.randint(2, 6))

        # Pattern B: Card velocity burst (card testing / credential stuffing)
        elif velocity_1h > 3 and time_since_last < 120 and geo_dist > 50:
            if np.random.rand() < 0.75:
                is_fraud = 1
                amount = float(np.random.uniform(500, 2500))

        # Organic rare fraud
        elif np.random.rand() < 0.0015:
            is_fraud = 1
            amount = amount * float(np.random.uniform(4.0, 9.0))
            device_trust = float(np.random.uniform(0.05, 0.25))

        amount_dev_ratio = round(amount / max(avg_spend, 50.0), 3)

        data.append({
            "transaction_id": txn_id,
            "account_id": acc,
            "timestamp": str(txn_time),
            "amount": round(amount, 2),
            "merchant_category": merchant,
            "hour_of_day": hour_of_day,
            "time_since_last_sec": round(time_since_last, 1),
            "velocity_1h": velocity_1h,
            "geo_distance_km": round(geo_dist, 2),
            "device_trust_score": round(device_trust, 3),
            "is_foreign": is_foreign,
            "amount_deviation_ratio": amount_dev_ratio,
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(data).sort_values("timestamp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Dataset generated at: {output_path}")
    print(f"Total Transactions: {len(df):,} | Fraudulent: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "transactions.csv"
    generate_fraud_dataset(out)
