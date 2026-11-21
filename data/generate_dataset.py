"""Generate a reproducible, explicitly synthetic transaction risk benchmark.

Labels are sampled from overlapping risk processes rather than deterministic rules.
This creates enough positives for a useful software/evaluation exercise, but it is
not a substitute for a licensed, representative payment dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


MERCHANTS = np.array(
    ["Grocery", "Dining", "Electronics", "Travel", "Gaming", "Luxury_Jewelry", "Digital_Wallet"]
)
MERCHANT_PROBS = np.array([0.34, 0.24, 0.14, 0.10, 0.08, 0.05, 0.05])


def generate_fraud_dataset(
    output_path: Path, n_records: int = 60_000, seed: int = 42, n_accounts: int = 12_000
) -> pd.DataFrame:
    """Write synthetic data with disjoint account cohorts across chronological windows."""
    if n_records < 3_000 or n_accounts < 600:
        raise ValueError("Use at least 3,000 transactions and 600 accounts for the temporal benchmark.")

    rng = np.random.default_rng(seed)
    # Each account belongs to exactly one chronological cohort. This intentionally
    # enables a future-window AND account-disjoint evaluation without a split leak.
    cohort_sizes = np.array([0.60, 0.20, 0.20])
    account_cohort = np.repeat(np.arange(3), np.floor(cohort_sizes * n_accounts).astype(int))
    if len(account_cohort) < n_accounts:
        account_cohort = np.concatenate([account_cohort, np.full(n_accounts - len(account_cohort), 2)])
    rng.shuffle(account_cohort)

    # Slightly unequal event counts are realistic and exercise clustered sampling.
    account_weights = rng.lognormal(mean=0.0, sigma=0.45, size=n_accounts)
    account_weights /= account_weights.sum()
    account_ids = np.arange(n_accounts)
    chosen_accounts = rng.choice(account_ids, size=n_records, p=account_weights)
    cohorts = account_cohort[chosen_accounts]
    starts = np.array([0, 36, 48])
    widths = np.array([36, 12, 12])
    day = starts[cohorts] + rng.integers(0, widths[cohorts])
    minute = day * 1440 + rng.integers(0, 1440, size=n_records)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(minute, unit="m")
    hour = timestamps.hour.to_numpy()

    account_propensity = rng.normal(0.0, 0.65, size=n_accounts)
    account_spend = rng.lognormal(np.log(1300.0), 0.75, size=n_accounts)
    base_amount = rng.exponential(account_spend[chosen_accounts] * 0.65) + 35.0
    merchant = rng.choice(MERCHANTS, size=n_records, p=MERCHANT_PROBS)
    time_since_last = rng.lognormal(np.log(5_400), 1.0, size=n_records)
    velocity = rng.poisson(0.7, size=n_records)
    geo = rng.exponential(11.0, size=n_records)
    device_trust = np.clip(rng.beta(7.0, 2.4, size=n_records), 0.02, 1.0)
    foreign = rng.binomial(1, 0.045, size=n_records)

    log_amount_ratio = np.log1p(base_amount / np.maximum(account_spend[chosen_accounts], 80.0))
    risky_merchant = np.isin(merchant, ["Gaming", "Luxury_Jewelry", "Digital_Wallet"]).astype(float)
    night = np.isin(hour, [0, 1, 2, 3, 4]).astype(float)
    burst = (velocity >= 3).astype(float)
    # Continuous, overlapping signal effects; no single hand-coded rule defines a label.
    linear_risk = (
        -5.4
        + 1.20 * (log_amount_ratio - 1.1)
        + 1.80 * (1.0 - device_trust)
        + 0.65 * np.log1p(geo)
        + 1.00 * burst
        + 0.40 * risky_merchant
        + 0.20 * night
        + 0.40 * foreign
        + 0.34 * (time_since_last < 90)
        + account_propensity[chosen_accounts]
    )
    # A new attack archetype appears in the last period. It is partially observable
    # through higher velocity and device novelty, while the model sees no cohort flag.
    late = cohorts == 2
    new_attack = late & (rng.random(n_records) < 0.035)
    velocity[new_attack] += rng.integers(2, 5, size=new_attack.sum())
    device_trust[new_attack] *= rng.uniform(0.18, 0.55, size=new_attack.sum())
    linear_risk += new_attack.astype(float) * 0.9

    fraud_probability = 1.0 / (1.0 + np.exp(-linear_risk))
    labels = rng.binomial(1, fraud_probability)
    # Amount is a post-risk transaction attribute generated without directly copying
    # the label; the same broad distributions apply to fraud and legitimate events.
    amount = base_amount * np.where(new_attack, rng.uniform(1.3, 3.0, n_records), 1.0)
    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:08d}" for i in range(n_records)],
            "account_id": [f"ACC_{i:06d}" for i in chosen_accounts],
            "timestamp": timestamps.astype(str),
            "amount": np.round(amount, 2),
            "merchant_category": merchant,
            "hour_of_day": hour,
            "time_since_last_sec": np.round(time_since_last, 1),
            "velocity_1h": velocity,
            "geo_distance_km": np.round(geo, 2),
            "device_trust_score": np.round(device_trust, 3),
            "is_foreign": foreign,
            "is_fraud": labels,
        }
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)
    # Make the account deviation feature computable online: each event uses only
    # that account's earlier observed amounts, with a fixed cold-start prior.
    # Do not expose the generator's latent account_spend parameter as a feature.
    account_amount_sum = df.groupby("account_id", sort=False)["amount"].cumsum() - df["amount"]
    account_amount_count = df.groupby("account_id", sort=False).cumcount()
    prior_average = np.divide(
        account_amount_sum.to_numpy(),
        account_amount_count.to_numpy(),
        out=np.full(len(df), 1500.0),
        where=account_amount_count.to_numpy() > 0,
    )
    df["amount_deviation_ratio"] = np.round(df["amount"].to_numpy() / np.maximum(prior_average, 80.0), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    positive = int(df["is_fraud"].sum())
    print(f"Synthetic benchmark: {len(df):,} transactions, {positive:,} positives ({positive / len(df):.2%})")
    return df


if __name__ == "__main__":
    generate_fraud_dataset(Path(__file__).resolve().parent / "transactions.csv")
