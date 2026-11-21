"""Synthetic-data reproducibility and online-feature availability checks."""

import numpy as np

from data.generate_dataset import generate_fraud_dataset


def test_generator_reproducible_and_amount_deviation_uses_prior_events(tmp_path):
    first = generate_fraud_dataset(tmp_path / "first.csv", n_records=3_000, seed=7, n_accounts=600)
    second = generate_fraud_dataset(tmp_path / "second.csv", n_records=3_000, seed=7, n_accounts=600)
    assert first.equals(second)
    assert first["timestamp"].is_monotonic_increasing

    account = first.groupby("account_id").size().idxmax()
    events = first[first.account_id == account].sort_values("timestamp")
    assert np.isclose(events.amount_deviation_ratio.iloc[0], round(events.amount.iloc[0] / 1500.0, 3))
    expected = events.amount.iloc[1] / events.amount.iloc[0]
    assert np.isclose(events.amount_deviation_ratio.iloc[1], round(expected, 3))
