"""Checks for reproducibility and leakage boundaries in the synthetic benchmark."""

import pandas as pd
import pytest

from data.generate_dataset import generate_fraud_dataset
from scripts.train_and_evaluate import temporal_account_split


def test_generator_is_deterministic_and_produces_nontrivial_labels(tmp_path):
    first = generate_fraud_dataset(
        tmp_path / "first.csv", n_records=6_000, n_accounts=1_200, seed=27
    )
    second = generate_fraud_dataset(
        tmp_path / "second.csv", n_records=6_000, n_accounts=1_200, seed=27
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["is_fraud"].sum() > 30
    assert first["is_fraud"].mean() < 0.10


def test_split_is_future_ordered_and_account_disjoint(tmp_path):
    data = generate_fraud_dataset(
        tmp_path / "split.csv", n_records=12_000, n_accounts=2_400, seed=42
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    splits, overlaps = temporal_account_split(data)
    assert overlaps == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()


def test_split_rejects_account_overlap(tmp_path):
    data = generate_fraud_dataset(
        tmp_path / "overlap.csv", n_records=12_000, n_accounts=2_400, seed=42
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.loc[data["timestamp"] >= "2026-02-18", "account_id"] = data.loc[
        data["timestamp"] < "2026-02-06", "account_id"
    ].iloc[0]
    with pytest.raises(ValueError, match="Account overlap"):
        temporal_account_split(data)
