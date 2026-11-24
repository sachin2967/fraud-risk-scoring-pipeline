import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_openml_benchmark import select_cost_threshold
from evaluate_stability import development_folds


def test_development_folds_do_not_split_simultaneous_transactions():
    frame = pd.DataFrame({"Time": np.repeat(np.arange(100), 2), "Class": [0, 1] * 100})
    folds = list(development_folds(frame))
    assert len(folds) == 3
    for train, validation in folds:
        assert train.Time.max() < validation.Time.min()
    assert len(set().union(*(set(v.index) for _, v in folds))) == sum(
        len(v) for _, v in folds
    )


def test_cost_sensitivity_can_choose_different_operating_policies():
    labels, scores = np.array([0, 0, 1]), np.array([0.8, 0.7, 0.6])
    low, low_cost = select_cost_threshold(labels, scores, false_negative_cost=1)
    high, high_cost = select_cost_threshold(labels, scores, false_negative_cost=10)
    assert low > 1 and low_cost == 1
    assert high == 0 and high_cost == 2


@pytest.mark.parametrize("cost", [-1, np.inf, np.nan])
def test_invalid_costs_are_rejected(cost):
    with pytest.raises(ValueError):
        select_cost_threshold(
            np.array([0, 1]), np.array([0.1, 0.9]), false_negative_cost=cost
        )
