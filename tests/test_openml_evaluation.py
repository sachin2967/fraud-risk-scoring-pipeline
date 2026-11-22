import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_openml_benchmark import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    chronological_split,
    select_cost_threshold,
    select_f1_threshold,
)


def test_chronological_split_is_ordered_and_disjoint():
    frame = pd.DataFrame({"Time": np.arange(100), "Class": [0, 1] * 50})
    train, validation, test = chronological_split(frame.sample(frac=1, random_state=3))
    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15
    assert train.Time.max() <= validation.Time.min()
    assert validation.Time.max() <= test.Time.min()


def test_chronological_split_keeps_equal_timestamps_together():
    times = np.repeat(np.arange(50), 2)
    frame = pd.DataFrame({"Time": times, "Class": [0, 1] * 50})
    train, validation, test = chronological_split(frame.sample(frac=1, random_state=9))
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()


def test_f1_threshold_selection_uses_validation_labels():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    assert 0.3 <= select_f1_threshold(y, scores) <= 0.7


def test_cost_threshold_matches_minimum_validation_cost():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.3, 0.9])
    threshold, cost = select_cost_threshold(y, scores)
    predictions = scores >= threshold
    expected = int(np.sum((~predictions) & (y == 1))) * FALSE_NEGATIVE_COST
    expected += int(np.sum(predictions & (y == 0))) * FALSE_POSITIVE_COST
    assert cost == expected


def test_cost_selection_can_choose_all_approve_with_exact_unit_scores():
    labels = np.array([0] * 100 + [1])
    scores = np.array([1.0] * 100 + [0.0])
    threshold, cost = select_cost_threshold(labels, scores)
    assert threshold > 1.0
    assert cost == FALSE_NEGATIVE_COST


@pytest.mark.parametrize("labels,scores", [(["0", "1"], [0.1, 0.9]),
    ([0, 1], [0.2]), ([0, 1], [0.2, float("nan")]), ([0, 1], [0.2, 1.1])])
def test_cost_selection_rejects_invalid_or_string_labels(labels, scores):
    with pytest.raises(ValueError):
        select_cost_threshold(np.array(labels), np.array(scores))
