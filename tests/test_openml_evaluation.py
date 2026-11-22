import numpy as np
import pandas as pd

from scripts.evaluate_openml_benchmark import chronological_split, select_f1_threshold


def test_chronological_split_is_ordered_and_disjoint():
    frame = pd.DataFrame({"Time": np.arange(100), "Class": [0, 1] * 50})
    train, validation, test = chronological_split(frame.sample(frac=1, random_state=3))
    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15
    assert train.Time.max() <= validation.Time.min()
    assert validation.Time.max() <= test.Time.min()


def test_f1_threshold_selection_uses_validation_labels():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    assert 0.3 <= select_f1_threshold(y, scores) <= 0.7
