"""
Tests for src/models/tune.py.

The selection rule is tested directly on hand-made trial results; the full
tuner is tested on a small synthetic classification problem.

Run from the project root:  pytest tests/test_tune.py -v
"""

import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tune import _select_best, tune_model  # noqa: E402


def trial(name, precision, recall):
    return {
        "params": {"id": name},
        "threshold": 0.3,
        "precision": precision,
        "recall": recall,
    }


def test_select_best_prefers_precision_above_the_recall_floor():
    candidates = [
        trial("high_precision_low_recall", 0.70, 0.70),  # below the floor
        trial("meets_floor", 0.50, 0.80),
        trial("better_precision", 0.55, 0.81),
    ]
    best, floor_met = _select_best(candidates, 0.80, 0.01)
    assert floor_met
    assert best["params"]["id"] == "better_precision"


def test_select_best_takes_higher_recall_at_similar_precision():
    candidates = [
        trial("best_precision", 0.500, 0.82),
        trial("similar_precision_more_recall", 0.495, 0.88),  # within 0.01
        trial("too_far_below", 0.450, 0.95),  # precision gap > tolerance
    ]
    best, _ = _select_best(candidates, 0.80, 0.01)
    assert best["params"]["id"] == "similar_precision_more_recall"


def test_select_best_falls_back_to_highest_recall_if_floor_unmet():
    candidates = [trial("a", 0.6, 0.60), trial("b", 0.5, 0.75)]
    best, floor_met = _select_best(candidates, 0.80, 0.01)
    assert not floor_met
    assert best["params"]["id"] == "b"


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=500, n_features=10, weights=[0.73], random_state=0
    )
    return pd.DataFrame(X).add_prefix("f"), pd.Series(y)


def test_tune_model_returns_params_and_threshold_in_range(data):
    X, y = data
    params, threshold = tune_model(X, y, min_recall=0.5)

    assert 0.25 <= threshold <= 0.5
    assert params["scale_pos_weight"] >= 1.0  # class weight is tuned
    assert "threshold" not in params  # not an XGBoost parameter


def test_tune_model_is_reproducible(data):
    X, y = data
    first = tune_model(X, y, min_recall=0.5)
    second = tune_model(X, y, min_recall=0.5)
    assert first == second
