"""
Tests for the score reference (src/serving/score_reference.py and
scripts/build_score_reference.py).

Run from the project root:  pytest tests/test_score_reference.py -v
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from telco_data import make_telco_df
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.serving.inference import ChurnPredictor  # noqa: E402
from src.serving.score_reference import (  # noqa: E402
    REFERENCE_FILE,
    ScoreReference,
    build_reference,
    save_reference,
)

THRESHOLD = 0.5


def make_scores(n=1000, seed=0):
    """Scores where a higher score really does mean more churn."""
    rng = np.random.default_rng(seed)
    scores = rng.beta(0.7, 1.0, size=n)
    churned = (rng.random(n) < scores).astype(int)
    return scores, churned


def reference(**changes):
    scores, churned = make_scores()
    args = dict(
        scores=scores,
        churned=churned,
        threshold=THRESHOLD,
        low_max=0.2,
        very_high_min=0.85,
        run_id="run-1",
    )
    args.update(changes)
    return build_reference(**args)


# === Building the reference ===


def test_every_customer_lands_in_exactly_one_band():
    ref = reference()
    assert sum(b["customers"] for b in ref["bands"]) == ref["customers"]
    assert [b["name"] for b in ref["bands"]] == [
        "Low",
        "Moderate",
        "High",
        "Very high",
    ]


def test_churn_rate_rises_with_the_band():
    rates = [b["churn_rate"] for b in reference()["bands"]]
    assert rates == sorted(rates)


def test_high_band_starts_at_the_decision_threshold():
    bands = {b["name"]: b for b in reference()["bands"]}
    assert bands["High"]["min"] == THRESHOLD
    assert bands["Moderate"]["max"] == THRESHOLD


def test_bands_with_too_few_customers_are_refused():
    with pytest.raises(ValueError, match="holds only"):
        reference(very_high_min=0.9999)


def test_bands_whose_churn_rate_does_not_rise_are_refused():
    scores, _ = make_scores()
    # Flip the relationship so the top band has the lowest churn rate
    inverted = (scores < 0.5).astype(int)
    with pytest.raises(ValueError, match="does not rise"):
        reference(churned=inverted)


def test_band_edges_must_be_in_order():
    with pytest.raises(ValueError, match="Band edges"):
        reference(low_max=0.6)


# === Looking a score up ===


def lookup_table():
    return ScoreReference(reference())


def test_percentile_never_decreases_as_the_score_rises():
    ref = lookup_table()
    percentiles = [
        ref.lookup(s)["risk_percentile"] for s in np.linspace(0, 1, 101)
    ]
    assert percentiles == sorted(percentiles)


def test_extreme_scores_get_extreme_percentiles():
    ref = lookup_table()
    assert ref.lookup(0.0)["risk_percentile"] == 0
    assert ref.lookup(1.0)["risk_percentile"] == 100


def test_median_score_is_near_the_50th_percentile():
    scores, _ = make_scores()
    median = float(np.median(scores))
    percentile = lookup_table().lookup(median)["risk_percentile"]
    assert 49 <= percentile <= 51


@pytest.mark.parametrize(
    "score, band",
    [
        (0.0, "Low"),
        (0.199, "Low"),
        (0.2, "Moderate"),
        (0.499, "Moderate"),
        (THRESHOLD, "High"),
        (0.849, "High"),
        (0.85, "Very high"),
        (1.0, "Very high"),
    ],
)
def test_band_edges_are_inclusive_at_the_bottom(score, band):
    assert lookup_table().lookup(score)["risk_band"] == band


def test_saved_file_round_trips(tmp_path):
    ref = reference()
    save_reference(ref, str(tmp_path / REFERENCE_FILE))
    loaded = json.loads((tmp_path / REFERENCE_FILE).read_text())
    assert loaded == json.loads(json.dumps(ref))
    reloaded = ScoreReference.load(str(tmp_path))
    assert reloaded.lookup(0.7) == lookup_table().lookup(0.7)


def test_missing_file_means_no_reference(tmp_path):
    assert ScoreReference.load(str(tmp_path)) is None


# === The build script, on a small synthetic model ===


@pytest.fixture(scope="module")
def script():
    path = os.path.join(PROJECT_ROOT, "scripts", "build_score_reference.py")
    spec = importlib.util.spec_from_file_location("build_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A trained model bundle plus the processed data it was trained from."""
    raw = make_telco_df(n=2500, seed=3)
    # Churn that really depends on tenure, so the bands have rising rates
    rng = np.random.default_rng(5)
    p_churn = np.clip(1 - raw["tenure"] / 72, 0.02, 0.98) ** 1.5
    raw["Churn"] = np.where(rng.random(len(raw)) < p_churn, "Yes", "No")

    processed = preprocess_data(raw.copy())
    features, schema = build_features(processed, "Churn")
    X, y = features.drop(columns=["Churn"]), features["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    # Band edges taken from this model's own scores (20th/50th/80th percent)
    scores = model.predict_proba(X_test)[:, 1]
    low_max, threshold, very_high_min = np.percentile(scores, [20, 50, 80])

    # Metrics as the pipeline would have logged them for this split
    decided = scores >= threshold
    info = {
        "run_id": "run-1",
        "threshold": float(threshold),
        "params": {"target": "Churn", "test_size": 0.2},
        "test_metrics": {
            "recall": recall_score(y_test, decided),
            "precision": precision_score(y_test, decided),
        },
    }
    folder = tmp_path_factory.mktemp("bundle")
    model.save_model(str(folder / "model.ubj"))
    (folder / "feature_schema.json").write_text(json.dumps(schema))
    (folder / "model_info.json").write_text(json.dumps(info))
    data = tmp_path_factory.mktemp("data") / "processed.csv"
    processed.to_csv(data, index=False)
    edges = {"low_max": low_max, "very_high_min": very_high_min}
    return folder, data, raw, edges


@pytest.fixture(autouse=True)
def small_bands_are_fine_for_small_data(monkeypatch):
    """The synthetic test set is small; real data needs 100 per band."""
    monkeypatch.setattr(
        "src.serving.score_reference.MIN_BAND_SIZE", 20, raising=True
    )


def run_script(script, folder, data, edges, **changes):
    from argparse import Namespace

    args = dict(model_dir=str(folder), input=str(data), **edges)
    args.update(changes)
    script.main(Namespace(**args))


def test_script_writes_a_reference_that_the_predictor_uses(script, world):
    folder, data, raw, edges = world
    run_script(script, folder, data, edges)
    assert (folder / REFERENCE_FILE).exists()

    result = ChurnPredictor(str(folder)).predict(
        raw.drop(columns="Churn").iloc[0].to_dict()
    )
    assert 0 <= result["risk_percentile"] <= 100
    assert result["risk_band"] in {"Low", "Moderate", "High", "Very high"}
    assert 0 <= result["observed_churn_rate"] <= 1


def test_script_refuses_data_that_does_not_match_the_model(
    script, world, tmp_path
):
    folder, data, _, edges = world
    other = pd.read_csv(data).sample(frac=0.8, random_state=1)
    other_path = tmp_path / "other.csv"
    other.to_csv(other_path, index=False)

    with pytest.raises(SystemExit, match="does not match the model"):
        run_script(script, folder, other_path, edges)


# === The predictor ===


def test_predictor_without_a_reference_returns_none(world, tmp_path):
    folder, _, raw, _ = world
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("model.ubj", "feature_schema.json", "model_info.json"):
        (bare / name).write_bytes((folder / name).read_bytes())

    result = ChurnPredictor(str(bare)).predict(
        raw.drop(columns="Churn").iloc[0].to_dict()
    )
    assert result["risk_percentile"] is None
    assert result["risk_band"] is None
    assert result["observed_churn_rate"] is None


def test_predictor_rejects_a_reference_from_another_run(world, tmp_path):
    folder, _, _, _ = world
    stale = tmp_path / "stale"
    stale.mkdir()
    for name in ("model.ubj", "feature_schema.json", "model_info.json"):
        (stale / name).write_bytes((folder / name).read_bytes())
    other_run = reference(run_id="a-different-run")
    save_reference(other_run, str(stale / REFERENCE_FILE))

    with pytest.raises(ValueError, match="different model run"):
        ChurnPredictor(str(stale))
