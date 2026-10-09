"""
Tests for src/serving/inference.py.

A small model is trained on synthetic data and written out as a model bundle
(the same three files scripts/export_model.py produces), then loaded by
ChurnPredictor exactly as a service would.

Run from the project root:  pytest tests/test_inference.py -v
"""

import json
import os
import sys

import numpy as np
import pytest
from telco_data import make_telco_df
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.build_features import (  # noqa: E402
    build_features,
    transform_features,
)
from src.serving.inference import ChurnPredictor  # noqa: E402

THRESHOLD = 0.4


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Train a tiny model on synthetic data and write a model bundle."""
    raw = make_telco_df(n=300)
    features, schema = build_features(preprocess_data(raw.copy()), "Churn")
    X, y = features.drop(columns=["Churn"]), features["Churn"]
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(X, y)

    folder = tmp_path_factory.mktemp("bundle")
    model.save_model(str(folder / "model.ubj"))
    (folder / "feature_schema.json").write_text(json.dumps(schema))
    info = {"threshold": THRESHOLD}
    (folder / "model_info.json").write_text(json.dumps(info))
    return folder, raw, model, schema


def customer(raw, i):
    """One raw customer record, as an API would receive it."""
    return raw.drop(columns="Churn").iloc[i].to_dict()


def test_single_prediction_matches_batch_scoring(bundle):
    """Scoring customers one by one equals scoring them all at once."""
    folder, raw, model, schema = bundle
    predictor = ChurnPredictor(str(folder))

    batch = transform_features(
        preprocess_data(raw.drop(columns="Churn").copy()), schema
    )
    expected = model.predict_proba(batch)[:, 1]

    for i in range(0, len(raw), 20):
        score = predictor.predict(customer(raw, i))["churn_score"]
        assert score == pytest.approx(expected[i], abs=1e-6)


def test_threshold_decides_the_label(bundle):
    folder, raw, _, _ = bundle
    predictor = ChurnPredictor(str(folder))

    for i in range(0, len(raw), 15):
        result = predictor.predict(customer(raw, i))
        assert result["threshold"] == THRESHOLD
        likely = result["churn_score"] >= THRESHOLD
        assert result["likely_to_churn"] == likely
        assert result["label"] == (
            "Likely to churn" if likely else "Not likely to churn"
        )


def test_scores_vary_between_customers(bundle):
    """Guards against every customer being encoded as the baseline."""
    folder, raw, _, _ = bundle
    predictor = ChurnPredictor(str(folder))
    scores = {
        predictor.predict(customer(raw, i))["churn_score"] for i in range(40)
    }
    assert len(scores) > 5


def test_missing_field_is_rejected(bundle):
    folder, raw, _, _ = bundle
    record = customer(raw, 0)
    del record["Contract"]
    with pytest.raises(ValueError, match="Missing required columns"):
        ChurnPredictor(str(folder)).predict(record)


def test_unseen_category_is_rejected(bundle):
    folder, raw, _, _ = bundle
    record = customer(raw, 0)
    record["PaymentMethod"] = "Cash"
    with pytest.raises(ValueError, match="PaymentMethod"):
        ChurnPredictor(str(folder)).predict(record)


def test_model_dir_environment_variable(bundle, monkeypatch):
    folder, raw, _, _ = bundle
    monkeypatch.setenv("MODEL_DIR", str(folder))
    result = ChurnPredictor().predict(customer(raw, 0))
    assert 0.0 <= result["churn_score"] <= 1.0


def test_mismatched_bundle_is_rejected(bundle, tmp_path):
    """A schema that doesn't match the model must fail at load time."""
    folder, _, _, schema = bundle
    for name in ("model.ubj", "model_info.json"):
        (tmp_path / name).write_bytes((folder / name).read_bytes())
    broken = dict(schema, columns=schema["columns"][:-1])
    (tmp_path / "feature_schema.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="do not belong together"):
        ChurnPredictor(str(tmp_path))
