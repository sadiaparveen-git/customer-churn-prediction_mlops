"""
Tests for src/features/build_features.py.

The point of the fit/transform split: a single customer must be encoded
exactly as the same customer would be inside a training batch.

Run from the project root:  pytest tests/test_features.py -v
"""

import json
import os
import sys

import pandas as pd
import pytest
from telco_data import make_telco_df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.build_features import (  # noqa: E402
    build_features,
    fit_features,
    transform_features,
)


@pytest.fixture(scope="module")
def prepared():
    """Preprocessed synthetic data, its features and the fitted schema."""
    df = preprocess_data(make_telco_df())
    features, schema = build_features(df, "Churn")
    return df, features, schema


def test_single_row_matches_batch(prepared):
    """Encoding a customer alone equals encoding it within the batch."""
    df, features, schema = prepared
    X = features.drop(columns=["Churn"])
    for i in range(0, len(df), 25):
        one = transform_features(df.iloc[[i]], schema)
        pd.testing.assert_frame_equal(one, X.iloc[[i]])


def test_non_baseline_categories_are_encoded(prepared):
    """The case a naive one-row get_dummies gets wrong: it drops them."""
    df, _, schema = prepared
    row = df.iloc[[0]].copy()
    row["Contract"] = "Two year"
    row["InternetService"] = "Fiber optic"
    out = transform_features(row, schema).iloc[0]

    assert out["Contract_Two year"] == 1
    assert out["Contract_One year"] == 0
    assert out["InternetService_Fiber optic"] == 1


def test_baseline_category_is_all_zeros(prepared):
    df, _, schema = prepared
    row = df.iloc[[0]].copy()
    row["Contract"] = schema["categorical_levels"]["Contract"][0]
    out = transform_features(row, schema).iloc[0]
    assert out[[c for c in out.index if c.startswith("Contract_")]].sum() == 0


def test_output_has_schema_columns_in_order_and_is_numeric(prepared):
    df, _, schema = prepared
    out = transform_features(df.iloc[[0]], schema)
    assert list(out.columns) == schema["columns"]
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_schema_survives_json_round_trip(prepared):
    df, _, schema = prepared
    restored = json.loads(json.dumps(schema))
    pd.testing.assert_frame_equal(
        transform_features(df, schema), transform_features(df, restored)
    )


def test_unseen_category_raises(prepared):
    df, _, schema = prepared
    row = df.iloc[[0]].copy()
    row["Contract"] = "Weekly"
    with pytest.raises(ValueError, match="Contract"):
        transform_features(row, schema)


def test_invalid_binary_value_raises(prepared):
    df, _, schema = prepared
    row = df.iloc[[0]].copy()
    row["gender"] = "Other"
    with pytest.raises(ValueError, match="gender"):
        transform_features(row, schema)


def test_missing_column_raises(prepared):
    df, _, schema = prepared
    with pytest.raises(ValueError, match="Missing required columns"):
        transform_features(df.iloc[[0]].drop(columns=["tenure"]), schema)


def test_fit_rejects_single_category_column():
    df = pd.DataFrame({"a": ["x", "x", "x"], "Churn": [0, 1, 0]})
    with pytest.raises(ValueError, match="at least 2"):
        fit_features(df, "Churn")
