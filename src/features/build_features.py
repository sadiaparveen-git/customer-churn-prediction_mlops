"""
Feature engineering, split into a fit step and a transform step.

- fit_features learns a small "schema" from the training data: which columns
  are binary, which are multi-category (and their categories), and the exact
  order of the final feature columns.
- transform_features applies that schema. It is the single place where raw
  customer data becomes model input, used for training AND for serving, so
  the two can never drift apart. It works the same on one row or thousands.
- build_features is the training entry point: fit + transform in one call.

The schema is plain JSON-serialisable data and is saved with the model.
"""

from typing import Tuple

import pandas as pd


def _binary_mapping(values) -> dict:
    """
    Deterministic 0/1 mapping for a column with exactly two categories.

    The mappings are fixed so training and serving always agree.
    """
    valset = set(values)

    # Yes/No mapping (most common pattern in telecom data)
    if valset == {"Yes", "No"}:
        return {"No": 0, "Yes": 1}

    # Gender mapping (demographic feature)
    if valset == {"Male", "Female"}:
        return {"Female": 0, "Male": 1}

    # Any other 2-category feature: stable alphabetical ordering
    low, high = sorted(valset)
    return {low: 0, high: 1}


def fit_features(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """
    Learn the feature schema from training data.

    Args:
        df: Preprocessed training data (may include the target column).
        target_col: Name of the target column, excluded from the features.

    Returns:
        dict with:
          passthrough_columns: numeric/boolean columns used as they are
          binary_mappings: {column: {category: 0/1}} for 2-category columns
          categorical_levels: {column: sorted categories} for columns with
              more than 2 categories (the first one is the dropped baseline)
          columns: the final feature columns, in the order the model expects
    """
    # === Identify feature types ===
    # "str" is included explicitly: pandas >=3 defaults CSV text columns to
    # StringDtype, and relying on "object" alone to match them is deprecated
    obj_cols = [
        c
        for c in df.select_dtypes(include=["object", "str"]).columns
        if c != target_col
    ]
    feature_cols = [c for c in df.columns if c != target_col]
    passthrough = [c for c in feature_cols if c not in obj_cols]

    # === Split categorical columns by number of categories ===
    # Binary (2 values) -> 0/1 mapping; multi-category (>2) -> one-hot
    binary_mappings = {}
    categorical_levels = {}
    for c in obj_cols:
        categories = sorted(df[c].dropna().astype(str).unique())
        if len(categories) < 2:
            raise ValueError(
                f"Column '{c}' has {len(categories)} categories; "
                "need at least 2 to encode it"
            )
        if len(categories) == 2:
            binary_mappings[c] = _binary_mapping(categories)
        else:
            categorical_levels[c] = categories

    # === Final column order ===
    # Same layout pandas get_dummies(drop_first=True) produces: original
    # columns (multi-category ones removed) first, then the dummy columns.
    # drop_first prevents multicollinearity; the first category is the
    # baseline.
    dummies = [
        f"{c}_{level}"
        for c, levels in categorical_levels.items()
        for level in levels[1:]
    ]
    kept = [c for c in feature_cols if c not in categorical_levels]
    columns = kept + dummies

    return {
        "passthrough_columns": passthrough,
        "binary_mappings": binary_mappings,
        "categorical_levels": categorical_levels,
        "columns": columns,
    }


def transform_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Turn preprocessed customer data into model-ready features.

    Works identically for a whole training set and for a single customer.
    Raises a clear error for missing columns or categories not seen in
    training, instead of silently producing wrong features.

    Args:
        df: Preprocessed data (one row or many).
        schema: The dict returned by fit_features.

    Returns:
        DataFrame with exactly schema["columns"], in that order, all numeric.
    """
    required = (
        schema["passthrough_columns"]
        + list(schema["binary_mappings"])
        + list(schema["categorical_levels"])
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = {}

    # Numeric/boolean columns pass through (booleans become 0/1)
    for c in schema["passthrough_columns"]:
        series = df[c]
        is_bool = pd.api.types.is_bool_dtype(series)
        out[c] = series.astype(int) if is_bool else series

    # Binary columns: fixed category -> 0/1 mapping
    for c, mapping in schema["binary_mappings"].items():
        mapped = df[c].astype(str).map(mapping)
        bad = mapped.isna()
        if bad.any():
            raise ValueError(
                f"Column '{c}' must be one of {sorted(mapping)}, got "
                f"{sorted(set(df.loc[bad, c].astype(str)))}"
            )
        out[c] = mapped.astype(int)

    # Multi-category columns: one-hot with the training categories, so a
    # single row still produces the full set of columns. The baseline
    # category (first level) is all zeros; a missing value is also all zeros.
    for c, levels in schema["categorical_levels"].items():
        values = df[c].astype(str)
        unseen = set(df[c].dropna().astype(str)) - set(levels)
        if unseen:
            raise ValueError(
                f"Column '{c}' must be one of {levels}, got {sorted(unseen)}"
            )
        for level in levels[1:]:
            out[f"{c}_{level}"] = (values == level).astype(int)

    return pd.DataFrame(out, index=df.index)[schema["columns"]]


def build_features(
    df: pd.DataFrame, target_col: str = "Churn"
) -> Tuple[pd.DataFrame, dict]:
    """
    Training entry point: learn the schema from df and apply it.

    The schema must be saved with the model and used by transform_features at
    serving time, so training and serving use identical transformations.

    Returns:
        (features DataFrame with the target column appended, schema dict)
    """
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    schema = fit_features(df, target_col)
    print(
        f"   🔢 Binary features: {len(schema['binary_mappings'])} | "
        f"Multi-category features: {len(schema['categorical_levels'])}"
    )
    print(f"      Binary: {list(schema['binary_mappings'])}")
    print(f"      Multi-category: {list(schema['categorical_levels'])}")

    features = transform_features(df, schema)
    if target_col in df.columns:
        features[target_col] = df[target_col]

    print(f"✅ Feature engineering complete: {len(schema['columns'])} features")
    return features, schema
