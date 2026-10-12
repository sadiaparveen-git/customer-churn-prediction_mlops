#!/usr/bin/env python3
"""
Build model/score_reference.json, which puts a model score in context.

The score itself ("0.12") means little to a business user. This script scores
the held-out test customers with the exported model and records how scores
relate to what really happened, so the API can say things like "riskier than
22% of customers" and "in testing, 17% of customers at this level left".

It does not retrain anything and does not change the pipeline. It rebuilds the
same test set the pipeline used (same data, same seeded split) and checks that
the model's recall and precision on it match the numbers logged at training
time. If they do not, the data and the model do not belong together and the
script stops instead of writing a misleading reference.

Run after exporting a model, from the project root:

    python scripts/build_score_reference.py
    python scripts/build_score_reference.py --model_dir model \\
        --input data/processed/telco_churn_processed.csv
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import transform_features  # noqa: E402
from src.serving.score_reference import (  # noqa: E402
    REFERENCE_FILE,
    build_reference,
    save_reference,
)

# Risk-band edges. "High" starts at the model's decision threshold, so the
# bands agree with the Likely / Not likely decision. The other two edges are
# chosen so every band holds enough customers; build_reference checks that.
LOW_MAX = 0.20
VERY_HIGH_MIN = 0.85


def score_test_set(model_dir: str, data_path: str):
    """Rebuild the pipeline's test set and score it with the exported model."""
    with open(os.path.join(model_dir, "model_info.json")) as f:
        info = json.load(f)
    with open(os.path.join(model_dir, "feature_schema.json")) as f:
        schema = json.load(f)
    params = info["params"]
    target = params["target"]

    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, "model.ubj"))

    # Same steps and the same seeded split as scripts/run_pipeline.py
    df = pd.read_csv(data_path)
    X = transform_features(df.drop(columns=[target]), schema)
    y = df[target]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        stratify=y,
        random_state=42,
    )
    scores = model.predict_proba(X_test)[:, 1]
    return info, scores, y_test.to_numpy()


def check_matches_training(info, scores, churned):
    """Stop if this test set is not the one the model was evaluated on."""
    decided = scores >= info["threshold"]
    found = {
        "recall": recall_score(churned, decided),
        "precision": precision_score(churned, decided),
    }
    for name, value in found.items():
        logged = info["test_metrics"][name]
        if abs(value - logged) > 1e-6:
            raise SystemExit(
                f"The rebuilt test set does not match the model: {name} is "
                f"{value:.4f} here but {logged:.4f} in model_info.json. "
                "The data file and the model bundle must come from the same "
                "pipeline run."
            )


def main(args):
    info, scores, churned = score_test_set(args.model_dir, args.input)
    check_matches_training(info, scores, churned)
    print(f"✅ Test set reproduced: {len(scores)} customers, metrics match")

    reference = build_reference(
        scores,
        churned,
        threshold=info["threshold"],
        low_max=args.low_max,
        very_high_min=args.very_high_min,
        run_id=info.get("run_id"),
    )

    output = os.path.join(args.model_dir, REFERENCE_FILE)
    save_reference(reference, output)

    print(f"📊 Overall churn rate: {reference['overall_churn_rate']:.1%}")
    for band in reference["bands"]:
        print(
            f"   {band['name']:<10} score {band['min']:.3f}-{band['max']:.3f}"
            f"  {band['customers']:>4} customers"
            f"  churn rate {band['churn_rate']:.1%}"
        )
    print(f"💾 Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the score reference for a model bundle."
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(PROJECT_ROOT, "model"),
        help="Model bundle folder (default: model/)",
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            PROJECT_ROOT, "data", "processed", "telco_churn_processed.csv"
        ),
        help="Processed data written by the pipeline",
    )
    parser.add_argument("--low_max", type=float, default=LOW_MAX)
    parser.add_argument("--very_high_min", type=float, default=VERY_HIGH_MIN)
    main(parser.parse_args())
