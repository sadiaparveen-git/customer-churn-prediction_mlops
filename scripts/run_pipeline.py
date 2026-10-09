#!/usr/bin/env python3
"""
End-to-end churn pipeline, run sequentially:
load -> validate -> preprocess -> feature engineering -> split
-> tune -> train (with best params) -> evaluate -> save serving artifacts

All run-level configuration (MLflow URI, experiment name, recall floor,
threshold search range, split size) lives here. The src/ modules only
contain the logic.

Run from the project root:

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py \\
        --input data/raw/Telcom-Customer-Churn.csv --min_recall 0.80
"""

import argparse
import json
import os
import sys

import joblib
import mlflow
from sklearn.model_selection import train_test_split

# === Fix import path for local modules ===
# Allows `from src...` imports when this script is run from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Local modules - one per pipeline stage
from src.data.load_data import load_data  # noqa: E402
from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.models.evaluate import evaluate_model  # noqa: E402
from src.models.train import train_model  # noqa: E402
from src.models.tune import tune_model  # noqa: E402
from src.utils.validate_data import validate_telco_data  # noqa: E402


def main(args):
    """Orchestrates the complete ML workflow inside a single MLflow run."""

    # === MLflow setup (single source of truth for tracking config) ===
    # MLflow 3.x deprecated the plain filesystem backend and raises on it
    # unless explicitly allowed, so opt in for the local ./mlruns store
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(
        args.mlflow_uri or f"file://{PROJECT_ROOT}/mlruns"
    )
    mlflow.set_experiment(args.experiment)  # created if it doesn't exist

    # Everything below is tracked under this one run
    with mlflow.start_run():
        # === Log run configuration ===
        # (the threshold and model hyperparameters are logged by train_model)
        mlflow.log_param("target", args.target)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("min_recall", args.min_recall)

        # === STAGE 1: Load raw data ===
        print("🔄 Loading data...")
        df = load_data(args.input)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        if args.target not in df.columns:
            raise ValueError(
                f"Target column '{args.target}' not found in data"
            )

        # === STAGE 2: Validate data quality (Great Expectations) ===
        # Must pass before any training happens; runs on the raw data
        print("🔍 Validating data quality...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            # Keep the failures with the run for debugging, then stop
            mlflow.log_text(
                json.dumps(failed, indent=2),
                artifact_file="failed_expectations.json",
            )
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")

        # === STAGE 3: Preprocess (cleaning, target -> 0/1) ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df, target_col=args.target)

        # Save the cleaned (not yet encoded) data for reproducibility and
        # debugging; data/processed/ is gitignored, so it stays local
        processed_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "telco_churn_processed.csv"
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed data saved to {processed_path} | {df.shape}")

        # === STAGE 4: Feature engineering (binary + one-hot encoding) ===
        # Also returns the feature schema, saved in stage 9 so a prediction
        # service can build the exact same features for a single customer
        print("🛠️  Building features...")
        df, schema = build_features(df, target_col=args.target)

        # === STAGE 5: Train/test split ===
        # The only split in the pipeline. The test set stays untouched until
        # evaluation, so tuning can't leak into the reported metrics.
        print("📊 Splitting data...")
        X = df.drop(columns=[args.target])
        y = df[args.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,  # keep the churn ratio in both sets
            random_state=42,  # reproducible split
        )
        print(f"✅ Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # === STAGE 6: Hyperparameter tuning (Optuna, train data only) ===
        # Maximises churn-class precision while keeping its recall >=
        # min_recall (higher recall wins at similar precision). Each trial
        # tunes the hyperparameters, the class weight and the threshold.
        print("🎛️  Tuning hyperparameters and threshold...")
        best_params, threshold = tune_model(
            X_train,
            y_train,
            min_recall=args.min_recall,
            threshold_range=(args.threshold_min, args.threshold_max),
        )

        # === STAGE 7: Train with the tuned hyperparameters ===
        print("🤖 Training model with best params...")
        model = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            threshold=threshold,
            params=best_params,
        )

        # === STAGE 8: Evaluate on the untouched test set ===
        print("📈 Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # === STAGE 9: Save serving artifacts (for the FastAPI app) ===
        # The feature schema holds the encodings and the exact column order,
        # so serving builds features identically to training
        print("💾 Saving serving artifacts...")
        artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        schema_path = os.path.join(artifacts_dir, "feature_schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        model_path = os.path.join(artifacts_dir, "model.joblib")
        joblib.dump(model, model_path)

        # Keep copies with the MLflow run too
        mlflow.log_artifact(schema_path)
        mlflow.log_artifact(model_path)
        print(
            f"✅ Saved feature schema ({len(schema['columns'])} features) "
            f"and model to {artifacts_dir}"
        )


if __name__ == "__main__":
    # === Run configuration (CLI) ===
    p = argparse.ArgumentParser(
        description="Run the churn pipeline: validate, tune, train, evaluate"
    )
    p.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data", "raw", "Telcom-Customer-Churn.csv"
        ),
        help="path to the raw CSV",
    )
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument(
        "--min_recall",
        type=float,
        default=0.80,
        help="minimum churn-class recall tuning must keep while maximising "
        "precision",
    )
    p.add_argument(
        "--threshold_min",
        type=float,
        default=0.25,
        help="lower bound of the decision-threshold search",
    )
    p.add_argument(
        "--threshold_max",
        type=float,
        default=0.5,
        help="upper bound of the decision-threshold search",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn - XGBoost")
    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="override MLflow tracking URI, else uses <project_root>/mlruns",
    )

    main(p.parse_args())
