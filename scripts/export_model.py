#!/usr/bin/env python3
"""
Export a trained model from MLflow as a small, self-contained serving bundle.

The bundle is everything an API needs to make predictions, and nothing else:

    model/
    ├── model.ubj              the XGBoost model (native format, no pickle)
    ├── feature_schema.json    how to build the features (encodings + order)
    └── model_info.json        threshold, run id, test metrics, library version

By default the most recent finished run in the experiment is exported. The
bundle is small enough to commit, so promoting a new model is just:
run the pipeline -> export -> commit.

Run from the project root:

    python scripts/export_model.py
    python scripts/export_model.py --run_id <run id> --output_dir model
"""

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import mlflow
import mlflow.artifacts
import mlflow.xgboost
import xgboost
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _number(value: str):
    """MLflow stores params as text; turn numeric ones back into numbers."""
    try:
        number = float(value)
    except ValueError:
        return value
    is_whole = number.is_integer() and "." not in value
    return int(number) if is_whole else number


def main(args):
    """Export one MLflow run's model into args.output_dir."""

    # === MLflow setup (same store the pipeline writes to) ===
    # MLflow 3.x raises on the plain filesystem backend unless allowed
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(
        args.mlflow_uri or f"file://{PROJECT_ROOT}/mlruns"
    )
    client = mlflow.MlflowClient()

    # === Pick the run: the one given, else the latest finished run ===
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"Experiment '{args.experiment}' not found")

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        finished = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not finished:
            raise ValueError(
                f"No finished runs in experiment '{args.experiment}'"
            )
        run = finished[0]
    run_id = run.info.run_id
    print(f"📦 Exporting run {run_id} ({run.info.run_name})")

    # === Find the model that run logged ===
    logged = [
        m
        for m in mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id], output_format="list"
        )
        if m.source_run_id == run_id
    ]
    if not logged:
        raise ValueError(f"Run {run_id} has no logged model")
    model_id = logged[0].model_id

    if "threshold" not in run.data.params:
        raise ValueError(f"Run {run_id} has no 'threshold' param")

    # === Write the bundle ===
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Model, saved in XGBoost's native format (not pickle)
    model = mlflow.xgboost.load_model(f"models:/{model_id}")
    model.save_model(os.path.join(args.output_dir, "model.ubj"))

    # 2) Feature schema (encodings + column order), saved with the run
    run_files = [a.path for a in client.list_artifacts(run_id)]
    if "feature_schema.json" not in run_files:
        raise ValueError(
            f"Run {run_id} has no feature_schema.json (it was trained "
            "before the schema was saved); re-run the pipeline"
        )
    with tempfile.TemporaryDirectory() as tmp:
        schema_file = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="feature_schema.json", dst_path=tmp
        )
        shutil.copy(
            schema_file, os.path.join(args.output_dir, "feature_schema.json")
        )

    # 3) Everything else a service needs to know about this model
    with open(os.path.join(args.output_dir, "feature_schema.json")) as f:
        feature_columns = json.load(f)["columns"]

    params = {k: _number(v) for k, v in run.data.params.items()}
    info = {
        "run_id": run_id,
        "run_name": run.info.run_name,
        "model_id": model_id,
        "experiment": args.experiment,
        "trained_at": datetime.fromtimestamp(
            run.info.start_time / 1000, tz=timezone.utc
        ).isoformat(),
        "threshold": params["threshold"],
        "n_features": len(feature_columns),
        "test_metrics": {
            k.removeprefix("test_"): v
            for k, v in run.data.metrics.items()
            if k.startswith("test_")
        },
        "params": params,
        "xgboost_version": xgboost.__version__,
    }
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # === Sanity check: the exported files load and agree with each other ===
    check = XGBClassifier()
    check.load_model(os.path.join(args.output_dir, "model.ubj"))
    if check.n_features_in_ != len(feature_columns):
        raise ValueError(
            f"Model expects {check.n_features_in_} features but "
            f"feature_schema.json lists {len(feature_columns)}"
        )

    print(f"✅ Exported to {args.output_dir}")
    print(
        f"   threshold: {info['threshold']:.3f} | "
        f"features: {len(feature_columns)}"
    )
    rounded = {k: round(v, 3) for k, v in info["test_metrics"].items()}
    print(f"   test metrics: {rounded}")


if __name__ == "__main__":
    # === Export configuration (CLI) ===
    p = argparse.ArgumentParser(
        description="Export a trained model as a small serving bundle"
    )
    p.add_argument("--experiment", type=str, default="Telco Churn - XGBoost")
    p.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="run to export; default is the latest finished run",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "model"),
        help="where to write the bundle",
    )
    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="override MLflow tracking URI, else uses <project_root>/mlruns",
    )

    main(p.parse_args())
