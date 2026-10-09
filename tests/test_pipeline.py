"""
Tests for scripts/run_pipeline.py and scripts/export_model.py.

The raw dataset is gitignored, so these tests build a small synthetic
Telco-style CSV and run the real pipeline against it. All outputs
(processed data, artifacts, MLflow runs) go to a temp directory, so the
project's own artifacts/, data/processed/ and mlruns/ are never touched.

Run from the project root:  pytest tests/test_pipeline.py -v
"""

import argparse
import importlib.util
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENT = "test-experiment"


def make_telco_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same schema/value sets as the raw Telco CSV."""
    rng = np.random.default_rng(seed)

    def pick(options):
        return rng.choice(options, size=n)

    internet = pick(["DSL", "Fiber optic", "No"])
    phone = pick(["Yes", "No"])

    def internet_addon():
        # Telco encodes "no internet" as its own category on add-on columns
        return np.where(
            internet == "No", "No internet service", pick(["Yes", "No"])
        )

    tenure = rng.integers(1, 73, size=n)
    monthly = np.round(rng.uniform(20, 110, size=n), 2)
    total = pd.Series(np.round(monthly * tenure, 2).astype(str))
    total.iloc[[3, 77]] = " "  # raw file has blank TotalCharges rows

    return pd.DataFrame(
        {
            "customerID": [f"ID-{i:04d}" for i in range(n)],
            "gender": pick(["Male", "Female"]),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": pick(["Yes", "No"]),
            "Dependents": pick(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "Yes", pick(["Yes", "No"]), "No phone service"
            ),
            "InternetService": internet,
            "OnlineSecurity": internet_addon(),
            "OnlineBackup": internet_addon(),
            "DeviceProtection": internet_addon(),
            "TechSupport": internet_addon(),
            "StreamingTV": internet_addon(),
            "StreamingMovies": internet_addon(),
            "Contract": pick(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": pick(["Yes", "No"]),
            "PaymentMethod": pick(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(rng.random(n) < 0.27, "Yes", "No"),
        }
    )


@pytest.fixture(scope="module")
def pipeline_module():
    """Import scripts/run_pipeline.py (it isn't a package) once per module."""
    path = os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py")
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def export_module():
    """Import scripts/export_model.py once per module."""
    path = os.path.join(PROJECT_ROOT, "scripts", "export_model.py")
    spec = importlib.util.spec_from_file_location("export_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline(pipeline_module, tmp_path, monkeypatch):
    """The pipeline module with its project root redirected to tmp_path."""
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", str(tmp_path))
    return pipeline_module


def make_args(tmp_path, df: pd.DataFrame = None, **overrides):
    """Write df (default: valid synthetic data) to CSV and build CLI args."""
    csv_path = tmp_path / "telco.csv"
    (make_telco_df() if df is None else df).to_csv(csv_path, index=False)
    args = {
        "input": str(csv_path),
        "target": "Churn",
        "threshold_min": 0.25,
        "threshold_max": 0.5,
        "min_recall": 0.0,  # synthetic labels are random; test wiring only
        "test_size": 0.2,
        "experiment": EXPERIMENT,
        "mlflow_uri": f"file://{tmp_path}/mlruns",
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def latest_run():
    """The most recent MLflow run in the test experiment."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT], order_by=["start_time DESC"]
    )
    return runs.iloc[0]


def test_pipeline_end_to_end(pipeline, tmp_path):
    """Happy path: every stage runs and produces its output."""
    pipeline.main(make_args(tmp_path))

    # Preprocessed data is saved: no ID column, numeric target, no NaNs
    processed = pd.read_csv(
        tmp_path / "data" / "processed" / "telco_churn_processed.csv"
    )
    assert "customerID" not in processed.columns
    assert set(processed["Churn"].unique()) <= {0, 1}
    assert not processed.isna().any().any()

    # Serving artifacts are saved and consistent with each other
    artifacts = tmp_path / "artifacts"
    columns = json.loads((artifacts / "feature_columns.json").read_text())
    assert "Churn" not in columns
    model = joblib.load(artifacts / "model.joblib")
    assert list(model.feature_names_in_) == columns

    # The MLflow run recorded validation, tuned params and test metrics
    run = latest_run()
    assert run["metrics.data_quality_pass"] == 1.0
    # the tuned threshold must land inside the searched range
    assert 0.25 <= float(run["params.threshold"]) <= 0.5
    assert "params.n_estimators" in run  # set from the tuned params
    assert float(run["params.scale_pos_weight"]) >= 1.0  # tuned class weight
    for metric in ("recall", "precision", "f1", "roc_auc"):
        assert 0.0 <= run[f"metrics.test_{metric}"] <= 1.0


def test_pipeline_stops_on_invalid_data(pipeline, tmp_path):
    """Bad data must fail validation before anything is written."""
    df = make_telco_df()
    df.loc[0, "gender"] = "Other"  # not in {Male, Female}
    df.loc[1, "tenure"] = -5  # negative tenure

    with pytest.raises(ValueError, match="Data quality check failed"):
        pipeline.main(make_args(tmp_path, df=df))

    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "artifacts").exists()
    assert latest_run()["metrics.data_quality_pass"] == 0.0


def test_pipeline_rejects_missing_target(pipeline, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        pipeline.main(make_args(tmp_path, target="DoesNotExist"))


def test_pipeline_rejects_missing_input_file(pipeline, tmp_path):
    args = make_args(tmp_path, input=str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        pipeline.main(args)


def export_args(tmp_path, out_dir, **overrides):
    args = {
        "experiment": EXPERIMENT,
        "run_id": None,
        "output_dir": str(out_dir),
        "mlflow_uri": f"file://{tmp_path}/mlruns",
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_export_creates_serving_bundle(pipeline, export_module, tmp_path):
    """Pipeline -> export gives a small bundle that matches the run."""
    pipeline.main(make_args(tmp_path))
    out = tmp_path / "bundle"
    export_module.main(export_args(tmp_path, out))

    assert sorted(p.name for p in out.iterdir()) == [
        "feature_columns.json",
        "model.ubj",
        "model_info.json",
    ]
    columns = json.loads((out / "feature_columns.json").read_text())
    info_text = (out / "model_info.json").read_text()
    info = json.loads(info_text)

    # model_info describes the run that was exported
    assert info["run_id"] == latest_run()["run_id"]
    assert 0.25 <= info["threshold"] <= 0.5
    assert info["n_features"] == len(columns)
    assert str(tmp_path) not in info_text  # no local paths leak out

    # The exported model predicts exactly like the one the pipeline saved
    exported = XGBClassifier()
    exported.load_model(str(out / "model.ubj"))
    original = joblib.load(tmp_path / "artifacts" / "model.joblib")
    X = pd.DataFrame(
        np.random.default_rng(0).random((50, len(columns))), columns=columns
    )
    assert np.allclose(
        exported.predict_proba(X)[:, 1], original.predict_proba(X)[:, 1]
    )


def test_export_rejects_unknown_experiment(export_module, tmp_path):
    args = export_args(tmp_path, tmp_path / "bundle", experiment="nope")
    with pytest.raises(ValueError, match="not found"):
        export_module.main(args)
