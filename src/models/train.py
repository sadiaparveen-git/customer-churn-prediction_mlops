import mlflow
import mlflow.data
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier

# Settings that stay the same regardless of tuning
FIXED_PARAMS = {"random_state": 42, "n_jobs": -1, "eval_metric": "logloss"}

# Fallback hyperparameters, used only when no tuned params are passed in
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def train_model(
    X_train, y_train, X_test, y_test, threshold: float, params: dict = None
):
    """
    Trains an XGBoost model and logs it to the active MLflow run.

    MLflow tracking (URI, experiment, run) is configured by the caller,
    see scripts/run_pipeline.py.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Held-out features, used for the logged metrics.
        y_test (pd.Series): Held-out target.
        threshold (float): Decision threshold applied to predict_proba.
        params (dict): Hyperparameters, e.g. best params from tune_model.
            Falls back to DEFAULT_PARAMS when not given.

    Returns:
        XGBClassifier: The trained model.
    """
    all_params = {**FIXED_PARAMS, **(params or DEFAULT_PARAMS)}

    # Churn is ~27% positive; handled via the lowered threshold below rather
    # than also weighting the minority class (stacking both just trades
    # precision for recall without improving F1, so pick one).
    model = XGBClassifier(**all_params)

    # Train model
    model.fit(X_train, y_train)

    # Threshold tuned below 0.5 to catch more churners, per EDA
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)

    # Log params, metrics, and model
    mlflow.log_params(all_params)
    mlflow.log_param("threshold", threshold)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", rec)
    mlflow.xgboost.log_model(model, name="model")

    # 🔑 Log dataset so it shows in MLflow UI
    train_ds = mlflow.data.from_pandas(
        pd.concat([X_train, y_train], axis=1), source="training_data"
    )
    mlflow.log_input(train_ds, context="training")

    print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")

    return model
