import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def tune_model(X_train, y_train, threshold: float):
    """
    Tunes an XGBoost model using Optuna.

    Pass only the training portion of the data. It is split again internally
    into fit/validation sets, so the final test set is never seen here.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        threshold (float): Decision threshold the model is deployed at.

    Returns:
        dict: Best hyperparameters found.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit)

        # Score at the same decision threshold the model is deployed at,
        # not sklearn's default 0.5 (what cross_val_score's "recall"
        # scoring would otherwise silently use)
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= threshold).astype(int)
        return recall_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params
