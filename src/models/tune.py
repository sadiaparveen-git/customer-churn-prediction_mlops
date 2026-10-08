import optuna
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42  # same seed as the data split and the models


def _select_best(candidates, min_recall, precision_tolerance):
    """
    Pick the winning trial from per-trial validation results.

    Rule: among trials with churn recall >= min_recall, take the best
    precision; then, among trials whose precision is within
    `precision_tolerance` of that best, take the one with the highest recall.
    If no trial reaches min_recall, fall back to the highest-recall trial.

    Args:
        candidates (list[dict]): one dict per trial with keys
            "params", "threshold", "precision", "recall".

    Returns:
        tuple: (chosen candidate dict, whether the recall floor was met)
    """
    feasible = [c for c in candidates if c["recall"] >= min_recall]
    if not feasible:
        return max(candidates, key=lambda c: c["recall"]), False

    best_precision = max(c["precision"] for c in feasible)
    similar = [
        c
        for c in feasible
        if c["precision"] >= best_precision - precision_tolerance
    ]
    return max(similar, key=lambda c: (c["recall"], c["precision"])), True


def tune_model(
    X_train,
    y_train,
    min_recall: float,
    threshold_range: tuple = (0.25, 0.5),
    precision_tolerance: float = 0.01,
):
    """
    Tunes an XGBoost model and its decision threshold using Optuna.

    Every trial searches the XGBoost hyperparameters, the class weight
    (scale_pos_weight) and the decision threshold. Goal: maximise class-1
    (churn) precision while keeping class-1 recall at or above `min_recall`;
    when recall can be raised at a similar precision, the higher recall wins
    (see _select_best). The search is seeded, so results are reproducible.

    Pass only the training portion of the data. It is split again internally
    into fit/validation sets, so the final test set is never seen here.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        min_recall (float): Minimum churn-class recall a trial must reach.
        threshold_range (tuple): (low, high) bounds for the threshold search.
        precision_tolerance (float): Precisions within this of the best
            count as "similar" when preferring higher recall.

    Returns:
        tuple: (best XGBoost hyperparameters dict, best decision threshold)
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    # Negatives per positive; the class weight is searched up to 2x this
    class_ratio = (y_fit == 0).sum() / (y_fit == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 1.0, 2 * class_ratio
            ),
            "random_state": SEED,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        threshold = trial.suggest_float("threshold", *threshold_range)

        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit)

        # Score the churn class (label 1) at this trial's threshold
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= threshold).astype(int)
        precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
        recall = recall_score(y_val, preds, pos_label=1)

        # Kept so the winner can be chosen with the tie-break rule afterwards
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)

        # Below the recall floor: return a negative score (how far short it
        # fell) so every such trial ranks below any trial that meets the
        # floor, while still steering the search toward it
        if recall < min_recall:
            return recall - min_recall

        # Floor met: rank by precision
        return precision

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=20)

    candidates = [
        {
            "params": {k: v for k, v in t.params.items() if k != "threshold"},
            "threshold": t.params["threshold"],
            "precision": t.user_attrs["precision"],
            "recall": t.user_attrs["recall"],
        }
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best, floor_met = _select_best(candidates, min_recall, precision_tolerance)

    if not floor_met:
        print(
            f"⚠️  No trial reached recall >= {min_recall}; using the "
            "highest-recall one. Consider more trials or a lower min_recall."
        )
    print("Best Params:", best["params"])
    print(
        f"Best threshold: {best['threshold']:.3f} | validation churn "
        f"precision: {best['precision']:.3f}, recall: {best['recall']:.3f}"
    )
    return best["params"], best["threshold"]
