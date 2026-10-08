from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, threshold: float):
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Decision threshold applied to predict_proba, rather than
            sklearn's default 0.5.

    Returns:
        dict: accuracy, precision, recall, f1 and roc_auc on the test data.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return {
        "accuracy": accuracy_score(y_test, preds),
        # precision / recall / f1 are for the churn class (label 1)
        "precision": precision_score(y_test, preds, pos_label=1),
        "recall": recall_score(y_test, preds, pos_label=1),
        "f1": f1_score(y_test, preds, pos_label=1),
        "roc_auc": roc_auc_score(y_test, proba),  # threshold-independent
    }
