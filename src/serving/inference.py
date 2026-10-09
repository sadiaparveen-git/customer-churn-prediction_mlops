"""
Churn inference: one raw customer record in, a churn decision out.

This is the only place prediction logic lives. The FastAPI app and the
Gradio/Streamlit UI should both call ChurnPredictor, so every entry point
gives the same answer.

It reuses the training code (preprocess_data and transform_features), so the
features a customer is scored on are built exactly as they were in training.
It deliberately does not import MLflow, which keeps a serving image small.

The model bundle (written by scripts/export_model.py) is a folder holding:
    model.ubj            the XGBoost model
    feature_schema.json  how to build the features
    model_info.json      the decision threshold and run details
"""

import json
import os

import pandas as pd
from xgboost import XGBClassifier

from src.data.preprocess import preprocess_data
from src.features.build_features import transform_features

# Bundle location: the MODEL_DIR environment variable if set (for example in
# a Docker image), otherwise the project's model/ folder
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "model"
)


class ChurnPredictor:
    """Loads a model bundle once and scores customers one at a time."""

    def __init__(self, model_dir: str = None):
        model_dir = (
            model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR
        )

        self.model = XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "model.ubj"))

        with open(os.path.join(model_dir, "feature_schema.json")) as f:
            self.schema = json.load(f)
        with open(os.path.join(model_dir, "model_info.json")) as f:
            self.info = json.load(f)

        # The threshold turns the model's score into a yes/no decision;
        # it was tuned together with the model, so it must be applied
        self.threshold = float(self.info["threshold"])

        if self.model.n_features_in_ != len(self.schema["columns"]):
            raise ValueError(
                f"Model expects {self.model.n_features_in_} features but the "
                f"schema describes {len(self.schema['columns'])}; the bundle "
                "files do not belong together"
            )

    def predict(self, customer: dict) -> dict:
        """
        Score one customer.

        Args:
            customer: Raw customer fields, as in the original dataset
                (gender, tenure, Contract, MonthlyCharges, ...). A
                customerID is ignored.

        Returns:
            dict with:
              churn_score: the model's score between 0 and 1. It is not a
                  calibrated probability (the model was trained with a class
                  weight), so do not show it to users as a percentage.
              likely_to_churn: True if the score reaches the threshold
              label: "Likely to churn" or "Not likely to churn"
              threshold: the decision threshold that was applied

        Raises:
            ValueError: a required field is missing, or a categorical field
                has a value the model was never trained on.
        """
        # Same steps as training: clean, then build features from the schema
        df = preprocess_data(pd.DataFrame([customer]))
        features = transform_features(df, self.schema)

        score = float(self.model.predict_proba(features)[0, 1])
        likely = score >= self.threshold

        return {
            "churn_score": score,
            "likely_to_churn": likely,
            "label": "Likely to churn" if likely else "Not likely to churn",
            "threshold": self.threshold,
        }
