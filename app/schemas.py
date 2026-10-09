"""
Request and response models for the churn API.

Pydantic validates every request before it reaches the model, so a bad
request is answered with a clear 422 error instead of a wrong prediction:

- every categorical field accepts only the values seen in training
- numeric fields must be in a sensible range (same limits as the data
  quality checks in src/utils/validate_data.py)
- unknown fields are rejected, which catches typos like "Contrat"
- services must be consistent (no multiple phone lines without a phone)

The allowed values mirror model/feature_schema.json. A test checks that the
two stay in sync, so a retrained model with new categories cannot ship
unnoticed.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]

INTERNET_ADD_ONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class CustomerData(BaseModel):
    """One customer, with the same fields as the original dataset."""

    # extra="forbid": a misspelled or unknown field is an error, not ignored
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER}
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if a senior citizen")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(
        ge=0, le=200, allow_inf_nan=False, description="Current monthly bill"
    )
    TotalCharges: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Total billed so far (0 for a brand-new customer)",
    )

    @model_validator(mode="after")
    def check_services_are_consistent(self):
        """
        Reject contradictory service combinations.

        In the training data a customer without phone service always has
        MultipleLines "No phone service" (and only then), and a customer
        without internet always has "No internet service" on every add-on
        (and only then). Anything else is an input mistake.
        """
        has_phone = self.PhoneService == "Yes"
        if has_phone == (self.MultipleLines == "No phone service"):
            raise ValueError(
                "MultipleLines must be 'No phone service' exactly when "
                "PhoneService is 'No'"
            )

        has_internet = self.InternetService != "No"
        for name in INTERNET_ADD_ONS:
            if has_internet == (getattr(self, name) == "No internet service"):
                raise ValueError(
                    f"{name} must be 'No internet service' exactly when "
                    "InternetService is 'No'"
                )
        return self


class PredictionResponse(BaseModel):
    """The model's decision for one customer."""

    churn_score: float = Field(
        ge=0,
        le=1,
        description=(
            "Model score between 0 and 1. Not a calibrated probability, "
            "so show the label rather than a percentage."
        ),
    )
    likely_to_churn: bool = Field(description="True if score >= threshold")
    label: Literal["Likely to churn", "Not likely to churn"]
    threshold: float = Field(description="Decision threshold that was applied")


class ModelInfoResponse(BaseModel):
    """Which model is serving, and how it performed when it was trained."""

    run_id: str
    run_name: str
    trained_at: str
    threshold: float
    n_features: int
    test_metrics: dict[str, float]
    xgboost_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
