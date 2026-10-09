"""
Tests for the FastAPI service in app/main.py.

They run against the committed model/ bundle, so they also check that the
request validation rules match the model that is actually shipped.

Run from the project root:  pytest tests/test_api.py -v
"""

import json
import os
import sys
from typing import get_args

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from app.main import app  # noqa: E402
from app.schemas import EXAMPLE_CUSTOMER, CustomerData  # noqa: E402
from src.serving.inference import ChurnPredictor  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Using the client as a context manager runs the startup (model loading)
    with TestClient(app) as test_client:
        yield test_client


def payload(**changes):
    """A valid customer, with some fields changed."""
    return {**EXAMPLE_CUSTOMER, **changes}


# A long-standing customer on a two-year contract with every add-on
LOW_RISK = payload(
    tenure=60,
    Contract="Two year",
    PaperlessBilling="No",
    PaymentMethod="Credit card (automatic)",
    PhoneService="Yes",
    MultipleLines="Yes",
    OnlineSecurity="Yes",
    OnlineBackup="Yes",
    DeviceProtection="Yes",
    TechSupport="Yes",
    MonthlyCharges=80.0,
    TotalCharges=4800.0,
)

# A brand-new fiber customer, month-to-month, paying by electronic check
HIGH_RISK = payload(
    tenure=1,
    InternetService="Fiber optic",
    OnlineSecurity="No",
    OnlineBackup="No",
    DeviceProtection="No",
    TechSupport="No",
    MonthlyCharges=85.0,
    TotalCharges=85.0,
)


# === Service endpoints ===


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_the_served_model(client):
    response = client.get("/model-info")
    assert response.status_code == 200

    body = response.json()
    with open(os.path.join(PROJECT_ROOT, "model", "model_info.json")) as f:
        info = json.load(f)
    assert body["run_id"] == info["run_id"]
    assert body["threshold"] == info["threshold"]
    assert 0 < body["test_metrics"]["recall"] <= 1


# === Predictions ===


def test_predict_returns_a_decision(client):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "churn_score",
        "likely_to_churn",
        "label",
        "threshold",
    }
    assert 0 <= body["churn_score"] <= 1
    expected = body["churn_score"] >= body["threshold"]
    assert body["likely_to_churn"] == expected


def test_api_gives_the_same_answer_as_the_predictor(client):
    """The API adds validation only; it must not change the prediction."""
    direct = ChurnPredictor().predict(EXAMPLE_CUSTOMER)
    assert client.post("/predict", json=EXAMPLE_CUSTOMER).json() == direct


def test_risk_ordering_is_sensible(client):
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()
    assert low["churn_score"] < high["churn_score"]


# === Validation: bad input is rejected with a 422, never scored ===


def rejected(client, body, field=None, text=None):
    """Assert a 422, optionally naming the field or message text."""
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    detail = json.dumps(response.json()["detail"])
    if field:
        assert field in detail
    if text:
        assert text in detail


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"Contract": "Weekly"}, "Contract"),
        ({"gender": "Other"}, "gender"),
        ({"PaymentMethod": "Cash"}, "PaymentMethod"),
        ({"SeniorCitizen": 2}, "SeniorCitizen"),
        ({"tenure": -1}, "tenure"),
        ({"tenure": 121}, "tenure"),
        ({"tenure": 5.5}, "tenure"),
        ({"tenure": "abc"}, "tenure"),
        ({"MonthlyCharges": -10}, "MonthlyCharges"),
        ({"MonthlyCharges": 500}, "MonthlyCharges"),
        ({"TotalCharges": -1}, "TotalCharges"),
    ],
)
def test_invalid_values_are_rejected(client, changes, field):
    rejected(client, payload(**changes), field=field)


def test_missing_field_is_rejected(client):
    body = payload()
    del body["Contract"]
    rejected(client, body, field="Contract")


def test_unknown_field_is_rejected(client):
    rejected(client, payload(customerID="7590-VHVEG"), text="customerID")


def test_phone_service_must_match_multiple_lines(client):
    rejected(
        client,
        payload(PhoneService="No", MultipleLines="Yes"),
        text="MultipleLines",
    )
    rejected(
        client,
        payload(PhoneService="Yes", MultipleLines="No phone service"),
        text="MultipleLines",
    )


def test_internet_service_must_match_add_ons(client):
    rejected(
        client,
        payload(InternetService="No"),  # add-ons still say Yes/No
        text="must be 'No internet service'",
    )
    rejected(
        client,
        payload(InternetService="DSL", TechSupport="No internet service"),
        text="TechSupport",
    )


def test_no_internet_customer_is_accepted(client):
    body = payload(
        InternetService="No",
        OnlineSecurity="No internet service",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
        MonthlyCharges=20.0,
        TotalCharges=20.0,
    )
    assert client.post("/predict", json=body).status_code == 200


# === The validation rules must match the shipped model ===


def test_request_fields_match_the_model_schema():
    """If the model is retrained with different features, this fails."""
    with open(os.path.join(PROJECT_ROOT, "model", "feature_schema.json")) as f:
        schema = json.load(f)

    expected = (
        set(schema["passthrough_columns"])
        | set(schema["binary_mappings"])
        | set(schema["categorical_levels"])
    )
    assert set(CustomerData.model_fields) == expected


def test_allowed_values_match_the_model_schema():
    """Every category the API accepts is one the model was trained on."""
    with open(os.path.join(PROJECT_ROOT, "model", "feature_schema.json")) as f:
        schema = json.load(f)

    def allowed(field):
        return set(get_args(CustomerData.model_fields[field].annotation))

    for field, mapping in schema["binary_mappings"].items():
        assert allowed(field) == set(mapping), field
    for field, levels in schema["categorical_levels"].items():
        assert allowed(field) == set(levels), field
