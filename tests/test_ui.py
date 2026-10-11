"""
Tests for the Gradio UI in app/app.py.

The UI only talks to the API over HTTP, so these tests point its HTTP calls
at the real FastAPI app (through a test client) and check the whole path:
form values -> request -> model -> text shown to the user.

Run from the project root:  pytest tests/test_ui.py -v
"""

import os
import sys

import pytest
import requests
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from app import app as ui  # noqa: E402
from app.main import app as api  # noqa: E402
from app.schemas import (  # noqa: E402
    EXAMPLE_CUSTOMER,
    INTERNET_ADD_ONS,
    CustomerData,
)

API = "http://testserver"


@pytest.fixture(scope="module")
def client():
    with TestClient(api, base_url=API) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def ui_talks_to_the_api(client, monkeypatch):
    """Send the UI's HTTP calls to the real API app instead of the network."""
    monkeypatch.setattr(ui, "API_URL", API)
    monkeypatch.setattr(
        ui.requests, "get", lambda url, timeout: client.get(url)
    )
    monkeypatch.setattr(
        ui.requests,
        "post",
        lambda url, json, timeout: client.post(url, json=json),
    )


def form_values(**changes):
    """The form's values, in the order of the UI's input components."""
    c = {**EXAMPLE_CUSTOMER, **changes}
    return [
        c["gender"],
        "Yes" if c["SeniorCitizen"] else "No",
        c["Partner"],
        c["Dependents"],
        c["tenure"],
        c["Contract"],
        c["PaperlessBilling"],
        c["PaymentMethod"],
        c["MonthlyCharges"],
        c["TotalCharges"],
        c["PhoneService"],
        c["MultipleLines"],
        c["InternetService"],
        *[c[name] for name in INTERNET_ADD_ONS],
    ]


# === Building the request ===


def test_form_values_become_the_exact_request_the_api_expects():
    payload = ui.build_payload(*form_values())
    assert payload == EXAMPLE_CUSTOMER
    CustomerData(**payload)  # also valid for the API's own schema


def test_senior_citizen_is_sent_as_a_number():
    payload = ui.build_payload(*form_values(SeniorCitizen=1))
    assert payload["SeniorCitizen"] == 1


# === Showing the answer ===


def test_high_risk_customer_is_flagged():
    high_risk = form_values(
        tenure=1,
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        MonthlyCharges=85.0,
        TotalCharges=85.0,
    )
    assert "Likely to churn" in ui.predict(*high_risk)


def test_low_risk_customer_is_not_flagged():
    low_risk = form_values(
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
    assert "Not likely to churn" in ui.predict(*low_risk)


def test_the_ui_decision_matches_the_api(client):
    direct = client.post("/predict", json=EXAMPLE_CUSTOMER).json()
    shown = ui.predict(*form_values())
    assert f"{direct['churn_score']:.2f}" in shown
    assert f"{direct['threshold']:.2f}" in shown


# === Errors are readable ===


def test_api_validation_errors_are_listed_for_the_user():
    # Bypasses the form's guard rails to check the message formatting
    text = ui.predict(*form_values(MonthlyCharges=500))
    assert "Could not score" in text
    assert "MonthlyCharges" in text


def test_unreachable_api_gives_a_helpful_message(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(ui.requests, "post", refuse)
    text = ui.predict(*form_values())
    assert "Cannot reach the prediction API" in text
    assert "Traceback" not in text


# === Model details ===


def test_model_details_show_the_served_metrics(client):
    info = client.get("/model-info").json()
    text = ui.describe_model()
    assert info["run_id"] in text
    assert f"{info['test_metrics']['recall']:.0%}" in text


# === The form cannot create contradictory services ===


def test_no_phone_service_forces_the_matching_multiple_lines():
    update = ui.update_lines("No", "Yes")
    assert update.value == "No phone service"
    assert update.choices == [("No phone service", "No phone service")]


def test_phone_service_restores_the_yes_no_choice():
    assert ui.update_lines("Yes", "No phone service").value == "No"
    assert ui.update_lines("Yes", "Yes").value == "Yes"


def test_no_internet_forces_every_add_on():
    updates = ui.update_add_ons("No", *["Yes"] * len(INTERNET_ADD_ONS))
    assert all(u.value == "No internet service" for u in updates)


def test_internet_service_keeps_existing_add_on_answers():
    current = ["Yes", "No", "Yes", "No", "No internet service", "Yes"]
    updates = ui.update_add_ons("DSL", *current)
    expected = ["Yes", "No", "Yes", "No", "No", "Yes"]
    assert [u.value for u in updates] == expected


@pytest.mark.parametrize("phone", ["Yes", "No"])
@pytest.mark.parametrize("internet", ["DSL", "Fiber optic", "No"])
def test_every_form_state_is_accepted_by_the_api(phone, internet):
    """Whatever the guard rails produce must pass the API's cross-checks."""
    lines = ui.update_lines(phone, "No").value
    add_ons = [u.value for u in ui.update_add_ons(internet, *["No"] * 6)]
    values = form_values(
        PhoneService=phone,
        MultipleLines=lines,
        InternetService=internet,
        **dict(zip(INTERNET_ADD_ONS, add_ons)),
    )
    assert "Could not score" not in ui.predict(*values)


def test_total_charges_suggestion():
    assert ui.estimate_total(12, 50.5) == 606.0
    assert ui.estimate_total(None, 50) == 0


# === The page itself ===


def test_the_page_builds():
    assert ui.build_ui() is not None


def test_dropdown_options_come_from_the_api_schema():
    assert set(ui.options("Contract")) == {
        "Month-to-month",
        "One year",
        "Two year",
    }
