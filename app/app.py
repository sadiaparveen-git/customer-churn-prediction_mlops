"""
Gradio web UI for the churn model.

The page collects a customer's details and shows the model's decision. It
does not load the model itself: it sends the form to the API (POST /predict)
over HTTP, so the decision threshold and the validation rules live in one
place (app/main.py and app/schemas.py).

Start the API first, then the UI, both from the project root:

    uvicorn app.main:app
    python -m app.app

The API address is read from the API_URL environment variable
(default http://127.0.0.1:8000).
"""

import os
from typing import get_args

import gradio as gr
import requests

from app.schemas import EXAMPLE_CUSTOMER, INTERNET_ADD_ONS, CustomerData

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 10  # seconds

NO_PHONE = "No phone service"
NO_INTERNET = "No internet service"


def options(field):
    """Allowed values of a field, read from the API's own schema."""
    return list(get_args(CustomerData.model_fields[field].annotation))


# === Talking to the API ===


def call_api(path, payload=None):
    """
    Call the API and return (json, error_message).

    Exactly one of the two is None. Connection problems and validation errors
    become readable messages instead of stack traces.
    """
    url = f"{API_URL}{path}"
    try:
        if payload is None:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.post(
                url, json=payload, timeout=REQUEST_TIMEOUT
            )
    except requests.exceptions.RequestException:
        return None, (
            f"Cannot reach the prediction API at {API_URL}. "
            "Start it with `uvicorn app.main:app`."
        )

    if response.status_code == 200:
        return response.json(), None
    if response.status_code == 422:
        return None, "The API rejected the input:\n" + format_validation(
            response.json()
        )
    return None, f"The API returned an error (HTTP {response.status_code})."


def format_validation(body):
    """Turn the API's 422 body into one bullet per problem."""
    lines = []
    for item in body.get("detail", []):
        field = ".".join(str(part) for part in item["loc"][1:])
        message = item["msg"].removeprefix("Value error, ")
        lines.append(f"- {field + ': ' if field else ''}{message}")
    return "\n".join(lines) or "- unknown validation error"


# === Turning the form into a request, and the answer into text ===


def build_payload(
    gender,
    senior,
    partner,
    dependents,
    tenure,
    contract,
    paperless,
    payment,
    monthly,
    total,
    phone,
    lines,
    internet,
    *add_ons,
):
    """Arrange the form values as the JSON body the API expects."""
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    payload.update(zip(INTERNET_ADD_ONS, add_ons))
    return payload


def format_result(result):
    """Markdown for the model's decision."""
    score = result["churn_score"]
    threshold = result["threshold"]
    if result["likely_to_churn"]:
        headline = "## 🔴 Likely to churn"
        advice = "Consider a retention offer for this customer."
    else:
        headline = "## 🟢 Not likely to churn"
        advice = "No action needed."
    return (
        f"{headline}\n\n{advice}\n\n"
        f"Risk score **{score:.2f}** (a customer is flagged at "
        f"**{threshold:.2f}** or above). The score ranks customers by risk; "
        "it is not a percentage chance."
    )


def predict(*form_values):
    """Button handler: validate and score the customer through the API."""
    result, error = call_api("/predict", build_payload(*form_values))
    if error:
        return f"## ⚠️ Could not score this customer\n\n{error}"
    return format_result(result)


def describe_model():
    """Markdown with the served model's headline numbers."""
    info, error = call_api("/model-info")
    if error:
        return f"Model details unavailable.\n\n{error}"
    metrics = info["test_metrics"]
    return (
        f"Trained on {info['trained_at'][:10]}, scored on customers the "
        "model never saw during training:\n\n"
        f"- Catches **{metrics['recall']:.0%}** of the customers who leave "
        "(recall)\n"
        f"- **{metrics['precision']:.0%}** of the customers it flags really "
        "leave (precision)\n"
        f"- Decision threshold: {info['threshold']:.3f}\n"
        f"- Run ID: `{info['run_id']}`"
    )


# === Keeping the form consistent ===
# The API rejects contradictory services (for example multiple phone lines
# without a phone). The form prevents them instead, so the user cannot make
# that mistake.


def update_lines(phone, current):
    if phone == "No":
        return gr.Dropdown(choices=[NO_PHONE], value=NO_PHONE)
    value = current if current in ("Yes", "No") else "No"
    return gr.Dropdown(choices=["Yes", "No"], value=value)


def update_add_ons(internet, *current):
    updates = []
    for value in current:
        if internet == "No":
            updates.append(
                gr.Dropdown(choices=[NO_INTERNET], value=NO_INTERNET)
            )
        else:
            value = value if value in ("Yes", "No") else "No"
            updates.append(gr.Dropdown(choices=["Yes", "No"], value=value))
    return updates


def estimate_total(tenure, monthly):
    """Suggested total charges: months as a customer x monthly bill."""
    return round(float(tenure or 0) * float(monthly or 0), 2)


# === The page ===


def build_ui():
    example = EXAMPLE_CUSTOMER
    with gr.Blocks(title="Customer Churn Prediction") as demo:
        gr.Markdown(
            "# Customer Churn Prediction\n"
            "Enter a telecom customer's details to see whether they are "
            "likely to leave."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Customer")
                gender = gr.Radio(
                    options("gender"), value=example["gender"], label="Gender"
                )
                senior = gr.Radio(
                    ["No", "Yes"], value="No", label="Senior citizen"
                )
                partner = gr.Radio(
                    ["Yes", "No"],
                    value=example["Partner"],
                    label="Has a partner",
                )
                dependents = gr.Radio(
                    ["Yes", "No"],
                    value=example["Dependents"],
                    label="Has dependents",
                )

                gr.Markdown("### Account")
                tenure = gr.Slider(
                    0, 72, value=example["tenure"], step=1,
                    label="Tenure (months as a customer)",
                )
                contract = gr.Dropdown(
                    options("Contract"),
                    value="Month-to-month",
                    label="Contract",
                )
                paperless = gr.Radio(
                    ["Yes", "No"],
                    value=example["PaperlessBilling"],
                    label="Paperless billing",
                )
                payment = gr.Dropdown(
                    options("PaymentMethod"),
                    value=example["PaymentMethod"],
                    label="Payment method",
                )
                monthly = gr.Number(
                    value=example["MonthlyCharges"], minimum=0, maximum=200,
                    label="Monthly charges ($)",
                )
                total = gr.Number(
                    value=example["TotalCharges"], minimum=0,
                    label="Total charges so far ($)",
                    info="Filled in as tenure x monthly; edit if known.",
                )

            with gr.Column():
                gr.Markdown("### Services")
                phone = gr.Radio(
                    ["Yes", "No"], value=example["PhoneService"],
                    label="Phone service",
                )
                lines = gr.Dropdown(
                    [NO_PHONE], value=NO_PHONE, label="Multiple lines"
                )
                internet = gr.Dropdown(
                    options("InternetService"),
                    value=example["InternetService"],
                    label="Internet service",
                )
                add_ons = [
                    gr.Dropdown(
                        ["Yes", "No"], value=example[name], label=label
                    )
                    for name, label in zip(
                        INTERNET_ADD_ONS,
                        [
                            "Online security",
                            "Online backup",
                            "Device protection",
                            "Tech support",
                            "Streaming TV",
                            "Streaming movies",
                        ],
                    )
                ]

                submit = gr.Button("Predict churn", variant="primary")
                output = gr.Markdown()

        with gr.Accordion("About the model", open=False):
            about = gr.Markdown()

        # Keep the form consistent
        phone.change(update_lines, [phone, lines], lines)
        internet.change(update_add_ons, [internet, *add_ons], add_ons)
        for field in (tenure, monthly):
            field.change(estimate_total, [tenure, monthly], total)

        form = [
            gender, senior, partner, dependents, tenure, contract, paperless,
            payment, monthly, total, phone, lines, internet, *add_ons,
        ]
        submit.click(predict, form, output)
        demo.load(describe_model, None, about)
    return demo


if __name__ == "__main__":
    build_ui().launch()
