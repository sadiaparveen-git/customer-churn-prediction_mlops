"""Synthetic Telco-style data for the tests (the real CSV is gitignored)."""

import numpy as np
import pandas as pd


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
