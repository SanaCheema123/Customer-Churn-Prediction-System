"""
data_generator.py
=================
Generates a realistic synthetic customer churn dataset with
10 000 records mimicking a telecom company's customer base.
Run standalone or imported by the pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42


def generate_churn_dataset(n_samples: int = 10_000, save_path: str = "data/raw/churn_data.csv") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Demographics ────────────────────────────────────────────────
    gender          = rng.choice(["Male", "Female"], n_samples)
    senior_citizen  = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner         = rng.choice(["Yes", "No"], n_samples)
    dependents      = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # ── Contract & Services ─────────────────────────────────────────
    contract        = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    internet        = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22])
    phone_service   = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multi_lines     = np.where(phone_service == "No", "No phone service",
                               rng.choice(["Yes", "No"], n_samples))
    online_security = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))
    online_backup   = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))
    device_prot     = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))
    tech_support    = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))
    streaming_tv    = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))
    streaming_mov   = np.where(internet == "No", "No internet service",
                               rng.choice(["Yes", "No"], n_samples))

    # ── Billing ─────────────────────────────────────────────────────
    paperless       = rng.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment         = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )

    # ── Usage Metrics ───────────────────────────────────────────────
    tenure = rng.integers(1, 73, n_samples).astype(float)
    # Newer customers (short tenure) → higher churn probability
    base_monthly = np.where(internet == "Fiber optic", rng.uniform(70, 115, n_samples),
                   np.where(internet == "DSL", rng.uniform(25, 75, n_samples),
                            rng.uniform(20, 30, n_samples)))
    monthly_charges   = np.round(base_monthly, 2)
    total_charges     = np.round(monthly_charges * tenure + rng.normal(0, 20, n_samples), 2)
    total_charges     = np.clip(total_charges, 0, None)
    num_support_calls = rng.integers(0, 15, n_samples)
    num_products      = rng.integers(1, 7, n_samples)
    days_last_contact = rng.integers(1, 365, n_samples)
    avg_monthly_usage = np.round(rng.uniform(10, 300, n_samples), 1)
    contract_months   = np.where(contract == "Month-to-month", 0,
                        np.where(contract == "One year", rng.integers(1, 13, n_samples),
                                 rng.integers(1, 25, n_samples)))

    # ── Churn Label (realistic ~26% rate) ───────────────────────────
    churn_score = (
        0.40 * (tenure < 12).astype(float)
      + 0.25 * (contract == "Month-to-month").astype(float)
      + 0.15 * (internet == "Fiber optic").astype(float)
      + 0.10 * (num_support_calls > 5).astype(float)
      + 0.08 * (payment == "Electronic check").astype(float)
      + 0.05 * (online_security == "No").astype(float)
      - 0.10 * (contract == "Two year").astype(float)
      - 0.05 * (num_products > 3).astype(float)
      + rng.uniform(-0.10, 0.10, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.5)))
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    # ── Assemble DataFrame ──────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID":             [f"CUST-{i:06d}" for i in range(1, n_samples + 1)],
        "Gender":                 gender,
        "SeniorCitizen":          senior_citizen,
        "Partner":                partner,
        "Dependents":             dependents,
        "tenure":                 tenure,
        "PhoneService":           phone_service,
        "MultipleLines":          multi_lines,
        "InternetService":        internet,
        "OnlineSecurity":         online_security,
        "OnlineBackup":           online_backup,
        "DeviceProtection":       device_prot,
        "TechSupport":            tech_support,
        "StreamingTV":            streaming_tv,
        "StreamingMovies":        streaming_mov,
        "Contract":               contract,
        "PaperlessBilling":       paperless,
        "PaymentMethod":          payment,
        "MonthlyCharges":         monthly_charges,
        "TotalCharges":           total_charges,
        "NumSupportCalls":        num_support_calls,
        "NumProductsUsed":        num_products,
        "DaysSinceLastContact":   days_last_contact,
        "AvgMonthlyUsage":        avg_monthly_usage,
        "ContractMonthsRemaining":contract_months,
        "Churn":                  churn,
    })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataGenerator] Dataset saved → {save_path}  |  Shape: {df.shape}  |  Churn rate: {df['Churn'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_churn_dataset()
