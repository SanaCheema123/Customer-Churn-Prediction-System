"""
predict.py
==========
Prediction utilities used by both the API and CLI.
Supports single-record and batch CSV prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.feature_engineering import engineer_features
from src.data_preprocessing  import DataPreprocessor


MODELS_DIR = Path("models")


def load_artifacts(model_name: str = None):
    """Load preprocessor + best model from disk."""
    preprocessor = DataPreprocessor.load(MODELS_DIR / "preprocessor.joblib")

    if model_name is None:
        lb_path = MODELS_DIR / "leaderboard.csv"
        if lb_path.exists():
            model_name = pd.read_csv(lb_path).iloc[0]["Model"]
        else:
            model_name = "XGBoost"

    path = MODELS_DIR / f"model_{model_name.lower().replace(' ', '_')}.joblib"
    model = joblib.load(path)
    print(f"[Predict] Loaded model: {model_name}")
    return model, preprocessor, model_name


def predict_single(record: dict, model=None, preprocessor=None, threshold: float = 0.50) -> dict:
    """Predict churn for a single customer record (dict)."""
    if model is None or preprocessor is None:
        model, preprocessor, _ = load_artifacts()

    df = pd.DataFrame([record])
    df = engineer_features(df)
    X, _ = preprocessor.transform(df)

    prob   = float(model.predict_proba(X)[0, 1])
    churn  = int(prob >= threshold)
    risk   = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  churn,
        "risk_level":        risk,
        "threshold_used":    threshold,
    }


def predict_batch(csv_path: str, output_path: str = "reports/batch_predictions.csv",
                  threshold: float = 0.50) -> pd.DataFrame:
    """Score an entire CSV file and write results."""
    model, preprocessor, model_name = load_artifacts()

    df_raw = pd.read_csv(csv_path)
    ids    = df_raw["CustomerID"].values if "CustomerID" in df_raw.columns else range(len(df_raw))

    df_eng = engineer_features(df_raw)
    X, _   = preprocessor.transform(df_eng)

    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= threshold).astype(int)
    risks  = pd.cut(probs, bins=[-0.001, 0.40, 0.70, 1.001],
                    labels=["Low", "Medium", "High"])

    out_df = pd.DataFrame({
        "CustomerID":        ids,
        "ChurnProbability":  probs.round(4),
        "ChurnPrediction":   preds,
        "RiskLevel":         risks,
        "ModelUsed":         model_name,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[Predict] Batch predictions → {output_path}  ({len(out_df)} records)")
    return out_df


if __name__ == "__main__":
    # Quick demo
    sample = {
        "CustomerID":             "CUST-999999",
        "Gender":                 "Female",
        "SeniorCitizen":          0,
        "Partner":                "No",
        "Dependents":             "No",
        "tenure":                 3,
        "PhoneService":           "Yes",
        "MultipleLines":          "No",
        "InternetService":        "Fiber optic",
        "OnlineSecurity":         "No",
        "OnlineBackup":           "No",
        "DeviceProtection":       "No",
        "TechSupport":            "No",
        "StreamingTV":            "Yes",
        "StreamingMovies":        "Yes",
        "Contract":               "Month-to-month",
        "PaperlessBilling":       "Yes",
        "PaymentMethod":          "Electronic check",
        "MonthlyCharges":         95.50,
        "TotalCharges":           286.50,
        "NumSupportCalls":        4,
        "NumProductsUsed":        2,
        "DaysSinceLastContact":   120,
        "AvgMonthlyUsage":        210.0,
        "ContractMonthsRemaining":0,
    }
    result = predict_single(sample)
    print("\nPrediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
