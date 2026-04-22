"""
feature_engineering.py
=======================
Creates derived features from the raw data to improve model performance.
Called BEFORE the preprocessor so engineered columns go through scaling/encoding.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven derived features to the raw DataFrame."""
    df = df.copy()

    # ── Revenue & Billing Risk ───────────────────────────────────────
    # Charges per month of tenure
    df["ChargesPerTenureMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Delta between monthly charge and average — signals price sensitivity
    df["ChargeAboveAvg"] = df["MonthlyCharges"] - df["MonthlyCharges"].mean()

    # High monthly charges flag
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] > 75).astype(int)

    # ── Engagement Risk ──────────────────────────────────────────────
    # Normalised support call rate (per year of tenure)
    df["SupportCallRate"] = np.where(
        df["tenure"] > 0,
        df["NumSupportCalls"] / (df["tenure"] / 12),
        df["NumSupportCalls"]
    )

    # Disengaged: long time since last contact AND few products
    df["DisengagementScore"] = (
        (df["DaysSinceLastContact"] / 365) * (1 / (df["NumProductsUsed"] + 1))
    )

    # ── Contract Stickiness ──────────────────────────────────────────
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsLongTermContract"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)
    df["ContractRiskScore"] = df["IsMonthToMonth"] * (1 - df["tenure"] / 72)

    # ── Service Bundle Richness ──────────────────────────────────────
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    present = [c for c in addon_cols if c in df.columns]
    df["AddOnCount"] = df[present].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["HasNoAddOns"] = (df["AddOnCount"] == 0).astype(int)

    # ── Internet Tier ────────────────────────────────────────────────
    df["IsFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["HasInternet"]  = (df["InternetService"] != "No").astype(int)

    # ── Payment Method Risk ──────────────────────────────────────────
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["IsAutoPay"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    # ── Tenure Buckets ────────────────────────────────────────────────
    df["TenureBucket"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "12-24mo", "24-48mo", "48-72mo"]
    ).astype(str)

    # ── Composite Churn Risk Index (interpretable proxy) ─────────────
    df["ChurnRiskIndex"] = (
        0.30 * df["IsMonthToMonth"]
      + 0.20 * (df["tenure"] < 12).astype(int)
      + 0.15 * df["IsFiberOptic"]
      + 0.15 * df["IsElectronicCheck"]
      + 0.10 * (df["NumSupportCalls"] > 5).astype(int)
      + 0.10 * df["HasNoAddOns"]
    )

    print(f"[FeatureEngineering] Features after engineering: {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/raw/churn_data.csv")
    df_eng = engineer_features(df)
    print(df_eng.columns.tolist())
    print(df_eng.head(3))
