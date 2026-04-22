"""
api/schemas.py
==============
Pydantic v2 models for request validation and response serialisation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class CustomerRecord(BaseModel):
    """Input schema for a single customer churn prediction request."""

    CustomerID:              Optional[str]   = Field(None,  example="CUST-000001")
    Gender:                  str             = Field(...,   example="Female")
    SeniorCitizen:           int             = Field(...,   ge=0, le=1, example=0)
    Partner:                 str             = Field(...,   example="Yes")
    Dependents:              str             = Field(...,   example="No")
    tenure:                  float           = Field(...,   ge=0, example=3.0)
    PhoneService:            str             = Field(...,   example="Yes")
    MultipleLines:           str             = Field(...,   example="No")
    InternetService:         str             = Field(...,   example="Fiber optic")
    OnlineSecurity:          str             = Field(...,   example="No")
    OnlineBackup:            str             = Field(...,   example="No")
    DeviceProtection:        str             = Field(...,   example="No")
    TechSupport:             str             = Field(...,   example="No")
    StreamingTV:             str             = Field(...,   example="Yes")
    StreamingMovies:         str             = Field(...,   example="Yes")
    Contract:                str             = Field(...,   example="Month-to-month")
    PaperlessBilling:        str             = Field(...,   example="Yes")
    PaymentMethod:           str             = Field(...,   example="Electronic check")
    MonthlyCharges:          float           = Field(...,   ge=0, example=95.50)
    TotalCharges:            float           = Field(...,   ge=0, example=286.50)
    NumSupportCalls:         int             = Field(...,   ge=0, example=4)
    NumProductsUsed:         int             = Field(...,   ge=1, example=2)
    DaysSinceLastContact:    int             = Field(...,   ge=0, example=120)
    AvgMonthlyUsage:         float           = Field(...,   ge=0, example=210.0)
    ContractMonthsRemaining: int             = Field(...,   ge=0, example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "CustomerID":             "CUST-000001",
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
        }


class PredictionResponse(BaseModel):
    """Output schema for churn prediction."""

    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    churn_prediction:  int   = Field(..., description="Binary label: 1 = Churn, 0 = Stay")
    risk_level:        str   = Field(..., description="Low / Medium / High")
    threshold_used:    float = Field(..., description="Decision threshold applied")


class BatchSummary(BaseModel):
    """Summary of a completed batch prediction run."""

    total_customers:   int
    predicted_churn:   int
    churn_rate:        float
    high_risk_count:   int
    medium_risk_count: int
    low_risk_count:    int
    avg_churn_prob:    float
