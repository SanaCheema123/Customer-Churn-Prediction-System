"""
api/main.py
===========
FastAPI REST API for the Customer Churn Prediction System.
Endpoints:
  POST /predict          — single customer prediction
  POST /predict/batch    — CSV file batch prediction
  GET  /health           — service health-check
  GET  /model/info       — current model metadata
"""

import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerRecord, PredictionResponse, BatchSummary
from src.predict import predict_single, predict_batch, load_artifacts

# ── App init ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="AIVONEX SMC-PVT LTD — ML-powered churn prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artefacts once at startup
_model, _preprocessor, _model_name = None, None, None


@app.on_event("startup")
def startup_event():
    global _model, _preprocessor, _model_name
    try:
        _model, _preprocessor, _model_name = load_artifacts()
        print(f"[API] Model loaded: {_model_name}")
    except Exception as e:
        print(f"[API] WARNING: Could not load model on startup: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["Utility"])
def root():
    return {
        "project": "Customer Churn Prediction API",
        "company": "AIVONEX SMC-PVT LTD",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/health",
        "model":   "/model/info",
    }


@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok", "model_loaded": _model is not None, "model": _model_name}


@app.get("/model/info", tags=["Utility"])
def model_info():
    lb_path = Path("models/leaderboard.csv")
    if lb_path.exists():
        lb = pd.read_csv(lb_path).to_dict(orient="records")
    else:
        lb = []
    return {
        "active_model": _model_name,
        "leaderboard":  lb,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_customer(record: CustomerRecord, threshold: float = Query(0.50, ge=0.01, le=0.99)):
    """
    Predict churn probability for a single customer.
    Returns churn probability, binary prediction, and risk tier.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")

    result = predict_single(record.dict(), model=_model, preprocessor=_preprocessor, threshold=threshold)
    return PredictionResponse(**result)


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.50, ge=0.01, le=0.99)
):
    """
    Score an entire CSV file of customers.
    Returns a downloadable CSV with churn probability and risk tier per customer.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    df_raw = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    tmp_in  = "data/processed/batch_input_tmp.csv"
    tmp_out = "reports/batch_predictions.csv"
    df_raw.to_csv(tmp_in, index=False)

    result_df = predict_batch(tmp_in, output_path=tmp_out, threshold=threshold)

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"}
    )


@app.get("/predict/batch/summary", tags=["Prediction"])
def batch_summary():
    """Return summary stats of the last batch prediction run."""
    path = Path("reports/batch_predictions.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch prediction found. Run /predict/batch first.")
    df = pd.read_csv(path)
    return {
        "total_customers":   len(df),
        "predicted_churn":   int(df["ChurnPrediction"].sum()),
        "churn_rate":        round(df["ChurnPrediction"].mean(), 4),
        "high_risk_count":   int((df["RiskLevel"] == "High").sum()),
        "medium_risk_count": int((df["RiskLevel"] == "Medium").sum()),
        "low_risk_count":    int((df["RiskLevel"] == "Low").sum()),
        "avg_churn_prob":    round(df["ChurnProbability"].mean(), 4),
    }