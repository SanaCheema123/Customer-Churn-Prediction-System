# 📉 Customer Churn Prediction System
**AIVONEX SMC-PVT LTD — Production ML Project**

---

## 📌 Project Summary

A full end-to-end **Customer Churn Prediction** system built for telecom companies.
The system ingests customer data, engineers domain-specific features, trains and evaluates
four ML classifiers, and exposes the best model through a **REST API** and an interactive
**Streamlit dashboard**.

---

## 🏗 Folder Structure

```
customer_churn_prediction/
│
├── config/
│   └── config.yaml               ← All project settings (paths, hyper-params, etc.)
│
├── data/
│   ├── raw/
│   │   └── churn_data.csv        ← Generated synthetic dataset (10,000 records)
│   └── processed/
│       └── churn_engineered.csv  ← Feature-engineered dataset
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py         ← Synthetic dataset generator
│   ├── data_preprocessing.py     ← DataPreprocessor class (encode, scale, split)
│   ├── feature_engineering.py    ← 15+ derived domain features
│   ├── model_training.py         ← Train 4 models + SMOTE + cross-validation
│   ├── model_evaluation.py       ← Charts, metrics, business impact report
│   └── predict.py                ← Single & batch prediction utilities
│
├── api/
│   ├── __init__.py
│   ├── main.py                   ← FastAPI application (3 endpoints)
│   └── schemas.py                ← Pydantic request/response models
│
├── dashboard/
│   └── app.py                    ← Streamlit 5-section analytics dashboard
│
├── models/                       ← Saved models + preprocessor + leaderboard (auto-created)
│   ├── preprocessor.joblib
│   ├── model_logistic_regression.joblib
│   ├── model_random_forest.joblib
│   ├── model_gradient_boosting.joblib
│   ├── model_xgboost.joblib
│   └── leaderboard.csv
│
├── reports/                      ← Evaluation artefacts (auto-created)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── score_distribution.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   ├── classification_report.txt
│   ├── business_impact_report.xlsx
│   └── batch_predictions.csv
│
├── notebooks/                    ← Jupyter notebooks for EDA
├── requirements.txt
├── run_pipeline.py               ← ✅ Master pipeline runner
└── README.md
```

---

## ⚙️ Setup

### 1. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Full Pipeline

```bash
python run_pipeline.py
```

This single command will:

| Step | Action |
|------|--------|
| 1 | Generate 10,000-record synthetic telecom dataset |
| 2 | Engineer 15+ domain-specific features |
| 3 | Preprocess, encode, scale, and split data (70/10/20) |
| 4 | Train 4 ML models with 5-fold CV + SMOTE |
| 5 | Evaluate best model on test set, save all charts |
| 6 | Print final summary + next-step instructions |

Skip data generation on subsequent runs:
```bash
python run_pipeline.py --skip-data
```

---

## 🌐 REST API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/docs` | GET | Swagger interactive docs |
| `/health` | GET | Service health check |
| `/model/info` | GET | Leaderboard + active model |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | CSV batch scoring (file upload) |
| `/predict/batch/summary` | GET | Stats of last batch run |

### Example — Single Prediction

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50, "TotalCharges": 286.50,
    "NumSupportCalls": 4, "NumProductsUsed": 2,
    "DaysSinceLastContact": 120, "AvgMonthlyUsage": 210.0,
    "ContractMonthsRemaining": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8312,
  "churn_prediction": 1,
  "risk_level": "High",
  "threshold_used": 0.5
}
```

---

## 📊 Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

**Dashboard Sections:**
- **Overview** — KPI cards, project summary
- **Model Leaderboard** — all model metrics + bar chart comparison
- **Evaluation Charts** — confusion matrix, ROC, PR, score dist., feature importance
- **Live Predictor** — real-time single-customer churn scoring
- **Batch Scoring** — CSV upload → scored CSV download

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | 200 trees, max_depth=10 |
| Gradient Boosting | sklearn GBM, 200 estimators |
| XGBoost | Best performer, eval_metric=logloss |

All trained with:
- **SMOTE** oversampling to handle ~26% churn imbalance
- **5-fold Stratified CV** for robust evaluation
- Saved as `.joblib` for fast loading

---

## 📈 Feature Engineering (15+ features)

| Feature | Description |
|---|---|
| `ChargesPerTenureMonth` | Avg monthly spend normalized by tenure |
| `ChargeAboveAvg` | Monthly charge delta from dataset mean |
| `SupportCallRate` | Support calls per year of tenure |
| `DisengagementScore` | Time since contact × inverse product count |
| `IsMonthToMonth` | High-churn contract flag |
| `ContractRiskScore` | Contract type × new-customer risk |
| `AddOnCount` | Count of subscribed add-on services |
| `ChurnRiskIndex` | Composite weighted risk score (0–1) |
| `TenureBucket` | Categorical tenure bands |
| `IsElectronicCheck` | High-churn payment method flag |
| `IsAutoPay` | Auto-payment loyalty indicator |
| + more | … |

---

## 📂 Deliverables for Client

| Deliverable | Location |
|---|---|
| Trained models | `models/*.joblib` |
| Preprocessor | `models/preprocessor.joblib` |
| Model leaderboard | `models/leaderboard.csv` |
| Classification report | `reports/classification_report.txt` |
| Evaluation charts (5x) | `reports/*.png` |
| Business impact (Excel) | `reports/business_impact_report.xlsx` |
| REST API | `api/` — deploy with `uvicorn` |
| Dashboard | `dashboard/app.py` — deploy with `streamlit` |

---

## 🏢 Company

**AIVONEX SMC-PVT LTD**
Bahawalpur, Pakistan
AI / ML Managed Services

---

*Version 1.0.0 | Customer Churn Prediction System*
