"""
dashboard/app.py
================
Streamlit analytics dashboard for the Customer Churn Prediction System.
Sections:
  1. KPI overview
  2. Model leaderboard
  3. Evaluation charts (confusion matrix, ROC, PR, scores, feature importance)
  4. Single-customer live predictor
  5. Batch upload & scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Churn Analytics | AIVONEX",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg,#1A237E 0%,#283593 100%);
    border-radius:12px; padding:22px 18px; color:white; text-align:center;
    box-shadow:0 4px 15px rgba(26,35,126,.25);
  }
  .metric-card .val  { font-size:2.2rem; font-weight:700; }
  .metric-card .lbl  { font-size:.85rem; opacity:.85; margin-top:4px; }
  .risk-high   { background:#E53935; color:white; border-radius:6px; padding:4px 10px; font-weight:600; }
  .risk-medium { background:#FB8C00; color:white; border-radius:6px; padding:4px 10px; font-weight:600; }
  .risk-low    { background:#43A047; color:white; border-radius:6px; padding:4px 10px; font-weight:600; }
  .stButton>button {
    background:linear-gradient(135deg,#1A237E,#283593);
    color:white; border:none; border-radius:8px;
    padding:10px 24px; font-weight:600; font-size:.95rem;
    transition:all .2s;
  }
  .stButton>button:hover { opacity:.88; transform:translateY(-1px); }
  h1,h2,h3 { color:#1A237E; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    from src.predict import load_artifacts
    try:
        model, preprocessor, model_name = load_artifacts()
        return model, preprocessor, model_name
    except Exception as e:
        return None, None, str(e)


def metric_card(value, label, col):
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{value}</div>
      <div class="lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


REPORTS = Path("reports")
MODELS  = Path("models")


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = Path(__file__).parent / "assets" / "aivonex_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    st.markdown("## 📉 Churn Analytics")
    st.markdown("---")
    section = st.radio("Navigate", [
        "🏠  Overview",
        "🏆  Model Leaderboard",
        "📊  Evaluation Charts",
        "🔍  Live Predictor",
        "📁  Batch Scoring",
    ])
    st.markdown("---")
    threshold = st.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption(f"Churn if P ≥ {threshold}")

st.title("Customer Churn Prediction System")
st.caption("Powered by Machine Learning | AIVONEX SMC-PVT LTD")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════
#  1. OVERVIEW
# ════════════════════════════════════════════════════════════════════
if "Overview" in section:
    st.subheader("📌 Project Overview")
    st.markdown("""
    This dashboard provides end-to-end visibility into the **Customer Churn Prediction System**:

    - **10,000-record** synthetic telecom dataset with realistic churn patterns
    - Four ML models trained: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
    - SMOTE applied for class-imbalance handling
    - Full REST API (`/predict`, `/predict/batch`) for production integration
    - Business impact estimation with revenue-at-risk analysis
    """)

    lb_path = MODELS / "leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        best = lb.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        metric_card(best["Model"],         "Best Model",      c1)
        metric_card(f"{best['Val_AUC']:.4f}", "Validation AUC", c2)
        metric_card(f"{best['Val_F1']:.4f}",  "F1 Score",       c3)
        metric_card(f"{best['Val_Recall']:.4f}","Recall",        c4)
        st.markdown("")
        c5,c6,c7,c8 = st.columns(4)
        metric_card("10,000",    "Training Records",    c5)
        metric_card("~26%",      "Churn Rate",          c6)
        metric_card("4",         "Models Trained",      c7)
        metric_card("25+",       "Features Engineered", c8)
    else:
        st.info("Run `python run_pipeline.py` to train models and populate this dashboard.")


# ════════════════════════════════════════════════════════════════════
#  2. LEADERBOARD
# ════════════════════════════════════════════════════════════════════
elif "Leaderboard" in section:
    st.subheader("🏆 Model Leaderboard")
    lb_path = MODELS / "leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        lb.insert(0, "Rank", range(1, len(lb)+1))
        st.dataframe(lb.style.highlight_max(
            subset=["Val_AUC","Val_F1","Val_Recall"],
            color="#C8E6C9"
        ).highlight_min(
            subset=["Train_Time_s"],
            color="#E3F2FD"
        ), use_container_width=True, height=300)

        st.markdown("### Metric Comparison")
        import plotly.graph_objects as go
        metrics = ["Val_AUC","Val_Accuracy","Val_Precision","Val_Recall","Val_F1"]
        fig = go.Figure()
        for _, row in lb.iterrows():
            fig.add_trace(go.Bar(name=row["Model"], x=metrics,
                                 y=[row[m] for m in metrics]))
        fig.update_layout(barmode="group", height=420, plot_bgcolor="white",
                          paper_bgcolor="white", font_family="Inter",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No leaderboard found. Run the training pipeline first.")


# ════════════════════════════════════════════════════════════════════
#  3. EVALUATION CHARTS
# ════════════════════════════════════════════════════════════════════
elif "Evaluation" in section:
    st.subheader("📊 Model Evaluation Visualisations")
    chart_files = {
        "Confusion Matrix":       REPORTS / "confusion_matrix.png",
        "ROC-AUC Curve":          REPORTS / "roc_curve.png",
        "Precision-Recall Curve": REPORTS / "pr_curve.png",
        "Score Distribution":     REPORTS / "score_distribution.png",
        "Feature Importance":     REPORTS / "feature_importance.png",
    }
    existing = {k: v for k, v in chart_files.items() if v.exists()}
    if not existing:
        st.info("No evaluation charts found. Run the training pipeline first.")
    else:
        tabs = st.tabs(list(existing.keys()))
        for tab, (name, path) in zip(tabs, existing.items()):
            with tab:
                st.image(str(path), use_column_width=True)

    # Business impact
    biz_path = REPORTS / "business_impact_report.xlsx"
    if biz_path.exists():
        st.markdown("### 💼 Business Impact Report")
        df_biz = pd.read_excel(biz_path)
        st.table(df_biz)


# ════════════════════════════════════════════════════════════════════
#  4. LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════════
elif "Predictor" in section:
    st.subheader("🔍 Live Single-Customer Predictor")

    model, preprocessor, model_name = load_model_artifacts()
    if model is None:
        st.error(f"Model not loaded: {model_name}")
    else:
        st.caption(f"Using: **{model_name}** | Threshold: **{threshold}**")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Demographics")
            gender     = st.selectbox("Gender",       ["Male","Female"])
            senior     = st.selectbox("Senior Citizen", [0,1])
            partner    = st.selectbox("Partner",      ["Yes","No"])
            dependents = st.selectbox("Dependents",   ["Yes","No"])
            tenure     = st.slider("Tenure (months)", 1, 72, 12)

        with col_r:
            st.markdown("##### Services")
            internet   = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            contract   = st.selectbox("Contract",     ["Month-to-month","One year","Two year"])
            payment    = st.selectbox("Payment Method",
                                      ["Electronic check","Mailed check",
                                       "Bank transfer (automatic)","Credit card (automatic)"])
            monthly    = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)
            support    = st.slider("Support Calls", 0, 15, 2)

        with col_l:
            st.markdown("##### Extras")
            paperless  = st.selectbox("Paperless Billing", ["Yes","No"])
            products   = st.slider("Products Used", 1, 6, 2)
            days_last  = st.slider("Days Since Last Contact", 1, 365, 60)
            usage      = st.number_input("Avg Monthly Usage (GB)", 10.0, 300.0, 100.0)

        record = {
            "CustomerID":             "LIVE-001",
            "Gender":                 gender,
            "SeniorCitizen":          senior,
            "Partner":                partner,
            "Dependents":             dependents,
            "tenure":                 float(tenure),
            "PhoneService":           "Yes",
            "MultipleLines":          "No",
            "InternetService":        internet,
            "OnlineSecurity":         "No",
            "OnlineBackup":           "No",
            "DeviceProtection":       "No",
            "TechSupport":            "No",
            "StreamingTV":            "No",
            "StreamingMovies":        "No",
            "Contract":               contract,
            "PaperlessBilling":       paperless,
            "PaymentMethod":          payment,
            "MonthlyCharges":         monthly,
            "TotalCharges":           round(monthly * tenure, 2),
            "NumSupportCalls":        support,
            "NumProductsUsed":        products,
            "DaysSinceLastContact":   days_last,
            "AvgMonthlyUsage":        usage,
            "ContractMonthsRemaining":0 if contract == "Month-to-month" else 6,
        }

        if st.button("⚡ Predict Churn"):
            from src.predict import predict_single
            result = predict_single(record, model=model, preprocessor=preprocessor,
                                    threshold=threshold)
            prob  = result["churn_probability"]
            risk  = result["risk_level"]
            pred  = result["churn_prediction"]

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("Churn Probability", f"{prob:.1%}")
            with rc2:
                st.metric("Prediction", "🔴 CHURN" if pred else "🟢 STAY")
            with rc3:
                risk_color = {"High":"🔴","Medium":"🟡","Low":"🟢"}[risk]
                st.metric("Risk Level", f"{risk_color} {risk}")

            st.progress(prob)

            if risk == "High":
                st.error("⚠️ HIGH RISK — Immediate retention action recommended.")
            elif risk == "Medium":
                st.warning("🔶 MEDIUM RISK — Monitor and engage proactively.")
            else:
                st.success("✅ LOW RISK — Customer appears stable.")


# ════════════════════════════════════════════════════════════════════
#  5. BATCH SCORING
# ════════════════════════════════════════════════════════════════════
elif "Batch" in section:
    st.subheader("📁 Batch Customer Scoring")

    st.markdown("""
    Upload a CSV file containing customer records.
    The file must include the same columns as the training data.
    Results are returned as a downloadable CSV with churn probabilities and risk tiers.
    """)

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Uploaded:** {len(df_up)} customers")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("🚀 Score All Customers"):
            model, preprocessor, model_name = load_model_artifacts()
            if model is None:
                st.error("Model not available.")
            else:
                from src.feature_engineering import engineer_features
                from src.predict import predict_batch
                import tempfile, os

                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    df_up.to_csv(tmp.name, index=False)
                    tmp_name = tmp.name

                out_path = "reports/batch_predictions.csv"
                result_df = predict_batch(tmp_name, out_path, threshold=threshold)
                os.unlink(tmp_name)

                st.success(f"✅ Scored {len(result_df)} customers")

                c1, c2, c3, c4 = st.columns(4)
                metric_card(len(result_df), "Total Customers", c1)
                metric_card(int(result_df["ChurnPrediction"].sum()), "Predicted Churn", c2)
                metric_card(f"{result_df['ChurnPrediction'].mean():.1%}", "Churn Rate", c3)
                metric_card(int((result_df["RiskLevel"]=="High").sum()), "High Risk", c4)

                st.dataframe(result_df, use_container_width=True, height=350)

                csv = result_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", csv,
                                   "churn_predictions.csv", "text/csv")

st.markdown("---")
st.caption("© 2024 AIVONEX SMC-PVT LTD | Customer Churn Prediction System v1.0")