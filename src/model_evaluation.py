"""
model_evaluation.py
===================
Comprehensive evaluation of the best model on the held-out test set.
Generates:
  - Classification report (text)
  - Confusion matrix plot
  - ROC-AUC curve
  - Precision-Recall curve
  - Feature importance chart
  - Business impact summary (Excel)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "primary":   "#1A237E",
    "secondary": "#E53935",
    "accent":    "#00ACC1",
    "light":     "#E8EAF6",
    "dark":      "#0D0D0D",
}


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#CCCCCC",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: list = None, model_name: str = "Best Model") -> dict:
    """Run full evaluation suite and save all artefacts."""
    _set_style()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    # ── Text report ─────────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"])
    rpt_path = REPORTS_DIR / "classification_report.txt"
    rpt_path.write_text(report)
    print("[Evaluation] Classification Report:")
    print(report)

    metrics = {
        "model_name": model_name,
        "test_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "report":     report,
    }

    # ── 1. Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"],
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("Actual",    fontsize=13, labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=15,
                 color=PALETTE["primary"], fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("[Evaluation] Confusion matrix saved.")

    # ── 2. ROC Curve ────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
            label=f"{model_name}  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC — AUC Curve", fontsize=14, fontweight="bold",
                 color=PALETTE["primary"])
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"[Evaluation] ROC-AUC = {roc_auc:.4f}  saved.")
    metrics["test_auc"] = round(roc_auc, 4)

    # ── 3. Precision-Recall Curve ────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color=PALETTE["secondary"], lw=2.5,
            label=f"AP = {ap:.4f}")
    ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["secondary"])
    baseline = y_test.mean()
    ax.axhline(baseline, color="#AAAAAA", lw=1.5, linestyle="--",
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold",
                 color=PALETTE["secondary"])
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)
    print(f"[Evaluation] PR curve (AP={ap:.4f}) saved.")

    # ── 4. Feature Importance ────────────────────────────────────────
    if feature_names and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        feat_df = feat_df.nlargest(20, "Importance").sort_values("Importance")
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                       color=PALETTE["primary"], alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#333333")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title("Top-20 Feature Importances", fontsize=14,
                     fontweight="bold", color=PALETTE["primary"])
        plt.tight_layout()
        fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
        feat_df.sort_values("Importance", ascending=False).to_csv(
            REPORTS_DIR / "feature_importance.csv", index=False)
        print("[Evaluation] Feature importance saved.")

    # ── 5. Business Impact Summary ───────────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    _save_business_report(tn, fp, fn, tp, roc_auc, model_name)

    # ── 6. Score distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_test == 0], bins=40, alpha=0.65, color=PALETTE["accent"],
            label="Not Churn", edgecolor="white")
    ax.hist(y_prob[y_test == 1], bins=40, alpha=0.65, color=PALETTE["secondary"],
            label="Churn", edgecolor="white")
    ax.axvline(0.50, color=PALETTE["dark"], lw=1.5, linestyle="--", label="Threshold = 0.50")
    ax.set_xlabel("Predicted Churn Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution", fontsize=14, fontweight="bold",
                 color=PALETTE["primary"])
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    print("[Evaluation] Score distribution saved.")

    print(f"\n[Evaluation] All reports → {REPORTS_DIR}/")
    return metrics


def _save_business_report(tn, fp, fn, tp, auc_score, model_name):
    """Excel summary with business KPIs."""
    avg_clv         = 1_200      # USD, average customer lifetime value
    retention_cost  = 50         # USD, cost of a retention action
    total_at_risk   = tp + fn

    correctly_retained = tp
    missed_churners    = fn
    false_alarms       = fp
    revenue_saved      = correctly_retained * avg_clv
    retention_spend    = (tp + fp) * retention_cost
    net_value          = revenue_saved - retention_spend

    summary = {
        "Metric":       ["Model", "Test AUC", "True Positives (caught churners)",
                         "False Negatives (missed churners)", "False Positives (false alarms)",
                         "Total Customers At Risk", "Estimated Revenue Saved ($)",
                         "Retention Spend ($)", "Net Business Value ($)"],
        "Value":        [model_name, auc_score, tp, fn, fp, total_at_risk,
                         f"${revenue_saved:,.0f}", f"${retention_spend:,.0f}", f"${net_value:,.0f}"]
    }
    df_summary = pd.DataFrame(summary)
    out_path = REPORTS_DIR / "business_impact_report.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Business Impact", index=False)
    print(f"[Evaluation] Business impact report → {out_path}")


if __name__ == "__main__":
    print("Run via run_pipeline.py")
