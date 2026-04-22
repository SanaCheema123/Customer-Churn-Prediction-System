"""
model_training.py
=================
Trains four classifiers with cross-validation and SMOTE for class imbalance.
Saves every model to disk and returns a summary leaderboard.
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics        import (roc_auc_score, f1_score, precision_score,
                                    recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def _build_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
        )
    return models


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Balance classes with SMOTE oversampling."""
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Before: {np.bincount(y_train)}  →  After: {np.bincount(y_res)}")
    return X_res, y_res


def train_all_models(X_train, y_train, X_val, y_val,
                     use_smote: bool = True, cv_folds: int = 5) -> pd.DataFrame:
    """
    Train all models, evaluate on val set, run CV, and save artefacts.
    Returns a DataFrame leaderboard sorted by val ROC-AUC.
    """
    if use_smote:
        X_tr, y_tr = apply_smote(X_train, y_train)
    else:
        X_tr, y_tr = X_train, y_train

    models   = _build_models()
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    records  = []

    for name, model in models.items():
        print(f"\n[Training] ── {name} ──")
        t0 = time.time()

        # Cross-validation on balanced training data
        cv_aucs = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  CV AUC : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

        # Final fit on full training set
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        # Validation metrics
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        rec = {
            "Model":            name,
            "CV_AUC_Mean":      round(cv_aucs.mean(), 4),
            "CV_AUC_Std":       round(cv_aucs.std(),  4),
            "Val_AUC":          round(roc_auc_score(y_val, y_prob), 4),
            "Val_Accuracy":     round(accuracy_score(y_val, y_pred), 4),
            "Val_Precision":    round(precision_score(y_val, y_pred, zero_division=0), 4),
            "Val_Recall":       round(recall_score(y_val, y_pred, zero_division=0), 4),
            "Val_F1":           round(f1_score(y_val, y_pred, zero_division=0), 4),
            "Train_Time_s":     round(elapsed, 2),
        }
        records.append(rec)
        print(f"  Val AUC={rec['Val_AUC']}  F1={rec['Val_F1']}  Recall={rec['Val_Recall']}")

        # Save model
        model_path = MODELS_DIR / f"model_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved → {model_path}")

    leaderboard = pd.DataFrame(records).sort_values("Val_AUC", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(MODELS_DIR / "leaderboard.csv", index=False)

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    return leaderboard


def load_best_model(leaderboard: pd.DataFrame = None, model_name: str = None):
    """Load the best-performing saved model."""
    if model_name is None:
        if leaderboard is None:
            leaderboard = pd.read_csv(MODELS_DIR / "leaderboard.csv")
        model_name = leaderboard.iloc[0]["Model"]

    path = MODELS_DIR / f"model_{model_name.lower().replace(' ', '_')}.joblib"
    model = joblib.load(path)
    print(f"[ModelLoader] Loaded best model: {model_name}  ← {path}")
    return model, model_name


if __name__ == "__main__":
    from src.data_generator   import generate_churn_dataset
    from src.feature_engineering import engineer_features
    from src.data_preprocessing  import prepare_splits

    generate_churn_dataset()
    prepare_splits()
