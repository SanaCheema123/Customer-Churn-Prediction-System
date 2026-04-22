"""
run_pipeline.py
===============
Master script — runs the full end-to-end training pipeline:

  Step 1: Generate synthetic dataset
  Step 2: Feature engineering
  Step 3: Preprocessing & splits
  Step 4: Train all models
  Step 5: Evaluate best model on test set
  Step 6: Save summary report

Usage:
  python run_pipeline.py              # full pipeline
  python run_pipeline.py --skip-data  # skip data generation (re-use existing CSV)
"""

import argparse
import time
from pathlib import Path

import pandas as pd


def banner(msg: str):
    print("\n" + "═" * 65)
    print(f"  {msg}")
    print("═" * 65)


def run_pipeline(skip_data: bool = False):
    t_start = time.time()

    # ── Step 1: Data Generation ──────────────────────────────────────
    banner("STEP 1 — Data Generation")
    if skip_data and Path("data/raw/churn_data.csv").exists():
        print("[Pipeline] Skipping generation — using existing data/raw/churn_data.csv")
        df_raw = pd.read_csv("data/raw/churn_data.csv")
    else:
        from src.data_generator import generate_churn_dataset
        df_raw = generate_churn_dataset(n_samples=10_000, save_path="data/raw/churn_data.csv")

    # ── Step 2: Feature Engineering ──────────────────────────────────
    banner("STEP 2 — Feature Engineering")
    from src.feature_engineering import engineer_features
    df_eng = engineer_features(df_raw)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_eng.to_csv("data/processed/churn_engineered.csv", index=False)
    print(f"[Pipeline] Engineered dataset → data/processed/churn_engineered.csv  {df_eng.shape}")

    # ── Step 3: Preprocessing & Splits ───────────────────────────────
    banner("STEP 3 — Preprocessing & Train/Val/Test Splits")
    from src.data_preprocessing import prepare_splits
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_splits(
        raw_path="data/processed/churn_engineered.csv",
        test_size=0.20,
        val_size=0.10,
    )

    # ── Step 4: Model Training ────────────────────────────────────────
    banner("STEP 4 — Model Training (4 classifiers + SMOTE)")
    from src.model_training import train_all_models
    leaderboard = train_all_models(X_train, y_train, X_val, y_val,
                                   use_smote=True, cv_folds=5)

    # ── Step 5: Evaluation ────────────────────────────────────────────
    banner("STEP 5 — Full Evaluation on Test Set")
    from src.model_training  import load_best_model
    from src.model_evaluation import evaluate_model

    model, model_name = load_best_model(leaderboard=leaderboard)
    metrics = evaluate_model(
        model, X_test, y_test,
        feature_names=preprocessor.feature_names_,
        model_name=model_name,
    )

    # ── Step 6: Pipeline Summary ──────────────────────────────────────
    banner("STEP 6 — Pipeline Complete")
    elapsed = time.time() - t_start
    print(f"  Best Model : {model_name}")
    print(f"  Test AUC   : {metrics['test_auc']}")
    print(f"  Total Time : {elapsed:.1f}s")
    print(f"\n  Artefacts saved to:")
    print(f"    models/       — trained models + preprocessor + leaderboard")
    print(f"    reports/      — charts + classification report + business impact")
    print(f"\n  Next steps:")
    print(f"    API       →  uvicorn api.main:app --reload --port 8000")
    print(f"    Dashboard →  streamlit run dashboard/app.py")
    print("═" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation if CSV already exists")
    args = parser.parse_args()
    run_pipeline(skip_data=args.skip_data)
