"""
data_preprocessing.py
=====================
Handles loading, cleaning, encoding, scaling, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


RANDOM_STATE = 42


class DataPreprocessor:
    """End-to-end preprocessing pipeline for churn data."""

    def __init__(self, target_col: str = "Churn", drop_cols: list = None):
        self.target_col     = target_col
        self.drop_cols      = drop_cols or ["CustomerID"]
        self.scaler         = StandardScaler()
        self.label_encoders = {}
        self.imputer_num    = SimpleImputer(strategy="median")
        self.imputer_cat    = SimpleImputer(strategy="most_frequent")
        self.feature_names_  = []
        self.num_cols_       = []
        self.cat_cols_       = []
        self._fitted         = False

    # ── Public API ───────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """Fit on training data and transform it."""
        df = self._clean(df)
        X, y = self._split_xy(df)
        X = self._encode_fit(X)
        X = self._scale_fit(X)
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame):
        """Transform unseen data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = self._clean(df, is_inference=True)
        if self.target_col in df.columns:
            X, y = self._split_xy(df)
        else:
            X, y = df.drop(columns=self.drop_cols, errors="ignore"), None
        X = self._encode_transform(X)
        X = self._scale_transform(X)
        return X, y

    def save(self, path: str = "models/preprocessor.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[Preprocessor] Saved → {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.joblib"):
        obj = joblib.load(path)
        print(f"[Preprocessor] Loaded ← {path}")
        return obj

    # ── Private Helpers ──────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
        df = df.copy()
        # TotalCharges can contain spaces in telecom datasets
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Drop fully-empty columns
        df.dropna(axis=1, how="all", inplace=True)
        return df

    def _split_xy(self, df: pd.DataFrame):
        y = df[self.target_col].values.astype(int)
        X = df.drop(columns=self.drop_cols + [self.target_col], errors="ignore")
        return X, y

    def _identify_cols(self, X: pd.DataFrame):
        self.num_cols_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()

    def _encode_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        self._identify_cols(X)
        # Impute
        X[self.num_cols_] = self.imputer_num.fit_transform(X[self.num_cols_])
        if self.cat_cols_:
            X[self.cat_cols_] = self.imputer_cat.fit_transform(X[self.cat_cols_])
        # Label encode categoricals
        for col in self.cat_cols_:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        self.feature_names_ = X.columns.tolist()
        return X

    def _encode_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        X[self.num_cols_] = self.imputer_num.transform(X[self.num_cols_])
        for col in self.cat_cols_:
            le = self.label_encoders[col]
            X[col] = X[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else 0
            )
        return X

    def _scale_fit(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names_].values.astype(float)
        return self.scaler.fit_transform(X_arr)

    def _scale_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names_].values.astype(float)
        return self.scaler.transform(X_arr)


def prepare_splits(raw_path: str = "data/raw/churn_data.csv",
                   test_size: float = 0.20,
                   val_size: float = 0.10,
                   random_state: int = RANDOM_STATE):
    """Load raw CSV → clean → encode → scale → split into train/val/test."""

    print("[Preprocessing] Loading raw data …")
    df = pd.read_csv(raw_path)
    print(f"  Raw shape : {df.shape}")
    print(f"  Churn rate: {df['Churn'].mean():.2%}")

    # First split: hold out test set
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["Churn"])
    # Second split: carve out validation from training
    val_relative = val_size / (1 - test_size)
    train, val   = train_test_split(train_val, test_size=val_relative, random_state=random_state, stratify=train_val["Churn"])

    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train)
    X_val,   y_val   = preprocessor.transform(val)
    X_test,  y_test  = preprocessor.transform(test)

    preprocessor.save("models/preprocessor.joblib")

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    prepare_splits()
