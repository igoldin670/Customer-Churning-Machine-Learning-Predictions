#!/usr/bin/env python3
"""
churn_model.py
Single-file churn modeling script for the provided telecom dataset.

Usage:
  python churn_model.py --train              # trains, evaluates, and writes churn_model.joblib
  python churn_model.py --predict PATH.csv   # predicts churn for new rows; prints CSV to stdout

The script is self-contained: it includes the sample dataset below.
You can also ignore the embedded data and pass your own CSV with the same columns.
"""

import argparse
import io
import sys
import textwrap

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib

# Embedded dataset --------------------------------------------------------------
EMBEDDED_CSV = """CustomerID,Gender,Age,Dependents,ContractType,Tenure,BillingMethod,PaymentMethod,MonthlyCharges,TotalCharges,InternetService,StreamingTV,StreamingMovies,PhoneService,MultipleLines,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,SupportCalls,EngagementLevel,DowntimeHistory,ServiceType,PromoOffer,PriceHikeNotice,Churn
C001,Male,30,No,Monthly,15,Yes,Credit Card,75.5,1100.5,Fiber optic,Yes,No,Yes,No,No,Yes,No,Yes,3,High,1,Premium,Yes,No,Yes
C002,Female,45,Yes,Two-year,24,Yes,Electronic Check,89.9,2157.6,DSL,No,Yes,Yes,Yes,Yes,No,Yes,No,1,Medium,0,Basic,No,Yes,No
C003,Male,34,No,One-year,10,No,Bank Transfer,65.0,650.0,Fiber optic,Yes,No,Yes,No,No,Yes,No,Yes,4,Low,2,Standard,Yes,No,No
C004,Female,29,No,Monthly,5,Yes,Mailed Check,55.3,276.5,None,No,No,No,No,No,No,No,No,2,Low,1,Basic,No,No,Yes
C005,Male,50,Yes,Two-year,36,No,Credit Card,99.9,3596.4,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Yes,1,High,0,Premium,No,Yes,No
C006,Female,42,No,One-year,18,No,Bank Transfer,80.93,3661.76,DSL,No,No,Yes,Yes,No,Yes,Yes,No,6,High,1,Basic,No,Yes,Yes
C007,Male,58,No,Monthly,7,Yes,Mailed Check,110.02,3364.74,None,No,No,No,Yes,Yes,No,No,Yes,6,Low,1,Standard,No,Yes,Yes
C008,Female,45,Yes,One-year,38,Yes,Credit Card,116.97,1913.28,DSL,Yes,Yes,No,No,No,No,Yes,No,1,Low,2,Basic,Yes,No,No
C009,Male,41,No,Monthly,34,No,Bank Transfer,95.99,3706.88,DSL,No,Yes,No,No,No,Yes,No,No,3,Medium,3,Premium,No,Yes,Yes
C010,Female,50,No,One-year,15,Yes,Credit Card,78.34,711.54,Fiber optic,Yes,No,No,No,No,No,No,Yes,9,High,1,Basic,Yes,Yes,Yes
C011,Female,41,Yes,One-year,12,No,Mailed Check,96.74,236.6,DSL,No,Yes,Yes,Yes,Yes,Yes,Yes,Yes,4,Medium,1,Standard,Yes,No,No
C012,Male,31,No,Two-year,18,No,Mailed Check,60.63,2673.59,DSL,No,Yes,No,No,No,Yes,Yes,No,3,Medium,4,Premium,No,Yes,No
C013,Male,18,Yes,Monthly,1,No,Mailed Check,43.9,4166.78,None,No,No,Yes,Yes,No,Yes,No,Yes,1,Medium,4,Basic,Yes,No,No
C014,Male,18,No,One-year,29,No,Electronic Check,46.95,3169.54,DSL,No,Yes,No,Yes,Yes,Yes,Yes,Yes,10,High,4,Basic,No,Yes,No
C015,Male,45,No,One-year,9,No,Mailed Check,50.23,786.98,DSL,No,Yes,No,No,No,No,Yes,No,7,High,2,Premium,Yes,No,No
C016,Female,35,No,One-year,33,No,Electronic Check,69.27,1079.47,DSL,No,Yes,No,Yes,No,No,Yes,No,8,High,5,Standard,Yes,Yes,Yes
C017,Male,49,Yes,Two-year,7,Yes,Mailed Check,45.51,4911.73,None,No,Yes,No,No,Yes,Yes,Yes,Yes,0,High,0,Standard,Yes,No,No
C018,Male,40,No,One-year,46,No,Credit Card,62.27,4836.46,DSL,Yes,No,No,Yes,Yes,No,No,No,7,Low,1,Premium,No,Yes,No
C019,Female,43,No,One-year,24,Yes,Bank Transfer,74.12,3370.35,None,Yes,Yes,Yes,Yes,No,Yes,Yes,No,7,Low,4,Standard,No,No,No
C020,Male,49,No,One-year,3,No,Bank Transfer,46.65,2206.33,DSL,No,No,Yes,Yes,Yes,No,No,No,10,High,2,Premium,No,No,Yes
C021,Female,27,No,Two-year,40,No,Mailed Check,77.19,428.86,DSL,Yes,Yes,Yes,No,Yes,No,No,Yes,0,Medium,3,Standard,No,No,Yes
C022,Female,50,No,Monthly,18,Yes,Bank Transfer,62.6,356.7,Fiber optic,Yes,No,No,No,Yes,No,Yes,No,9,High,2,Standard,No,No,No
C023,Male,49,Yes,Monthly,9,Yes,Electronic Check,59.2,2020.44,DSL,No,No,Yes,Yes,Yes,No,Yes,Yes,2,High,4,Premium,Yes,No,No
C024,Male,48,No,Two-year,42,No,Bank Transfer,87.51,335.47,Fiber optic,No,Yes,No,Yes,Yes,Yes,Yes,Yes,10,Low,1,Premium,No,No,No
C025,Female,46,No,Two-year,41,No,Credit Card,103.19,2450.45,None,Yes,Yes,No,No,Yes,Yes,Yes,Yes,0,Low,2,Basic,Yes,Yes,No
C026,Female,53,Yes,One-year,15,No,Credit Card,71.77,3244.9,None,Yes,No,No,Yes,No,No,Yes,Yes,1,High,4,Standard,No,Yes,No
C027,Female,68,No,Two-year,28,No,Mailed Check,113.77,858.02,None,No,No,Yes,No,No,Yes,Yes,No,4,High,1,Basic,Yes,No,Yes
C028,Female,32,No,Monthly,30,No,Electronic Check,48.91,2841.34,DSL,No,Yes,No,Yes,No,No,Yes,No,4,High,4,Premium,Yes,Yes,Yes
C029,Female,26,No,Monthly,27,No,Mailed Check,114.73,1774.64,None,No,Yes,Yes,Yes,No,Yes,Yes,Yes,10,Low,0,Basic,Yes,No,Yes
C030,Female,18,Yes,Monthly,47,Yes,Credit Card,82.48,983.59,None,No,No,Yes,Yes,Yes,Yes,No,Yes,7,Medium,0,Premium,Yes,No,Yes
C031,Male,39,No,One-year,40,No,Bank Transfer,118.14,1000.57,None,Yes,No,No,No,No,Yes,No,No,9,Low,3,Standard,No,Yes,No
C032,Male,36,No,Two-year,31,Yes,Electronic Check,107.95,761.65,Fiber optic,Yes,Yes,No,Yes,Yes,No,No,No,5,Medium,3,Standard,Yes,Yes,No
C033,Male,64,Yes,Two-year,34,Yes,Electronic Check,65.68,838.94,Fiber optic,No,Yes,Yes,Yes,Yes,No,Yes,Yes,3,Medium,1,Standard,Yes,Yes,No
C034,Male,26,Yes,Monthly,9,Yes,Credit Card,65.33,4101.23,None,Yes,Yes,Yes,No,Yes,No,Yes,Yes,9,Low,2,Premium,No,No,Yes
C035,Male,63,Yes,One-year,33,Yes,Credit Card,80.87,461.02,None,Yes,No,Yes,No,No,No,Yes,No,6,Medium,3,Standard,No,Yes,Yes
C036,Female,34,No,One-year,16,No,Electronic Check,85.75,2207.04,None,No,Yes,Yes,No,No,No,Yes,No,3,Low,0,Premium,No,Yes,Yes
C037,Female,21,Yes,Two-year,38,Yes,Electronic Check,51.19,1920.29,Fiber optic,Yes,Yes,Yes,No,Yes,Yes,Yes,Yes,9,Low,1,Standard,No,No,Yes
C038,Female,30,No,Two-year,41,No,Bank Transfer,41.24,3413.41,None,No,Yes,No,Yes,Yes,Yes,Yes,Yes,10,Medium,2,Basic,No,No,No
C039,Male,67,No,Two-year,25,Yes,Mailed Check,104.38,2806.04,None,No,No,No,No,No,Yes,Yes,No,9,High,1,Standard,Yes,No,Yes
C040,Female,37,No,Two-year,47,No,Electronic Check,96.68,4824.79,None,No,No,Yes,No,No,Yes,No,No,3,Medium,2,Premium,No,No,No
C041,Female,58,Yes,One-year,32,No,Mailed Check,46.09,1529.52,Fiber optic,Yes,Yes,Yes,No,No,No,No,No,8,Low,0,Premium,No,No,Yes
C042,Female,62,Yes,Two-year,24,No,Electronic Check,54.39,3723.64,DSL,Yes,Yes,No,Yes,No,No,No,Yes,8,High,0,Premium,Yes,Yes,No
C043,Male,50,Yes,Monthly,33,No,Bank Transfer,114.68,1776.67,DSL,No,No,Yes,Yes,Yes,No,Yes,Yes,0,Medium,1,Standard,No,No,No
C044,Male,30,No,Monthly,27,Yes,Bank Transfer,45.71,4365.18,DSL,No,No,No,No,No,Yes,No,Yes,2,Medium,0,Standard,No,Yes,Yes
C045,Female,45,No,One-year,44,No,Credit Card,87.62,1279.7,Fiber optic,Yes,No,No,No,Yes,Yes,No,Yes,7,Low,4,Standard,No,No,Yes
C046,Male,24,Yes,One-year,17,No,Mailed Check,107.26,4683.6,None,No,Yes,No,Yes,Yes,Yes,Yes,Yes,1,Medium,5,Premium,Yes,Yes,Yes
C047,Female,65,Yes,Monthly,46,Yes,Credit Card,82.07,1388.34,None,Yes,Yes,No,Yes,No,Yes,Yes,No,10,Medium,4,Standard,No,Yes,Yes
C048,Female,53,Yes,Two-year,46,Yes,Bank Transfer,114.76,2174.48,None,No,Yes,Yes,No,Yes,Yes,No,Yes,1,Low,2,Standard,No,No,No
C049,Male,30,Yes,Monthly,34,Yes,Mailed Check,60.8,3901.49,None,Yes,Yes,No,Yes,No,Yes,No,No,0,High,0,Basic,Yes,Yes,No
C050,Female,64,Yes,One-year,41,No,Credit Card,40.67,1017.83,DSL,Yes,Yes,No,Yes,No,No,Yes,Yes,3,Medium,4,Basic,Yes,Yes,Yes
C051,Male,61,Yes,One-year,36,No,Credit Card,51.71,4121.11,Fiber optic,Yes,Yes,Yes,Yes,No,No,Yes,Yes,6,High,3,Standard,Yes,No,Yes
C052,Male,45,Yes,One-year,29,Yes,Credit Card,48.27,3152.48,Fiber optic,Yes,Yes,No,Yes,Yes,Yes,No,Yes,0,High,3,Standard,No,No,Yes
C053,Male,56,No,Monthly,3,No,Bank Transfer,115.67,510.79,DSL,No,No,No,No,No,No,No,No,9,High,3,Premium,Yes,Yes,No
C054,Female,56,Yes,Two-year,48,No,Credit Card,89.2,1749.56,DSL,No,Yes,No,Yes,Yes,No,No,No,2,High,2,Basic,Yes,No,Yes
C055,Female,63,No,One-year,23,No,Electronic Check,65.42,1945.04,None,No,Yes,Yes,No,Yes,Yes,Yes,Yes,3,High,2,Standard,No,No,Yes
C056,Male,29,No,Monthly,24,Yes,Credit Card,72.14,398.8,DSL,No,Yes,No,Yes,No,No,Yes,No,10,Low,3,Standard,No,Yes,No
C057,Female,29,Yes,Two-year,41,Yes,Credit Card,88.5,4603.61,Fiber optic,No,No,Yes,Yes,Yes,No,Yes,Yes,2,Low,4,Premium,No,Yes,No
C058,Male,39,Yes,One-year,21,Yes,Credit Card,45.77,3602.25,DSL,Yes,Yes,Yes,No,No,No,Yes,Yes,9,Low,0,Premium,Yes,No,No
C059,Female,41,No,Two-year,27,Yes,Mailed Check,94.82,4566.23,DSL,Yes,No,Yes,No,No,No,No,Yes,5,Low,5,Premium,Yes,No,No
C060,Female,59,Yes,One-year,15,No,Mailed Check,89.36,1055.38,Fiber optic,Yes,Yes,No,Yes,Yes,No,No,Yes,2,High,1,Standard,Yes,Yes,Yes
C061,Female,25,No,One-year,25,No,Mailed Check,119.13,2472.17,Fiber optic,Yes,Yes,No,Yes,No,No,Yes,No,0,High,0,Standard,Yes,No,Yes
C062,Female,42,Yes,Monthly,9,No,Credit Card,76.17,3387.26,None,No,No,No,No,No,Yes,Yes,Yes,8,Medium,3,Premium,No,Yes,Yes
C063,Male,49,No,Monthly,39,Yes,Bank Transfer,67.31,1034.89,DSL,Yes,Yes,Yes,No,No,No,Yes,No,0,High,3,Standard,No,Yes,No
C064,Female,49,Yes,Two-year,20,No,Credit Card,50.99,658.18,DSL,Yes,Yes,No,No,Yes,No,Yes,Yes,2,Medium,2,Premium,Yes,Yes,Yes
C065,Male,43,No,Monthly,14,No,Mailed Check,66.66,3833.86,None,No,No,No,No,Yes,Yes,Yes,No,7,Medium,5,Basic,No,No,No
C066,Female,65,No,One-year,35,Yes,Credit Card,49.31,311.27,DSL,No,Yes,No,No,No,No,Yes,Yes,8,Medium,5,Premium,Yes,No,No
C067,Male,29,No,One-year,21,Yes,Electronic Check,118.04,600.41,DSL,Yes,No,Yes,No,Yes,Yes,No,Yes,6,Medium,3,Standard,No,No,Yes
C068,Female,23,Yes,One-year,28,No,Bank Transfer,106.65,3145.47,DSL,Yes,No,No,No,Yes,Yes,No,No,6,Medium,4,Premium,Yes,Yes,Yes
C069,Female,67,No,Monthly,30,Yes,Mailed Check,115.17,4117.86,DSL,Yes,No,No,Yes,Yes,Yes,No,No,10,High,4,Premium,No,No,No
C070,Female,61,Yes,Two-year,26,Yes,Bank Transfer,86.36,1315.18,DSL,No,No,No,No,No,No,Yes,Yes,6,High,1,Basic,Yes,Yes,No
C071,Female,27,No,Two-year,10,No,Electronic Check,45.27,2387.55,Fiber optic,Yes,Yes,Yes,Yes,Yes,No,Yes,Yes,8,High,4,Standard,No,Yes,Yes
C072,Female,64,No,Monthly,19,Yes,Credit Card,77.39,3666.53,None,No,No,Yes,Yes,Yes,Yes,No,Yes,5,Medium,1,Standard,No,No,Yes
C073,Male,44,Yes,One-year,35,No,Electronic Check,63.72,3112.85,Fiber optic,No,Yes,No,Yes,Yes,Yes,No,No,5,Medium,1,Premium,Yes,Yes,Yes
C074,Male,41,No,One-year,46,Yes,Electronic Check,96.57,3119.46,Fiber optic,Yes,Yes,No,Yes,Yes,No,No,No,1,High,3,Standard,Yes,No,No
C075,Male,35,No,One-year,42,No,Credit Card,48.4,893.42,Fiber optic,No,Yes,Yes,No,Yes,Yes,No,Yes,9,High,1,Basic,No,Yes,No
C076,Female,54,No,One-year,46,Yes,Credit Card,102.28,2260.69,None,No,Yes,No,No,No,No,Yes,Yes,8,High,3,Basic,No,No,Yes
C077,Female,27,Yes,Two-year,36,Yes,Electronic Check,80.24,3831.04,Fiber optic,No,Yes,No,No,Yes,No,Yes,Yes,1,Medium,3,Basic,Yes,No,No
C078,Male,57,Yes,One-year,28,Yes,Bank Transfer,61.16,564.46,None,Yes,Yes,No,Yes,No,Yes,Yes,No,2,Medium,2,Premium,No,No,Yes
C079,Male,54,Yes,Two-year,35,Yes,Mailed Check,117.44,4864.42,DSL,No,Yes,Yes,No,Yes,Yes,Yes,Yes,4,High,0,Basic,Yes,Yes,Yes
C080,Male,68,No,Monthly,20,No,Electronic Check,66.76,3636.42,DSL,Yes,No,Yes,No,Yes,No,Yes,Yes,2,Low,2,Basic,Yes,Yes,No
C081,Male,57,Yes,Monthly,15,Yes,Mailed Check,62.81,4271.12,Fiber optic,No,Yes,No,No,Yes,No,Yes,No,6,Medium,1,Premium,No,Yes,No
C082,Female,18,No,Monthly,10,No,Credit Card,90.48,4836.95,DSL,Yes,Yes,Yes,Yes,No,Yes,No,No,0,Low,5,Standard,No,Yes,Yes
C083,Female,57,Yes,One-year,47,Yes,Credit Card,40.14,3916.48,Fiber optic,No,Yes,No,No,Yes,Yes,Yes,No,1,High,5,Basic,Yes,Yes,No
C084,Female,55,No,Two-year,6,No,Electronic Check,88.93,2816.62,Fiber optic,Yes,No,Yes,Yes,Yes,No,Yes,No,3,High,0,Standard,No,No,No
C085,Female,22,No,Two-year,25,No,Electronic Check,76.75,602.84,Fiber optic,No,Yes,No,No,No,Yes,No,No,7,High,3,Standard,No,Yes,No
C086,Female,39,No,Monthly,23,No,Credit Card,66.89,4894.91,Fiber optic,No,No,Yes,Yes,Yes,No,Yes,Yes,8,Low,3,Standard,Yes,No,Yes
C087,Female,29,Yes,One-year,46,Yes,Bank Transfer,75.95,951.97,Fiber optic,Yes,No,Yes,No,Yes,Yes,Yes,Yes,2,Medium,1,Basic,Yes,Yes,No
C088,Female,26,No,One-year,2,No,Electronic Check,107.03,2706.86,None,Yes,Yes,Yes,No,Yes,No,No,No,10,Medium,3,Standard,Yes,Yes,No
C089,Male,44,No,One-year,24,Yes,Electronic Check,81.88,2342.83,None,No,No,Yes,No,Yes,Yes,Yes,Yes,4,Low,0,Basic,No,No,Yes
C090,Female,23,Yes,Two-year,18,No,Bank Transfer,60.08,408.97,Fiber optic,Yes,Yes,Yes,No,No,Yes,Yes,No,0,Low,0,Basic,No,Yes,Yes
C091,Female,20,Yes,One-year,32,Yes,Credit Card,112.75,4609.76,Fiber optic,No,No,Yes,No,No,No,No,Yes,1,High,1,Standard,No,No,Yes
C092,Female,22,No,Monthly,47,Yes,Bank Transfer,52.49,3842.5,DSL,Yes,Yes,No,No,No,No,No,Yes,4,Medium,5,Standard,Yes,No,Yes
C093,Female,57,No,Monthly,30,No,Bank Transfer,69.94,3235.52,Fiber optic,No,Yes,No,No,Yes,Yes,Yes,No,1,High,0,Basic,No,No,No
C094,Male,37,No,One-year,36,No,Electronic Check,112.69,4524.08,None,Yes,No,No,No,No,Yes,No,Yes,3,High,3,Premium,No,Yes,Yes
C095,Male,47,Yes,One-year,2,No,Credit Card,111.91,4777.03,Fiber optic,Yes,Yes,Yes,No,Yes,No,Yes,Yes,3,Low,3,Basic,Yes,No,No
C096,Male,26,No,One-year,35,No,Credit Card,74.13,4317.81,None,No,No,No,Yes,Yes,No,Yes,Yes,0,High,0,Basic,No,No,No
C097,Male,53,No,One-year,11,No,Bank Transfer,95.63,4471.37,DSL,Yes,Yes,No,Yes,No,Yes,No,No,3,Medium,1,Premium,No,Yes,Yes
C098,Female,64,No,Two-year,17,No,Mailed Check,42.34,2147.63,DSL,No,Yes,Yes,Yes,No,Yes,No,Yes,4,Medium,5,Standard,No,No,Yes
C099,Female,56,Yes,Monthly,18,Yes,Credit Card,112.55,3457.25,None,No,Yes,No,No,Yes,Yes,No,No,10,Medium,1,Basic,Yes,Yes,No
C100,Male,65,No,Monthly,10,No,Mailed Check,86.14,4394.59,Fiber optic,Yes,Yes,No,No,Yes,Yes,Yes,Yes,6,Medium,3,Premium,No,Yes,No
C101,Female,51,No,One-year,5,Yes,Electronic Check,83.34,3477.83,None,Yes,No,No,Yes,No,Yes,Yes,Yes,2,High,1,Standard,No,No,Yes
C102,Female,64,Yes,Monthly,9,Yes,Electronic Check,63.49,1946.72,None,Yes,No,Yes,Yes,Yes,No,Yes,Yes,8,Medium,2,Basic,Yes,Yes,No
C103,Female,46,No,One-year,20,No,Mailed Check,88.24,494.32,Fiber optic,No,Yes,Yes,Yes,Yes,Yes,No,No,9,Medium,4,Basic,No,No,Yes
C104,Female,60,No,One-year,47,Yes,Bank Transfer,103.74,4744.06,DSL,Yes,Yes,No,No,No,Yes,No,Yes,0,Low,3,Basic,Yes,Yes,No
C105,Female,29,No,One-year,24,No,Mailed Check,64.4,1696.26,None,No,Yes,No,No,Yes,Yes,Yes,No,7,Low,3,Basic,No,Yes,No
"""

YESNO = {"Yes": 1, "No": 0}

TARGET = "Churn"

DROP = ["CustomerID"]  # identifiers to drop


def load_dataframe(path: str | None) -> pd.DataFrame:
    if path is None:
        df = pd.read_csv(io.StringIO(EMBEDDED_CSV))
    else:
        df = pd.read_csv(path)
    # Coerce numeric columns that may look like strings
    for col in ["MonthlyCharges", "TotalCharges", "Age", "Tenure", "SupportCalls", "DowntimeHistory"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Map Yes/No to 1/0 across the frame where applicable
    for col in df.columns:
        if df[col].dtype == "object":
            # If values are exactly subset of Yes/No, map
            uniques = set(str(x) for x in df[col].dropna().unique())
            if uniques.issubset({"Yes", "No"}):
                df[col] = df[col].map(YESNO).astype("float")
    return df


def split_xy(df: pd.DataFrame):
    y = df[TARGET].map(YESNO) if df[TARGET].dtype == "object" else df[TARGET]
    X = df.drop(columns=[TARGET] + [c for c in DROP if c in df.columns])
    return X, y.astype(int)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identify column types after Yes/No mapping
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Simple and strong baseline for small tabular data
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
    return pipe


def train_and_eval(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    X, y = split_xy(df)
    pipe = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "n_test": int(len(y_test)),
    }

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, pred))

    # Persist the full pipeline
    joblib.dump(pipe, "churn_model.joblib")
    print("\nWrote churn_model.joblib")

    return pipe, (X_test, y_test), metrics


def predict_file(model_path: str, data_path: str):
    pipe: Pipeline = joblib.load(model_path)
    df_new = load_dataframe(data_path)

    # Drop target if present
    if TARGET in df_new.columns:
        df_new = df_new.drop(columns=[TARGET])
    if "CustomerID" in df_new.columns:
        ids = df_new["CustomerID"].astype(str).values
    else:
        ids = np.arange(len(df_new)).astype(str)

    # Align columns naming as in training preprocessor
    # The pipeline can handle differences thanks to imputer+OHE
    proba = pipe.predict_proba(df_new)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = pd.DataFrame({
        "ID": ids,
        "churn_probability": proba,
        "churn_pred": pred
    })
    out.to_csv(sys.stdout, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Churn model trainer and predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(\"\"\"
        Examples:
          # Train on embedded data and write model to churn_model.joblib
          python churn_model.py --train

          # Predict on a new CSV file with the same columns (Churn optional)
          python churn_model.py --predict my_new_data.csv > predictions.csv
        \"\"\"),
    )
    parser.add_argument("--train", action="store_true", help="Train and evaluate on the embedded dataset")
    parser.add_argument("--predict", metavar="PATH", help="Predict churn for new rows in PATH CSV")
    parser.add_argument("--model", default="churn_model.joblib", help="Path to .joblib model")

    args = parser.parse_args()

    if args.train:
        df = load_dataframe(None)
        train_and_eval(df)
    elif args.predict:
        predict_file(args.model, args.predict)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
