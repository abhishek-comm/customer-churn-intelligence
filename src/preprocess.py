"""Data loading and reusable preprocessing for the churn project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV, normalize mixed types, and return features plus a binary target."""
    df = pd.read_csv(path)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # TotalCharges is mixed text/numeric in the source data; invalid blanks become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    X = df.drop(columns=[TARGET_COLUMN, *ID_COLUMNS])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create the training/inference transformer used inside every model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def clean_feature_name(name: str) -> str:
    """Make sklearn's transformer-prefixed names easier to read in the UI."""
    return name.replace("numeric__", "").replace("categorical__", "")
