from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

def build_preprocessor(X):
    """Transform X data into numeric and categorical features for preprocessing."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),

    ])

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)    
        ]
    )

    return preprocessor

def add_features(feat) -> pd.DataFrame:
    """Add feature engineering."""
    feat['TotalCharges'] = pd.to_numeric(feat['TotalCharges'], errors='coerce')
    feat['tenure'] = pd.to_numeric(feat['tenure'], errors='coerce')
    #average charge per month
    feat['avg_charge'] = feat["TotalCharges"] / (feat['tenure'] + 1)

    # Separate tenure into groups
    feat['tenure_group'] = pd.cut(
        feat['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['new', 'mid', 'loyal', 'very loyal'],
    )

    # Family dependents
    feat['has_family'] = ((feat['Partner'] == 'Yes') | (feat['Dependents'] == 'Yes')).astype(int)

    # Extra services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    feat['addon_count'] = feat[addon_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
    feat['normalize_addon'] = feat['addon_count'] / len(addon_cols)

    # Contract mapping
    contract_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0
    }
    feat['contract_risk'] = feat['Contract'].map(contract_map)

    return feat

