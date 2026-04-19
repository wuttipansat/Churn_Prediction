from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_data

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TEST_DATA_PATH = ARTIFACT_DIR / "test_sample.csv"

def get_model(model_name: str):
    """Select Basic model including Logistic Regression and Random Forest Classification."""
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    return ValueError(f"Unsupported model: {model_name}")

def build_pipeline(X, model_name: str = "logreg") -> Pipeline:
    """Build pipeline including numeric and categorical columns transformer and selected model."""
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

    model = get_model(model_name)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

def train(
        csv_path: str = "data/dataset.csv",
        target_column: str = "label",
        model_name: str = "logreg",
) -> dict:
    """Train the model."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(csv_path=csv_path, target_column=target_column)
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = build_pipeline(X, model_name=model_name)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "csv_path": csv_path,
        "target_column": target_column,
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": X.columns.tolist(),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    sample = X_test.head(5).copy()
    sample['label'] = y_test.head(5).copy()
    sample.to_csv(TEST_DATA_PATH, index=False)

    return metrics


def parse_args():
    """Parse arguments including csv file, target column, and model."""
    parser = argparse.ArgumentParser(description="Train ML model from CSV")
    parser.add_argument(
        "--csv-path",
        type=str,
        default='data/dataset.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default='label',
        help='Name of target column in CSV'
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=['logreg', 'rf'],
        default='logreg',
        help='Model to train'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    result = train(
        csv_path=args.csv_path,
        target_column=args.target_column,
        model_name=args.model_name,
    )

    print(json.dumps(result, indent=2))