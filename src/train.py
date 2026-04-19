from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.features import build_preprocessor
from src.models import get_model

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"



def build_pipeline(X, random_state: int, model_name: str = "logreg") -> Pipeline:
    """Build pipeline including numeric and categorical columns transformer and selected model."""
    
    preprocessor = build_preprocessor(X)
    model = get_model(model_name, random_state)

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
        test_size: float = 0.2,
        random_state: int = RANDOM_STATE,
        output_dir: str = ARTIFACT_DIR
) -> dict:
    """Train the model."""
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    test_sample_path = artifact_dir / "test_sample.csv"

    df = load_data(csv_path=csv_path, target_column=target_column)
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = build_pipeline(X, random_state, model_name=model_name)
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

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    sample = X_test.head(5).copy()
    sample['label'] = y_test.head(5).copy()
    sample.to_csv(test_sample_path, index=False)

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

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help='Test split size'
    )

    parser.add_argument(
        "--random-state",
        type=float,
        default=RANDOM_STATE,
        help='random state'
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=ARTIFACT_DIR,
        help='output directory'
    )


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    result = train(
        csv_path=args.csv_path,
        target_column=args.target_column,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

    print(json.dumps(result, indent=2))