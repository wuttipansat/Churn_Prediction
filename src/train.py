from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

from data import load_data
from features import build_preprocessor, add_features
from models import get_models

RANDOM_STATE = 30
ARTIFACT_DIR = "artifacts"
OUTPUT_DIR = "outputs"
DATA_FILE = "./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"





def train(
        csv_path: str = "data/dataset.csv",
        target_column: str = "label",
        random_state: int = RANDOM_STATE,
        artifact: str = ARTIFACT_DIR,
        output: str = OUTPUT_DIR
) -> dict:
    """Train the model."""
    artifact_dir = Path(artifact)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    test_sample_path = artifact_dir / "test_sample.csv"

    df = load_data(csv_path=Path(csv_path), target_column=target_column)

    X = df.drop(columns=['label'])
    y = df['label']

    models = get_models(random_state)
    results = []
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineering", FunctionTransformer(add_features)),
                ("preprocessor", build_preprocessor()),
                ("model", model)
            ]
        )
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1

        )

        result = {
            "model": name,
            "accuracy": round(cv_results["test_accuracy"].mean(), 4),
            "f1": round(cv_results["test_f1"].mean(), 4),
            "roc_auc": round(cv_results["test_roc_auc"].mean(), 4),
            "roc_auc_std": round(cv_results["test_roc_auc"].std(), 4)
        }

        results.append(result)

    best_result = sorted(results, key=lambda x: x["roc_auc"], reverse=True)[0]
    best_model_name = best_result['model']
    
    models = get_models(random_state)
    best_model = models[best_model_name]

    final_pipeline = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_features)),
            ('preprocessor', build_preprocessor()),
            ("model", best_model)
        ]
    )

    final_pipeline.fit(X, y)

    model_step = final_pipeline.named_steps["model"]
    if hasattr(model_step, "predict_proba"):
        y_prob = final_pipeline.predict_proba(X)[:, 1]
    else:
        y_prob = final_pipeline.decision_function(X)

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    joblib.dump(final_pipeline, model_path)

    df_results = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    metrics = {
    "csv_path": csv_path,
    "random_state": random_state,
    "target_column": target_column,
    "results": sorted(results, key=lambda x: x['roc_auc'], reverse=True),
    "n_rows": int(len(X))
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    sample = X.head(5).copy()
    sample['label'] = y.head(5).copy()
    sample.to_csv(test_sample_path, index=False)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.axis('off')

    table = ax.table(cellText=df_results.values,
                     colLabels=df_results.columns,
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_results.columns))))

    fig_path = output_dir / "model_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)

    plt.close()

    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='blue', linewidth=2.0)
    plt.plot([0, 1], [0, 1], linestyle="--", color='black', linewidth=2.0)

    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(f"ROC Curve ({best_model_name})", fontsize=20)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(loc="lower right")

    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight", dpi=300)
    plt.close()

    return metrics


def parse_args():
    """Parse arguments including csv file, target column, and model."""
    parser = argparse.ArgumentParser(description="Train ML model from CSV")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DATA_FILE,
        help='Path to input CSV file'
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default='Churn',
        help='Name of target column in CSV'
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help='random state'
    )

    parser.add_argument(
        "--artifact",
        type=str,
        default=ARTIFACT_DIR,
        help='artifact directory'
    )


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    result = train(
        csv_path=args.csv_path,
        target_column=args.target_column,
        random_state=args.random_state,
        artifact=args.artifact
    )
    

    print(json.dumps(result, indent=2))