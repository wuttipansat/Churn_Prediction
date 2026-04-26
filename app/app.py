from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from features import add_features

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title = "Customer Churn Prediction API",
    description ="API for predicting customer churn using the trained sklearn pipeline.",
    version = "1.0.0"
)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description="List of raw customer records. Extra columns are allowed and ignored by the trained pipeline.",
    )


class PredictionResult(BaseModel):
    index: int
    churn_prediction: int
    churn_probability: float
    risk_segment: str


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]


REQUIRED_RAW_COLUMNS = [
    "TotalCharges",
    "tenure",
    "Partner",
    "Dependents",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
]


def assign_risk_segment(probability: float) -> str:
    """
    Convert churn probability into a business-friendly risk group.
    Adjust thresholds if needed.
    """
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate only raw columns required by add_features().

    Do not validate engineered columns because the pipeline creates them itself.
    Extra columns are allowed.
    """
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Input records cannot be empty.",
        )

    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing raw columns required by add_features().",
                "missing_columns": missing_cols,
                "note": "Do not send engineered columns such as avg_charge or tenure_group. The pipeline creates them automatically.",
            },
        )


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "pipeline_steps": list(model.named_steps.keys())
        if hasattr(model, "named_steps")
        else None,
        "note": "The API accepts raw customer columns. Engineered columns are created inside the saved pipeline.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame(request.records)

    validate_input(df)

    X = df.drop(columns=["label", "Churn"], errors="ignore")

    try:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(e),
            },
        )

    results: list[PredictionResult] = []

    for i, probability in enumerate(probabilities):
        results.append(
            PredictionResult(
                index=i,
                churn_prediction=int(predictions[i]),
                churn_probability=round(float(probability), 4),
                risk_segment=assign_risk_segment(float(probability)),
            )
        )

    return PredictResponse(predictions=results)