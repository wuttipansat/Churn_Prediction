from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_data(
        csv_path: str | Path = "data/dataset.csv",
        target_column: str = "label",
) -> pd.DataFrame:
    """Load dataset from a CSV file and normalize target column to 'label'."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"CSV must contain target column: {target_column}")

    if df[target_column].dtype == "object":
        unique_values = set(df[target_column].dropna().astype(str).unique())
        if unique_values <= {"Yes", "No"}:
            df[target_column] = df[target_column].map({"No": 0, "Yes": 1})
   
    if target_column != "label":
        df = df.rename(columns={target_column: "label"})

    return clean_data(df)

def clean_data(df) -> pd.DataFrame:
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"], errors="ignore")

    df = df.drop_duplicates()

    return df

