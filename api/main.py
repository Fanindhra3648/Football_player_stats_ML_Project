
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_goals_model.pkl"
FEATURE_PATH = ROOT / "artifacts" / "feature_columns.json"
META_PATH = ROOT / "artifacts" / "preprocessing_metadata.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "processed_football.csv"
RAW_PATH = ROOT / "data" / "raw" / "football_player_stats.csv"

# Load trained model and artifacts
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Training data supplies the exact numerical medians
processed_df = pd.read_csv(PROCESSED_PATH)
medians = processed_df[feature_columns].median().fillna(0)

# Original dataset supplies valid input column names
raw_df = pd.read_csv(RAW_PATH, sep=";", encoding="latin1")

TARGET = "Goals"
DROP_COLUMNS = ["Rk", "Player", "G/Sh", "G/SoT"]

categorical_columns = metadata["categorical_columns_encoded"]
allowed_columns = [
    col for col in raw_df.columns
    if col not in DROP_COLUMNS + [TARGET]
]

app = FastAPI(
    title="Football Goals Prediction API",
    description="Predict football player goals using a trained Gradient Boosting model.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    player_stats: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>Football Goals Prediction</title></head>
      <body style="font-family:Arial;text-align:center;padding:50px">
        <h1>Football Goals Prediction API</h1>
        <p>Gradient Boosting Regression Model</p>
        <p>Server is running successfully!</p>
        <a href="/docs">Open Swagger UI and test predictions</a>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "features": len(feature_columns)
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        supplied = request.player_stats

        unknown = set(supplied) - set(allowed_columns)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown input columns: {sorted(unknown)}"
            )

        # Start with all expected original input columns
        row = {col: None for col in allowed_columns}
        row.update(supplied)

        if "Nation" in row and row["Nation"] is None:
            row["Nation"] = "Unknown"

        input_df = pd.DataFrame([row])

        # Match the training preprocessing
        encoded = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=True,
            dtype=int
        )

        # Convert numeric columns and fill missing values
        for col in encoded.columns:
            if col not in categorical_columns:
                encoded[col] = pd.to_numeric(
                    encoded[col], errors="coerce"
                )

        encoded = encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Fill missing numeric features with training medians
        encoded = encoded.fillna(medians).fillna(0)

        prediction = float(model.predict(encoded)[0])

        return {
            "success": True,
            "predicted_goals": round(max(0.0, prediction), 2),
            "model": type(model).__name__
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
