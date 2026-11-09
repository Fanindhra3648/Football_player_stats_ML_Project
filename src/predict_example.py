from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def predict_from_processed_row(row_index=0):
    """Demonstration: predict Goals for one row in the encoded dataset."""
    data = pd.read_csv(ROOT / "data/processed/processed_football.csv")
    X = data.drop(columns=["Goals"])
    model = joblib.load(ROOT / "models/best_goals_model.pkl")
    prediction = float(model.predict(X.iloc[[row_index]])[0])
    print(f"Predicted goals for processed row {row_index}: {prediction:.2f}")
    return prediction

if __name__ == "__main__":
    predict_from_processed_row(0)
