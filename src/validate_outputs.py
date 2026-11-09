from pathlib import Path
import json
import pandas as pd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def validate_outputs():
    errors = []
    processed = ROOT / "data/processed/processed_football.csv"
    model_path = ROOT / "models/best_goals_model.pkl"
    if not processed.exists(): errors.append("Processed dataset is missing.")
    if not model_path.exists(): errors.append("Trained model is missing.")
    if processed.exists():
        df = pd.read_csv(processed)
        if "Goals" not in df: errors.append("Target column Goals is missing.")
        elif df["Goals"].isna().any(): errors.append("Target contains missing values.")
        if df.drop(columns=["Goals"], errors="ignore").isna().sum().sum():
            errors.append("Feature matrix still contains missing values.")
    if model_path.exists():
        model = joblib.load(model_path)
        if not hasattr(model, "predict"): errors.append("Saved object does not support predict().")
    report = {"status": "PASSED" if not errors else "FAILED", "errors": errors}
    out = ROOT / "artifacts/output_validation_report.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{'SUCCESS' if not errors else 'ERROR'}] Output validation {report['status']}")
    for error in errors: print(" -", error)
    return not errors

if __name__ == "__main__":
    validate_outputs()
