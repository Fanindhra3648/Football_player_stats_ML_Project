from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]

def run_test():
    df = pd.read_csv(ROOT / "data/processed/processed_football.csv")
    X, y = df.drop(columns=["Goals"]), df["Goals"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Use a compact deterministic model for repeatability check.
    params = {"n_estimators": 60, "max_depth": 8, "random_state": 42, "n_jobs": -1}
    scores = []
    for _ in range(2):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        scores.append(float(mean_absolute_error(y_test, model.predict(X_test))))
    passed = bool(np.isclose(scores[0], scores[1], rtol=0, atol=1e-12))
    report = {"test_name": "Deterministic training check", "parameters": params,
              "execution_1_MAE": scores[0], "execution_2_MAE": scores[1],
              "is_reproducible": passed, "status": "PASSED" if passed else "FAILED"}
    out = ROOT / "artifacts/reproducibility_report.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{'SUCCESS' if passed else 'ERROR'}] Reproducibility: {report['status']}")
    return passed

if __name__ == "__main__":
    run_test()
