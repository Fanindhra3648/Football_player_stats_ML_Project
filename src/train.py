from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_football.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
TARGET = "Goals"

def train():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=180, max_depth=18, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=180, max_depth=18, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            loss="huber", random_state=42
        )
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "model": name,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
            "R2": float(r2_score(y_test, pred))
        }
        rows.append(metrics)
        fitted[name] = model
        print(f"       MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    comparison = pd.DataFrame(rows).sort_values("R2", ascending=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_goals_model.pkl")
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    (ARTIFACTS / "split_data.json").write_text(json.dumps({
        "test_size": 0.20, "random_state": 42,
        "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
        "target": TARGET, "best_model": best_name
    }, indent=2), encoding="utf-8")
    (ARTIFACTS / "feature_columns.json").write_text(json.dumps(X.columns.tolist()), encoding="utf-8")
    print(f"[SUCCESS] Best model: {best_name}")
    return best_model, best_name, X_train, X_test, y_train, y_test, comparison

if __name__ == "__main__":
    train()
