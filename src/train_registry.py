from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def train_and_log():
    df = pd.read_csv(ROOT / "data/processed/processed_football.csv")
    X, y = df.drop(columns=["Goals"]), df["Goals"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Football_Goals_Prediction")
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, loss="huber", random_state=42)
    }
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mlflow.log_param("model_family", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics({
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "R2": r2_score(y_test, pred)
            })
            mlflow.sklearn.log_model(model, name="goals_regression_model")
            print(f"[SUCCESS] MLflow logged {name}")

if __name__ == "__main__":
    train_and_log()
