from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_football.csv"
MODEL = ROOT / "models" / "best_goals_model.pkl"
OUT = ROOT / "outputs"
TARGET = "Goals"

def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = joblib.load(MODEL)
    pred = model.predict(X_test)
    metrics = {
        "model": type(model).__name__,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
        "test_samples": int(len(y_test))
    }
    (OUT / "regression_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"Actual_Goals": y_test.to_numpy(), "Predicted_Goals": pred}).to_csv(
        OUT / "predictions.csv", index=False
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.55)
    low, high = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Goals"); plt.ylabel("Predicted Goals")
    plt.title("Actual vs Predicted Goals")
    plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close()

    residuals = y_test.to_numpy() - pred
    plt.figure(figsize=(7, 5))
    plt.scatter(pred, residuals, alpha=0.55)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Goals"); plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout(); plt.savefig(OUT / "residual_plot.png", dpi=160); plt.close()

    # 5-fold CV on the full feature matrix (R2).
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    (OUT / "cross_validation_results.txt").write_text(
        f"5-fold R2 scores: {cv_scores.tolist()}\nMean CV R2: {float(cv_scores.mean()):.6f}\n",
        encoding="utf-8"
    )
    if hasattr(model, "feature_importances_"):
        pd.Series(model.feature_importances_, index=X.columns).sort_values(
            ascending=False
        ).to_csv(OUT / "feature_importance.csv", header=["importance"])
    (OUT / "evaluation_report.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in metrics.items()), encoding="utf-8"
    )
    print("[SUCCESS] Evaluation completed.")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    evaluate()
