from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "football_player_stats.csv"
PROCESSED = ROOT / "data" / "processed" / "processed_football.csv"
META = ROOT / "artifacts" / "preprocessing_metadata.json"

TARGET = "Goals"
DROP_COLUMNS = ["Rk", "Player", "G/Sh", "G/SoT"]

def preprocess():
    df = pd.read_csv(RAW, sep=";", encoding="latin1")
    original_shape = list(df.shape)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    df["Nation"] = df["Nation"].fillna("Unknown")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    # One-hot encode categorical input columns. Target remains numeric and separate.
    categorical = df.drop(columns=[TARGET]).select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical, drop_first=True, dtype=int)

    # Fill any remaining numeric missing values using each column's median.
    feature_cols = [c for c in df.columns if c != TARGET]
    missing_before = int(df[feature_cols].isna().sum().sum())
    for col in feature_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    metadata = {
        "original_shape": original_shape,
        "processed_shape": list(df.shape),
        "target": TARGET,
        "dropped_columns": dropped,
        "duplicate_rows_removed": duplicates,
        "categorical_columns_encoded": categorical,
        "encoding": "one-hot encoding with drop_first=True",
        "missing_nation": "filled with Unknown",
        "remaining_feature_missing_values_before_fill": missing_before,
        "target_type": "regression (numeric goal count)",
        "leakage_note": "G/Sh and G/SoT removed because they are derived from Goals"
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Processed data saved: {PROCESSED}")
    print(f"[INFO] Shape: {df.shape}; input features: {df.shape[1]-1}")
    return df

if __name__ == "__main__":
    preprocess()
