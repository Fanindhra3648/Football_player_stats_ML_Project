from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "football_player_stats.csv"
REPORT = ROOT / "artifacts" / "data_validation_report.json"

def validate_data():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")
    df = pd.read_csv(RAW, sep=";", encoding="latin1")
    issues = []
    if "Goals" not in df.columns:
        issues.append("Required target column 'Goals' is missing.")
    if df["Goals"].isna().any():
        issues.append("Target column contains missing values.")
    if (pd.to_numeric(df["Goals"], errors="coerce") < 0).any():
        issues.append("Target column contains negative goal counts.")
    report = {
        "status": "PASSED" if not issues else "FAILED",
        "shape": list(df.shape),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "target": "Goals",
        "target_summary": df["Goals"].describe().to_dict() if "Goals" in df else {},
        "issues": issues
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[{'SUCCESS' if not issues else 'ERROR'}] Data validation {report['status']}")
    print(f"[INFO] Dataset shape: {df.shape}; duplicates: {report['duplicate_rows']}")
    return not issues

if __name__ == "__main__":
    validate_data()
