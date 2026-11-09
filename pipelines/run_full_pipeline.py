from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validate_data import validate_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.validate_outputs import validate_outputs
from src.validate_reproducibility import run_test

def main():
    print("\n========== FOOTBALL GOALS PREDICTION PIPELINE ==========")
    if not validate_data():
        raise SystemExit("Data validation failed. See artifacts/data_validation_report.json")
    preprocess()
    train()
    evaluate()
    validate_outputs()
    run_test()
    print("\n[SUCCESS] Complete project pipeline finished.")
    print("Check outputs/ and artifacts/ for reports and plots.")

if __name__ == "__main__":
    main()
