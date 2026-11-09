from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import preprocess
from src.train_registry import train_and_log

if __name__ == "__main__":
    preprocess()
    train_and_log()
