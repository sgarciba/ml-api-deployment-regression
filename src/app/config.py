from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "regressor.joblib"
MODEL_VERSION = "0.1.0"

TARGET_UNIT_USD = 100_000   # California Housing target is in $100,000s