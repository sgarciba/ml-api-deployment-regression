from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "regressor.joblib"
MODEL_VERSION = "1.0.0"  # house-price-regressor v1 (MLflow registry)

TARGET_UNIT_USD = 100_000  # California Housing target is in $100,000s
