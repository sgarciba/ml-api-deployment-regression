import joblib
import pandas as pd

from app.config import MODEL_PATH, TARGET_UNIT_USD


def load_model():
    # 1: load the pipeline from MODEL_PATH and return it
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def predict(model, features: dict) -> dict:
    # 2: turn the features dict into a one-row DataFrame
    #         (hint: wrap the dict in a list)
    X = pd.DataFrame([features])
    
    # 3: get the prediction (a single value, in $100,000s)
    y_pred = model.predict(X)[0]
    y_pred_usd = round(y_pred * TARGET_UNIT_USD, -3)

    return {
        "predicted_price_usd": float(y_pred_usd)
    }