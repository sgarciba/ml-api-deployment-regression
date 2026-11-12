import argparse
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "regressor.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODELS = {
    "linear": LinearRegression(),
    "hgb": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "hgb_tuned": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="hgb")
    parser.add_argument("--track", action="store_true", help="Log this run to MLflow")
    return parser.parse_args()


def log_to_mlflow(pipeline, args, mae, r_square):
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("house-price")
    
    with mlflow.start_run():
        model_params = pipeline.named_steps["model"].get_params()
        mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})
        mlflow.log_params(
            {
                "model_type": args.model,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics({"mae": mae, "r2": r_square})
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )


def main():

    args = parse_args()

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", MODELS[args.model]),
        ]
    )

    pipeline = pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r_square = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f} (~${mae * 100_000:,.0f})")
    print(f"R2: {r_square:.3f}")

    if args.track:
        log_to_mlflow(pipeline, args, mae, r_square)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
