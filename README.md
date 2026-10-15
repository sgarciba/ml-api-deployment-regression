# California House Price Predictor API

A production-style machine learning service that serves a scikit-learn regression model through a REST API built with **FastAPI**. It is tested with **pytest** and packaged with **Docker**.

This service was built from my [ML service template](#using-this-as-a-template), originally created for a breast cancer classifier, which shows that the same structure can serve different model types with minimal changes.

---

## Features

- **Single pipeline artifact**: preprocessing (`StandardScaler`) and model (`LinearRegression`) are saved together, preventing training–serving skew
- **Business-ready output**: predictions are converted from the dataset's $100,000 units into US dollars and rounded to the nearest $1,000
- **Input validation** with Pydantic: malformed or incomplete requests are rejected with a clear `422` error before reaching the model
- **Model loaded once at startup** using FastAPI's lifespan, not on every request
- **Unit and API tests** with pytest and FastAPI's `TestClient`, including behavioural tests of the model
- **Reproducible Docker image**: the model is trained during the image build, so the scikit-learn version used for training always matches the one used for serving
- **Interactive API docs**, generated automatically at `/docs`

---

## Project Structure

```
├── src/app/
│   ├── main.py          # FastAPI app: /health and /predict endpoints
│   ├── schemas.py       # Pydantic request/response models
│   ├── model.py         # Model loading and prediction logic
│   └── config.py        # Paths, model version and target units
├── training/
│   └── train.py         # Trains the pipeline and saves the artifact
├── models/              # Saved model artifacts (not tracked in git)
├── tests/
│   ├── test_model.py    # Unit tests for the prediction logic
│   └── test_api.py      # API tests for the endpoints
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Model

| | |
|---|---|
| **Dataset** | California Housing (scikit-learn), 20,640 block groups, 8 features |
| **Target** | Median house value, in units of $100,000 (capped at $500,000) |
| **Pipeline** | `StandardScaler` → `LinearRegression` |
| **Split** | 80/20, random |
| **MAE** | 0.533 (≈ **$53,000** average error) |
| **R²** | 0.576 |

**Features:**

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group (in $10,000s) |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

A linear model is used deliberately to keep the focus on the serving infrastructure. Non-linear models such as gradient boosting typically reach an R² of around 0.8 on this dataset, and the pipeline design means swapping models is a one-line change.

---

## Quickstart

### Option 1: Run with Docker (recommended)

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

Note: the dataset is downloaded during the build, so an internet connection is required.

### Option 2: Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python training/train.py         # trains and saves the model
uvicorn app.main:app --reload --app-dir src
```

Then open **http://127.0.0.1:8000/docs** to explore the API interactively.

---

## API Usage

### `GET /health`

Checks that the service is running.

```json
{ "status": "ok" }
```

### `POST /predict`

Accepts the 8 block-group features and returns the predicted median house value in US dollars.

**Request:**

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.02381,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response:**

```json
{
  "predicted_price_usd": 415000,
  "model_version": "0.1.0"
}
```

If a feature is missing or has the wrong type, the API returns `422 Unprocessable Entity` with details of the invalid field.

---

## Running Tests

```bash
python -m pytest -v
```

The test suite covers:
- **Prediction contract**: output keys, a positive float price, and rounding to the nearest $1,000
- **Known-answer test with tolerance**: a house worth $452,600 is predicted within a plausible range, which catches unit-conversion errors while still allowing the model to be retrained
- **Directional behaviour**: a higher median income predicts a higher price, as domain knowledge suggests
- **API behaviour**: health check, a valid prediction request end to end, and `422` rejection of a request with a missing feature

---

## Design Decisions

- **Why convert to dollars inside the service?** API consumers read numbers literally. Returning `4.15` would require every client to know the dataset's units; returning `415000` in a field named `predicted_price_usd` removes that ambiguity.
- **Why round to the nearest $1,000?** With an average error of around $53,000, returning a price to the cent would suggest far more precision than the model has.
- **Why snake_case fields with PascalCase aliases?** Python code follows PEP 8 (`med_inc`), while clients send the dataset's original column names (`MedInc`). Pydantic's `to_pascal` alias generator maps between the two, so the pipeline receives the exact column names it was trained on.
- **Why tolerance-based tests instead of exact values?** An exact-value test breaks every time the model is legitimately improved. Ranges and directional checks catch real bugs without blocking model updates.
- **Why train inside the Docker build?** A `.joblib` artifact may not load correctly under a different scikit-learn version. Training in the image keeps both versions identical and lets the image be built from the repository alone. For larger models, the artifact would instead be versioned in a model registry.

---

## Using This as a Template

The project separates **reusable infrastructure** from **model-specific code**, so a new model can be served by changing only a handful of files.

**Reusable (unchanged between models):**
- `src/app/main.py`: API endpoints, lifespan and model loading (only the app title changes)
- `Dockerfile`, `.dockerignore`, `pyproject.toml`
- The structure of the tests: fixtures, `TestClient` setup and the 422 validation pattern

**Model-specific (change for each new model):**

| File | What to change |
|---|---|
| `training/train.py` | Dataset, model, evaluation metrics |
| `src/app/config.py` | Artifact filename, target units |
| `src/app/model.py` | How the raw prediction is turned into the response |
| `src/app/schemas.py` | Input features and output fields |
| `tests/` | Sample data and model-specific assertions |

**Steps to create a new service:**
1. Click **"Use this template"** on GitHub to create a new repository.
2. Update the model-specific files listed above.
3. Run `python training/train.py`, then `python -m pytest -v`.
4. Build and run the Docker image.

