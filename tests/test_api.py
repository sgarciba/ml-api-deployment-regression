import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import fetch_california_housing

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    

@pytest.fixture(scope="module")
def valid_payload():
    X, _ = fetch_california_housing(return_X_y=True, as_frame=True)
    return X.iloc[0].to_dict() 


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predict_valid_request(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert set(response.json()) == {"predicted_price_usd", "model_version"}


def test_predict_missing_feature_returns_422(client, valid_payload):

    payload = valid_payload.copy()
    del payload["MedInc"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "MedInc" in response.text