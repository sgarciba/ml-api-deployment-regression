import pytest
from sklearn.datasets import fetch_california_housing

from app.model import load_model, predict


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_house():
    X, _ = fetch_california_housing(return_X_y=True, as_frame=True)
    return X.iloc[0].to_dict()


def test_predict_returns_expected_keys(model, sample_house):
    result = predict(model, sample_house)
    assert set(result) == {"predicted_price_usd"}
    
    

def test_price_is_positive_float(model, sample_house):
    price = predict(model, sample_house)["predicted_price_usd"]
    assert isinstance(price, float)
    assert price > 0


def test_price_is_rounded_to_nearest_thousand(model, sample_house):
    price = predict(model, sample_house)["predicted_price_usd"]
    assert price % 1000 == 0


def test_price_is_in_plausible_range_for_known_house(model, sample_house):
    price = predict(model, sample_house)["predicted_price_usd"]
    assert 300_000 <= price <= 600_000


def test_higher_income_predicts_higher_price(model, sample_house):
    richer_area = {**sample_house, "MedInc": sample_house["MedInc"] + 3}
    base_price = predict(model, sample_house)["predicted_price_usd"]
    richer_price = predict(model, richer_area)["predicted_price_usd"]
    assert richer_price > base_price

    