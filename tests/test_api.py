"""API contract tests with a stubbed champion bundle."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from causalift.api.main import create_app
from causalift.models import predict


class StubModel:
    name = "x_learner"

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        # Recent, engaged customers are persuadable; long-lapsed are sleeping dogs.
        return np.where(x["recency"] < 4, 0.08, np.where(x["recency"] > 12, -0.05, 0.0))


@pytest.fixture()
def client(monkeypatch):
    bundle = {
        "model": StubModel(),
        "features": ["recency", "history", "mens", "womens", "newbie"],
        "name": "x_learner",
    }
    predict.load_champion.cache_clear()
    monkeypatch.setattr(predict, "load_champion", lambda: bundle)
    return TestClient(create_app())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "x_learner"
    assert body["n_features"] == 5


def test_score_classifies_persuadable_and_sleeping_dog(client):
    r = client.post(
        "/score",
        json={
            "customers": [
                {"recency": 2, "history": 300, "womens": 1},
                {"recency": 18, "history": 50, "newbie": 0},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"][0] == "target"
    assert body["recommendation"][1] == "avoid (sleeping dog)"
    assert body["uplift"][0] > 0 > body["uplift"][1]


def test_score_rejects_empty(client):
    assert client.post("/score", json={"customers": []}).status_code == 422


def test_score_rejects_negative_recency(client):
    r = client.post("/score", json={"customers": [{"recency": -1, "history": 10}]})
    assert r.status_code == 422
