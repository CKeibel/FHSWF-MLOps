# -*- coding: utf-8 -*-
"""API integration tests."""
import os

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="session")
def test_app():
    """Fixture to create a app instance."""
    os.environ["DYNCONF_MLFLOW_DATA_PATH"] = "Test"
    os.environ["DYNCONF_MLFLOW_URI"] = "Test"
    client = TestClient(app)
    yield client


def test_health(test_app):
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
