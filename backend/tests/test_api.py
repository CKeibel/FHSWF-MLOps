# -*- coding: utf-8 -*-
"""API integration tests."""
import pytest
from fastapi.testclient import TestClient
from src.backend.main import app


@pytest.fixture(scope="session")
def test_app():
    """Fixture to create a app instance."""
    client = TestClient(app)
    yield client


def test_health(test_app):
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
