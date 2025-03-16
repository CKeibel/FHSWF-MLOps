import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup the testing environment with the required environment variables."""
    # Store original environment
    original_env = os.environ.copy()

    # Set environment variables that Dynaconf will pick up
    os.environ["DYNACONF_MLFLOW__TRACKING_URI"] = "test_mlruns"
    os.environ["DYNACONF_MLFLOW__EXPERIMENT_NAME"] = "TestExperiment"
    os.environ["DYNACONF_OPTIMIZATION__N_TRIALS"] = "1"
    os.environ["DYNACONF_MODELS"] = '["XGBClassifier"]'

    # Testing will use a temporary dataset created in the test_data fixture

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_data(tmp_path):
    """Create a small test dataset in a temporary directory."""
    # Create test data that mimics your adult income dataset structure
    data = {
        "age": [25, 45, 36, 28, 52],
        "educational-num": [13, 10, 16, 12, 9],
        "capital-gain": [0, 5000, 0, 0, 15000],
        "capital-loss": [0, 0, 1500, 0, 0],
        "hours-per-week": [40, 60, 40, 30, 40],
        "workclass": ["Private", "Self-emp", "Private", "Federal-gov", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters", "Associates", "HS-grad"],
        "marital-status": [
            "Never-married",
            "Married",
            "Divorced",
            "Married",
            "Widowed",
        ],
        "occupation": ["Tech", "Management", "Sales", "Admin", "Blue-collar"],
        "relationship": [
            "Not-in-family",
            "Husband",
            "Unmarried",
            "Wife",
            "Not-in-family",
        ],
        "race": ["White", "Black", "White", "Asian-Pac", "White"],
        "gender": ["Male", "Male", "Female", "Female", "Male"],
        "native-country": [
            "United-States",
            "United-States",
            "Mexico",
            "India",
            "United-States",
        ],
        "fnlwgt": [100000, 200000, 150000, 80000, 180000],
        "income": [0, 1, 0, 1, 0],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to temporary CSV file
    temp_csv_path = tmp_path / "test_data.csv"
    df.to_csv(temp_csv_path, index=False)

    # Set the dataset path in the environment
    os.environ["DYNACONF_DATASET__PATH"] = str(temp_csv_path)
    os.environ["DYNACONF_DATASET__COLUMNS__TARGET"] = "income"
    os.environ["DYNACONF_DATASET__COLUMNS__FEATURES"] = (
        '["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week", "workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "fnlwgt"]'
    )

    return str(temp_csv_path)


@pytest.fixture
def test_app(test_data):
    """Create a TestClient for the app with test settings."""
    # Import app here to ensure environment variables are set first
    from src.main import app

    client = TestClient(app)
    return client


def test_health(test_app):
    """Test the health endpoint."""
    response = test_app.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    # Note: The actual JSON might depend on your service logic - adjust assertion as needed
