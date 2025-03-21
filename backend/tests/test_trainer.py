import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.config import settings
from src.training import Trainer
from src.training.dataset import AdultIncomeData
from src.training.models import ModelFactory


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
    os.environ["DYNACONF_DATASET__PATH"] = "test_data.csv"  # Dummy path
    os.environ["DYNACONF_DATASET__COLUMNS__TARGET"] = "income"
    os.environ["DYNACONF_DATASET__COLUMNS__FEATURES"] = '["age", "educational-num"]'
    os.environ["DYNACONF_DATASET__ORIGIN_PATH"] = "test_data.csv"  # Dummy path

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
    os.environ["DYNACONF_DATASET__ORIGIN_PATH"] = str(temp_csv_path)
    os.environ["DYNACONF_DATASET__COLUMNS__TARGET"] = "income"
    os.environ["DYNACONF_DATASET__COLUMNS__FEATURES"] = (
        '["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week", "workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "fnlwgt"]'
    )

    return str(temp_csv_path)


@pytest.fixture
def trainer(test_data):
    """Create a Trainer instance with test settings."""
    settings.reload()  # Reload settings to apply the mocked values
    return Trainer()


def test_trainer_initialization(trainer):
    """Test that the Trainer class initializes correctly."""
    assert settings.MLFLOW__TRACKING_URI == "test_mlruns"
    assert settings.MLFLOW__EXPERIMENT_NAME == "TestExperiment"
    assert settings.OPTIMIZATION__N_TRIALS == 1
    assert settings.MODELS == ["XGBClassifier"]
    assert isinstance(trainer.data, AdultIncomeData)
    assert isinstance(trainer.preprocessor, ColumnTransformer)
    assert isinstance(trainer.customprocessor, FunctionTransformer)
    assert len(trainer.models) > 0


def test_load_models_from_config(trainer):
    """Test loading models from the configuration."""
    models = trainer._Trainer__load_models_from_config(settings.MODELS)
    assert isinstance(models, list)
    assert len(models) > 0
    # Assert that each element in models is an instance of BaseModel
    # Assuming BaseModel is the base class for your models
    from src.training.models import BaseModel  # Import BaseModel

    for model in models:
        assert isinstance(model, BaseModel)


def test_init_preprocessor(trainer):
    """Test the initialization of the preprocessor."""
    preprocessor = trainer._Trainer__init_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_init_customprocessor(trainer):
    """Test the initialization of the custom preprocessor."""
    customprocessor = trainer._Trainer__init_customprocessor()
    assert isinstance(customprocessor, FunctionTransformer)


def test_custom_preprocessing(trainer, test_data):
    """Test the custom preprocessing method."""
    test_df = pd.read_csv(test_data)
    processed_df = trainer.custom_preprocessing(test_df.copy())

    # Assert that 'native-country' and 'race' columns are modified as expected
    assert all(
        x in ["United-States", "Other Countries"]
        for x in processed_df["native-country"]
    )
    assert all(x in ["White", "Other"] for x in processed_df["race"])
