import os

import numpy as np
import pandas as pd
import pytest
from src.training.dataset import AdultIncomeData


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample dataset config with test data."""
    # Create test data
    data = {
        "age": [25, 45, 36, 28, 52],
        "educational-num": [13, 10, 16, 12, 9],
        "workclass": ["Private", "Self-emp", "Private", "Federal-gov", "Private"],
        "income": ["<=50K", ">50K", "<=50K", ">50K", "<=50K"],
    }

    # Save to temporary CSV file
    temp_csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(data).to_csv(temp_csv_path, index=False)

    # Create config dictionary
    config = {
        "path": str(temp_csv_path),
        "origin_path": None,
        "columns": {
            "target": "income",
            "features": ["age", "educational-num", "workclass"],
        },
    }

    return config


@pytest.fixture
def adult_income_data(sample_config):
    """Create an instance of AdultIncomeData with test config."""
    return AdultIncomeData(sample_config)


def test_initialization(adult_income_data):
    """Test that AdultIncomeData initializes correctly."""
    # Check basic attributes
    assert adult_income_data.target == "income"
    assert adult_income_data.features == ["age", "educational-num", "workclass"]

    # Check data loading
    assert isinstance(adult_income_data.current_data, pd.DataFrame)
    assert len(adult_income_data.current_data) == 5

    # Check processing results
    assert hasattr(adult_income_data, "X_train")
    assert hasattr(adult_income_data, "X_test")
    assert hasattr(adult_income_data, "y_train")
    assert hasattr(adult_income_data, "y_test")

    # Check feature type detection
    assert list(adult_income_data.numerical_features) == ["age", "educational-num"]
    assert list(adult_income_data.categorical_features) == ["workclass"]


def test_feature_type_detection(adult_income_data):
    """Test the feature type detection methods."""
    # Test numerical features getter
    numerical = adult_income_data.get_nummerical_features()
    assert len(numerical) == 2
    assert "age" in numerical
    assert "educational-num" in numerical

    # Test categorical features getter
    categorical = adult_income_data.get_categorical_features()
    assert len(categorical) == 1
    assert "workclass" in categorical


def test_data_cleaning(adult_income_data):
    """Test the data cleaning process."""
    # Create test dataframe with NAs and original income values
    test_df = pd.DataFrame(
        {
            "age": [25, 45, None, 28],
            "educational-num": [13, 10, 16, 12],
            "workclass": ["Private", None, "Private", "Federal-gov"],
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )

    # Apply cleaning
    cleaned_df = adult_income_data._clean_data(test_df)

    # Check NA removal
    assert len(cleaned_df) == 2  # Should have removed rows with NA

    # Check target transformation
    assert all(isinstance(val, int) for val in cleaned_df["income"])
    assert cleaned_df["income"].iloc[0] == 0  # <=50K becomes 0
    assert cleaned_df["income"].iloc[1] == 1  # >50K becomes 1


def test_append_data_success(adult_income_data, sample_config):
    """Test successful appending of new data."""
    # Create new data with same structure
    new_data = pd.DataFrame(
        {
            "age": [30, 55],
            "educational-num": [14, 8],
            "workclass": ["Private", "Self-emp"],
            "income": ["<=50K", ">50K"],
        }
    )

    # Initial counts
    initial_count = len(adult_income_data.current_data)

    # Append new data
    adult_income_data.append_data(new_data)

    # Check results
    assert len(adult_income_data.current_data) == initial_count + 2

    # Verify the last rows match our appended data (after processing)
    # Note: The actual data might change due to processing like target transformation


def test_append_data_missing_columns(adult_income_data):
    """Test error handling for missing columns in append_data."""
    # Create new data with missing columns
    new_data = pd.DataFrame(
        {
            "age": [30, 55],
            "educational-num": [14, 8],
            # Missing 'workclass'
            "income": ["<=50K", ">50K"],
        }
    )

    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        adult_income_data.append_data(new_data)

    assert "Missing required columns" in str(excinfo.value)


def test_append_data_extra_columns(adult_income_data, capfd):
    """Test handling of extra columns in append_data."""
    # Create new data with extra columns
    new_data = pd.DataFrame(
        {
            "age": [30, 55],
            "educational-num": [14, 8],
            "workclass": ["Private", "Self-emp"],
            "income": ["<=50K", ">50K"],
            "extra_column": [1, 2],  # Extra column
        }
    )

    # Should print a warning but not fail
    adult_income_data.append_data(new_data)

    # Capture stdout and check for warning
    out, _ = capfd.readouterr()
    assert "Warning: Ignoring extra columns" in out

    # Check that the extra column is not in the data
    assert "extra_column" not in adult_income_data.current_data.columns


def test_append_data_type_mismatch_numeric(adult_income_data):
    """Test error handling for numeric type mismatches in append_data."""
    # Create new data with type mismatch for age (numeric column)
    new_data = pd.DataFrame(
        {
            "age": ["thirty", "fifty-five"],  # String instead of numeric
            "educational-num": [14, 8],
            "workclass": ["Private", "Self-emp"],
            "income": ["<=50K", ">50K"],
        }
    )

    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        adult_income_data.append_data(new_data)

    assert "Data type mismatch" in str(excinfo.value)
    assert "Expected numeric" in str(excinfo.value)


def test_append_data_type_mismatch_categorical(adult_income_data):
    """Test error handling for categorical type mismatches in append_data."""
    # Create new data with type mismatch for workclass (categorical column)
    new_data = pd.DataFrame(
        {
            "age": [30, 55],
            "educational-num": [14, 8],
            "workclass": [1, 2],  # Numeric instead of string
            "income": ["<=50K", ">50K"],
        }
    )

    # Convert workclass to numeric type explicitly
    new_data["workclass"] = new_data["workclass"].astype(np.int64)

    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        adult_income_data.append_data(new_data)

    assert "Data type mismatch" in str(excinfo.value)
    assert "Expected object/string" in str(excinfo.value)
