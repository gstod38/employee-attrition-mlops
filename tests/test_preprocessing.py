import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    """Creates a dummy dataset that mimics the IBM HR structure"""
    return pd.DataFrame({
        'Attrition': ['Yes', 'No', 'No'],
        'Age': [34, 28, 45],
        'Department': ['Sales', 'R&D', 'Sales'],
        'EmployeeCount': [1, 1, 1],  # Constant column to be dropped
        'Over18': ['Y', 'Y', 'Y'],    # Constant column to be dropped
        'DailyRate': [800, 1200, np.nan] # Column with a missing value
    })

@pytest.fixture
def config():
    """Mock config for testing"""
    return {
        'train': {'target_column': 'Attrition'},
        'data': {'processed_path': 'data/test_processed.csv'}
    }

def test_target_encoding(sample_data, config):
    """Test 1: Verify 'Yes'/'No' becomes 1/0"""
    processed = preprocess_data(sample_data.copy(), config)
    assert processed['Attrition'].tolist() == [1, 0, 0]

def test_drop_constant_columns(sample_data, config):
    """Test 2: Verify EmployeeCount and Over18 are removed"""
    processed = preprocess_data(sample_data.copy(), config)
    assert 'EmployeeCount' not in processed.columns
    assert 'Over18' not in processed.columns

def test_original_not_modified(sample_data, config):
    """Test 3: Verify the function uses a copy and doesn't mutate input"""
    original = sample_data.copy()
    _ = preprocess_data(sample_data, config)
    pd.testing.assert_frame_equal(sample_data, original)

def test_categorical_encoding(sample_data, config):
    """Test 4: Verify Department is one-hot encoded"""
    processed = preprocess_data(sample_data.copy(), config)
    assert any('Department_' in col for col in processed.columns)

def test_handle_missing_values(sample_data, config):
    """Test 5: Verify missing values are imputed (no NaNs remain)"""
    processed = preprocess_data(sample_data.copy(), config)
    assert processed['DailyRate'].isnull().sum() == 0

def test_invalid_input_error(config):
    """Test 6: Verify it raises an error if passed something other than a DataFrame"""
    with pytest.raises(Exception):
        preprocess_data("Not a dataframe", config)