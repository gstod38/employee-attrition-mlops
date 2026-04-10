import os 
import pytest
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

@pytest.fixture
def raw_data():
    """Logic to load data or skip if not available in CI"""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found at {DATA_PATH}. Skipping data validation.")
    return pd.read_csv(DATA_PATH)

def test_column_presence(raw_data):
    """Test 7: Check for essential columns"""
    required = ['Attrition', 'Age', 'MonthlyIncome', 'JobRole']
    for col in required:
        assert col in raw_data.columns

def test_target_values(raw_data):
    """Test 8: Ensure target is binary"""
    assert set(raw_data['Attrition'].unique()) == {'Yes', 'No'}

def test_age_range(raw_data):
    """Test 9: Verify realistic age ranges"""
    assert raw_data['Age'].min() >= 18
    assert raw_data['Age'].max() <= 100