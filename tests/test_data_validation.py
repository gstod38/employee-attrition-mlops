import pytest
import pandas as pd

@pytest.fixture
def raw_data():
    return pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

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