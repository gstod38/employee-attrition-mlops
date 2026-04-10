import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.train import train_model

@pytest.fixture
def dummy_training_data():
    """Create a tiny dataset for model smoke tests"""
    X = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1.0, 0.9, 0.8, 0.7]
    })
    y = pd.Series([0, 1, 0, 1], dtype=int)
    return X, y

def test_model_prediction_shape(dummy_training_data):
    """Test 10: Verify the model outputs the correct shape"""
    X, y = dummy_training_data
    model = RandomForestClassifier(max_depth=2, n_estimators=10)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == len(X), "Prediction count does not match input count"

def test_model_output_type(dummy_training_data):
    """Test 11: Verify model output is binary (0 or 1)"""
    X, y = dummy_training_data
    model = RandomForestClassifier(max_depth=2, n_estimators=10)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert np.all((preds == 0) | (preds == 1)), "Model produced non-binary predictions"