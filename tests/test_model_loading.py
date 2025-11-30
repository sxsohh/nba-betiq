"""
Test suite for model loading and predictions.
"""
import pytest
import os
import joblib
import numpy as np
import pandas as pd
from backend.utils import load_model


MODEL_DIR = "ml/models"


def test_model_files_exist():
    """Test that required model files exist."""
    required_models = [
        "home_win_xgb.pkl",
        "home_win_scaler.pkl",
        "home_win_logreg_calibrated.pkl",
        "spread_xgb.pkl",
        "spread_scaler.pkl",
        "spread_logreg_calibrated.pkl",
        "ou_xgb.pkl",
        "ou_scaler.pkl",
        "ou_logreg_calibrated.pkl",
    ]

    missing = []
    for model_file in required_models:
        path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(path):
            missing.append(model_file)

    if missing:
        pytest.skip(f"Model files not found: {missing}. Run training first.")


def test_load_home_win_model():
    """Test loading home win XGBoost model."""
    model_path = os.path.join(MODEL_DIR, "home_win_xgb.pkl")

    if not os.path.exists(model_path):
        pytest.skip("home_win_xgb.pkl not found")

    try:
        model = load_model("home_win_xgb.pkl")
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")


def test_model_prediction_shape():
    """Test model prediction output shape."""
    model_path = os.path.join(MODEL_DIR, "home_win_xgb.pkl")

    if not os.path.exists(model_path):
        pytest.skip("Model not found")

    model = joblib.load(model_path)

    # Create dummy input
    n_features = model.n_features_in_
    X_dummy = np.random.rand(10, n_features)

    # Test predict
    preds = model.predict(X_dummy)
    assert preds.shape == (10,)

    # Test predict_proba
    proba = model.predict_proba(X_dummy)
    assert proba.shape == (10, 2)


def test_model_prediction_range():
    """Test that predicted probabilities are in valid range [0, 1]."""
    model_path = os.path.join(MODEL_DIR, "home_win_xgb.pkl")

    if not os.path.exists(model_path):
        pytest.skip("Model not found")

    model = joblib.load(model_path)
    n_features = model.n_features_in_
    X_dummy = np.random.rand(10, n_features)

    proba = model.predict_proba(X_dummy)[:, 1]

    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_scaler_transformation():
    """Test that scaler transforms data correctly."""
    scaler_path = os.path.join(MODEL_DIR, "home_win_scaler.pkl")

    if not os.path.exists(scaler_path):
        pytest.skip("Scaler not found")

    try:
        scaler = joblib.load(scaler_path)
    except (ModuleNotFoundError, ImportError) as e:
        pytest.skip(f"Cannot load scaler due to numpy version mismatch: {e}")

    # Create dummy input matching expected feature count
    n_features = scaler.n_features_in_
    X_dummy = np.random.rand(10, n_features)

    # Transform (should not raise an error)
    X_scaled = scaler.transform(X_dummy)

    # Verify shape is preserved
    assert X_scaled.shape == X_dummy.shape
    # Verify output is finite (no NaN or inf values)
    assert np.all(np.isfinite(X_scaled))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
