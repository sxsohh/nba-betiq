"""
Test suite for model calibration.
"""
import pytest
import numpy as np
from sklearn.calibration import calibration_curve


def test_calibration_curve_perfect():
    """Test calibration with perfectly calibrated predictions."""
    n = 1000
    np.random.seed(42)

    # Perfect calibration: predicted prob == actual frequency
    y_true = np.random.binomial(1, 0.7, n)
    y_prob = np.full(n, 0.7)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)

    # Expected Calibration Error should be small
    ece = np.mean(np.abs(prob_true - prob_pred))
    assert ece < 0.1  # Reasonable tolerance


def test_calibration_curve_overconfident():
    """Test calibration with over confident predictions."""
    n = 1000
    np.random.seed(42)

    # Overconfident: predicting higher prob than actual
    y_true = np.random.binomial(1, 0.5, n)
    y_prob = np.full(n, 0.9)  # Predicting 90% but true rate is 50%

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)

    # True probabilities should be lower than predicted
    assert np.mean(prob_true) < np.mean(prob_pred)


def test_expected_calibration_error():
    """Test ECE calculation."""
    # Synthetic data
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.85, 0.3, 0.1, 0.75, 0.95, 0.15, 0.88])

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=3)
    ece = np.mean(np.abs(prob_true - prob_pred))

    assert ece >= 0  # ECE should always be non-negative
    assert ece <= 1  # ECE should be bounded by 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
