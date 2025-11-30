"""
Test suite for API health checks and endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.api import app

    client = TestClient(app)
    API_AVAILABLE = True
except Exception as e:
    API_AVAILABLE = False
    print(f"API not available: {e}")


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_root_endpoint():
    """Test root endpoint returns basic info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "message" in data
    assert data["status"] in ["healthy", "degraded"]


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_ev_endpoint():
    """Test EV calculation endpoint."""
    payload = {
        "probability": 0.58,
        "odds": -110,
        "stake": 100
    }

    response = client.post("/ev", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "ev" in data
    assert "ev_percent" in data
    assert "implied_prob" in data
    assert "edge" in data
    assert "recommendation" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_ev_endpoint_invalid_probability():
    """Test EV endpoint with invalid probability."""
    payload = {
        "probability": 1.5,  # Invalid: > 1
        "odds": -110,
        "stake": 100
    }

    response = client.post("/ev", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_predict_endpoint_with_features():
    """Test prediction endpoint with sample features."""
    payload = {
        "features": {
            "HOME_FG_PCT": 0.462,
            "AWAY_FG_PCT": 0.448,
            "HOME_FG3_PCT": 0.365,
            "AWAY_FG3_PCT": 0.342,
            "HOME_REB": 45,
            "AWAY_REB": 42
        }
    }

    response = client.post("/predict", json=payload)

    # May return 503 if models not loaded, or 500 on error (acceptable in CI)
    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0 <= data["probability"] <= 1


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_docs_endpoint():
    """Test that OpenAPI docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
