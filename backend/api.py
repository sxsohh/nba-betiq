"""
NBA BetIQ API - Production-ready FastAPI application.
Provides endpoints for NBA game predictions and expected value calculations.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

from .config import settings
from .schemas import (
    HealthResponse,
    FeaturePayload,
    HomeWinPrediction,
    SpreadPrediction,
    OUPrediction,
    EVRequest,
    EVResponse,
    PredictEVRequest,
    PredictEVResponse,
)
from .utils import (
    load_model,
    payload_to_dataframe,
    calculate_ev,
    american_to_implied_prob,
    logger,
)

# Global model cache
models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup, clean up on shutdown.
    """
    logger.info("Loading ML models...")

    try:
        # Load home win models
        models["home_win_xgb"] = load_model(settings.home_win_model)
        models["home_win_scaler"] = load_model(settings.home_win_scaler)
        models["home_win_calib"] = load_model(settings.home_win_calib)

        # Load spread models
        models["spread_xgb"] = load_model(settings.spread_model)
        models["spread_scaler"] = load_model(settings.spread_scaler)
        models["spread_calib"] = load_model(settings.spread_calib)

        # Load O/U models
        models["ou_xgb"] = load_model(settings.ou_model)
        models["ou_scaler"] = load_model(settings.ou_scaler)
        models["ou_calib"] = load_model(settings.ou_calib)

        logger.info("✓ All models loaded successfully")

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        logger.warning("API starting without models - predictions will fail")
    except Exception as e:
        logger.error(f"Unexpected error loading models: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down API...")
    models.clear()


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for model errors
@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Model not available", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/", tags=["Root"])
def root():
    """Root endpoint - API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns API status and model availability.
    """
    model_status = {
        "home_win": "home_win_xgb" in models,
        "spread": "spread_xgb" in models,
        "ou": "ou_xgb" in models,
    }

    all_loaded = all(model_status.values())

    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        message=f"NBA BetIQ API is running. Models loaded: {sum(model_status.values())}/3"
    )


@app.post("/predict", response_model=HomeWinPrediction, tags=["Predictions"])
def predict_home_win(payload: FeaturePayload):
    """
    Predict home team win probability.

    **Example Request:**
    ```json
    {
        "features": {
            "HOME_FG_PCT": 0.462,
            "AWAY_FG_PCT": 0.448,
            "HOME_FG3_PCT": 0.365,
            "AWAY_FG3_PCT": 0.342
        }
    }
    ```
    """
    try:
        if "home_win_xgb" not in models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Home win model not loaded"
            )

        X = payload_to_dataframe(payload.features)
        model = models["home_win_xgb"]

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        return HomeWinPrediction(
            prediction=pred,
            probability=prob,
            prob_home_win=prob,
            model_version=settings.api_version,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/spread", response_model=SpreadPrediction, tags=["Predictions"])
def predict_spread(payload: FeaturePayload):
    """
    Predict probability that home team covers the spread.
    """
    try:
        if "spread_xgb" not in models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Spread model not loaded"
            )

        X = payload_to_dataframe(payload.features)
        model = models["spread_xgb"]

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        return SpreadPrediction(
            prediction=pred,
            probability=prob,
            prob_home_covers=prob,
            model_version=settings.api_version,
        )

    except Exception as e:
        logger.error(f"Spread prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/ou", response_model=OUPrediction, tags=["Predictions"])
def predict_ou(payload: FeaturePayload):
    """
    Predict over/under probability.
    """
    try:
        if "ou_xgb" not in models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="O/U model not loaded"
            )

        X = payload_to_dataframe(payload.features)
        model = models["ou_xgb"]

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        return OUPrediction(
            prediction=pred,
            probability=prob,
            prob_over=prob,
            model_version=settings.api_version,
        )

    except Exception as e:
        logger.error(f"O/U prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/ev", response_model=EVResponse, tags=["Expected Value"])
def calculate_expected_value(request: EVRequest):
    """
    Calculate expected value for a bet given win probability and odds.

    **Example Request:**
    ```json
    {
        "probability": 0.58,
        "odds": -110,
        "stake": 100
    }
    ```

    **Returns:**
    - EV in dollars
    - EV as percentage
    - Implied probability from odds
    - Edge (model prob - implied prob)
    - Recommendation (BET/PASS)
    """
    try:
        ev_data = calculate_ev(
            win_prob=request.probability,
            odds=request.odds,
            stake=request.stake
        )

        return EVResponse(
            ev=ev_data["ev"],
            ev_percent=ev_data["ev_percent"],
            implied_prob=ev_data["implied_prob"],
            model_prob=request.probability,
            edge=ev_data["edge"],
            recommendation=ev_data["recommendation"]
        )

    except Exception as e:
        logger.error(f"EV calculation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"EV calculation failed: {str(e)}"
        )


@app.post("/predict_ev", response_model=PredictEVResponse, tags=["Combined"])
def predict_and_calculate_ev(request: PredictEVRequest):
    """
    Combined endpoint: Make prediction AND calculate EV in one call.

    **Bet Types:**
    - `home_win`: Predict home team win
    - `spread`: Predict spread cover
    - `ou`: Predict over/under

    **Example Request:**
    ```json
    {
        "features": {
            "HOME_FG_PCT": 0.462,
            "AWAY_FG_PCT": 0.448
        },
        "odds": -110,
        "stake": 100,
        "bet_type": "home_win"
    }
    ```
    """
    try:
        # Select model based on bet type
        model_map = {
            "home_win": "home_win_xgb",
            "spread": "spread_xgb",
            "ou": "ou_xgb",
        }

        model_key = model_map.get(request.bet_type)
        if not model_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid bet_type. Must be one of: {list(model_map.keys())}"
            )

        if model_key not in models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model for {request.bet_type} not loaded"
            )

        # Make prediction
        X = payload_to_dataframe(request.features)
        model = models[model_key]

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        # Calculate EV
        ev_data = calculate_ev(
            win_prob=prob,
            odds=request.odds,
            stake=request.stake
        )

        return PredictEVResponse(
            prediction=pred,
            probability=prob,
            ev=ev_data["ev"],
            ev_percent=ev_data["ev_percent"],
            implied_prob=ev_data["implied_prob"],
            edge=ev_data["edge"],
            recommendation=ev_data["recommendation"],
            model_version=settings.api_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict + EV error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction + EV failed: {str(e)}"
        )
