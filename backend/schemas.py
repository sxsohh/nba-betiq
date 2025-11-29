"""
Pydantic schemas for request/response validation in NBA BetIQ API.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    message: str


class FeaturePayload(BaseModel):
    """
    Payload for prediction requests.
    Features should match column names used during model training.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names to values",
        example={
            "HOME_FG_PCT": 0.462,
            "AWAY_FG_PCT": 0.448,
            "HOME_FG3_PCT": 0.365,
            "AWAY_FG3_PCT": 0.342,
            "home_spread": -5.5,
            "pinnacle_total": 218.5,
        }
    )


class PredictionResponse(BaseModel):
    """Base prediction response schema."""
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability")
    model_version: str = Field(default="1.0.0", description="Model version used")


class HomeWinPrediction(PredictionResponse):
    """Home team win probability prediction."""
    prob_home_win: float = Field(..., ge=0.0, le=1.0, description="Probability home team wins")


class SpreadPrediction(PredictionResponse):
    """Spread cover probability prediction."""
    prob_home_covers: float = Field(..., ge=0.0, le=1.0, description="Probability home team covers spread")


class OUPrediction(PredictionResponse):
    """Over/Under probability prediction."""
    prob_over: float = Field(..., ge=0.0, le=1.0, description="Probability of over")


class EVRequest(BaseModel):
    """Expected value calculation request."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Win probability from model")
    odds: int = Field(..., description="American odds (e.g., -110, +150)")
    stake: float = Field(default=100.0, gt=0, description="Bet amount in dollars")


class EVResponse(BaseModel):
    """Expected value calculation response."""
    ev: float = Field(..., description="Expected value in dollars")
    ev_percent: float = Field(..., description="Expected value as percentage of stake")
    implied_prob: float = Field(..., ge=0.0, le=1.0, description="Implied probability from odds")
    model_prob: float = Field(..., ge=0.0, le=1.0, description="Model predicted probability")
    edge: float = Field(..., description="Edge over bookmaker (model_prob - implied_prob)")
    recommendation: str = Field(..., description="Bet recommendation (BET/PASS)")


class PredictEVRequest(BaseModel):
    """Combined prediction + EV calculation request."""
    features: Dict[str, float] = Field(..., description="Feature dictionary for prediction")
    odds: int = Field(..., description="American odds")
    stake: float = Field(default=100.0, gt=0, description="Bet amount")
    bet_type: str = Field(..., description="Type of bet: 'home_win', 'spread', or 'ou'")


class PredictEVResponse(BaseModel):
    """Combined prediction + EV response."""
    prediction: int
    probability: float
    ev: float
    ev_percent: float
    implied_prob: float
    edge: float
    recommendation: str
    model_version: str = "1.0.0"
