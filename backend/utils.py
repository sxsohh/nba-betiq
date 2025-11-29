"""
Utility functions for NBA BetIQ API.
Includes odds conversion, EV calculation, and model loading.
"""
import os
import logging
from pathlib import Path
from typing import Tuple, Dict
import joblib
import pandas as pd
import numpy as np
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds

    Examples:
        >>> american_to_decimal(-110)
        1.909
        >>> american_to_decimal(150)
        2.5
    """
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def american_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds

    Returns:
        Implied probability (0 to 1)

    Examples:
        >>> american_to_implied_prob(-110)
        0.5238
        >>> american_to_implied_prob(150)
        0.4
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def calculate_ev(
    win_prob: float,
    odds: int,
    stake: float = 100.0
) -> Dict[str, float]:
    """
    Calculate expected value (EV) for a bet.

    Args:
        win_prob: Probability of winning (0 to 1)
        odds: American odds
        stake: Bet amount

    Returns:
        Dictionary containing:
            - ev: Expected value in dollars
            - ev_percent: EV as percentage of stake
            - implied_prob: Implied probability from odds
            - edge: Model probability - implied probability
            - recommendation: "BET" if EV > 0, else "PASS"
    """
    decimal_odds = american_to_decimal(odds)
    implied_prob = american_to_implied_prob(odds)

    # Calculate payouts
    win_amount = stake * (decimal_odds - 1)
    lose_amount = -stake

    # Expected value
    ev = (win_prob * win_amount) + ((1 - win_prob) * lose_amount)
    ev_percent = (ev / stake) * 100

    # Edge
    edge = win_prob - implied_prob

    # Recommendation
    recommendation = "BET" if ev > 0 else "PASS"

    return {
        "ev": round(ev, 2),
        "ev_percent": round(ev_percent, 2),
        "implied_prob": round(implied_prob, 4),
        "edge": round(edge, 4),
        "recommendation": recommendation
    }


def load_model(model_name: str):
    """
    Load a trained model from disk with error handling.

    Args:
        model_name: Name of the model file

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    model_path = os.path.join(settings.model_dir, model_name)

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_name}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise


def payload_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """
    Convert feature dictionary to DataFrame for model prediction.

    Args:
        features: Dictionary of feature names to values

    Returns:
        Single-row DataFrame
    """
    return pd.DataFrame([features])


def validate_features(features: Dict[str, float], required_features: list) -> Tuple[bool, str]:
    """
    Validate that all required features are present.

    Args:
        features: Feature dictionary
        required_features: List of required feature names

    Returns:
        Tuple of (is_valid, error_message)
    """
    missing = set(required_features) - set(features.keys())

    if missing:
        return False, f"Missing required features: {missing}"

    return True, ""


def kelly_criterion(
    win_prob: float,
    odds: int,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.

    Args:
        win_prob: Probability of winning
        odds: American odds
        kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly)

    Returns:
        Optimal bet size as fraction of bankroll
    """
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1  # Net odds received on the wager

    # Kelly formula: f = (bp - q) / b
    # where p = win prob, q = loss prob, b = net odds
    q = 1 - win_prob
    kelly = (b * win_prob - q) / b

    # Apply fractional Kelly
    kelly = max(0, kelly * kelly_fraction)

    return round(kelly, 4)


def calculate_roi(
    wins: int,
    losses: int,
    avg_odds: int,
    stake: float = 100.0
) -> Dict[str, float]:
    """
    Calculate return on investment for a betting strategy.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        avg_odds: Average American odds
        stake: Stake per bet

    Returns:
        Dictionary with ROI metrics
    """
    total_bets = wins + losses
    total_staked = total_bets * stake

    decimal_odds = american_to_decimal(avg_odds)
    win_amount = wins * stake * (decimal_odds - 1)
    loss_amount = losses * stake

    profit = win_amount - loss_amount
    roi = (profit / total_staked) * 100 if total_staked > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_bets, 4) if total_bets > 0 else 0,
        "total_staked": total_staked,
        "profit": round(profit, 2),
        "roi_percent": round(roi, 2)
    }
