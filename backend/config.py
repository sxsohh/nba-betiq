"""
Configuration management for NBA BetIQ API.
Loads settings from environment variables with defaults.
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_title: str = "NBA BetIQ API"
    api_version: str = "1.0.0"
    api_description: str = "Estimate win probabilities and expected value for NBA bets."

    # Environment
    environment: str = "development"

    # Database
    db_path: str = str(ROOT_DIR / "db" / "nba_betting.db")

    # ML Models
    model_dir: str = str(ROOT_DIR / "ml" / "models")
    home_win_model: str = "home_win_xgb.pkl"
    home_win_scaler: str = "home_win_scaler.pkl"
    home_win_calib: str = "home_win_logreg_calibrated.pkl"

    spread_model: str = "spread_xgb.pkl"
    spread_scaler: str = "spread_scaler.pkl"
    spread_calib: str = "spread_logreg_calibrated.pkl"

    ou_model: str = "ou_xgb.pkl"
    ou_scaler: str = "ou_scaler.pkl"
    ou_calib: str = "ou_logreg_calibrated.pkl"

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Logging
    log_level: str = "INFO"
    log_file: str = str(ROOT_DIR / "logs" / "api.log")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
