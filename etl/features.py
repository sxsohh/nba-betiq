"""
ETL Step 3: Feature Engineering
Creates features for machine learning from merged game data.
"""
import os
import pandas as pd
import numpy as np
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"
DB_PATH = "db/nba_betting.db"
os.makedirs("db", exist_ok=True)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def engineer_betting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from betting odds.

    Features created:
    - Implied probabilities from moneyline, spread, total
    - Vig (house edge) calculations
    - Line movement indicators
    """
    logger.info("Engineering betting features...")

    # Convert odds columns to numeric
    odds_cols = [
        "pinnacle_ml_home", "pinnacle_ml_away",
        "home_spread", "away_spread",
        "pinnacle_total", "home_spread_odds", "away_spread_odds",
        "home_ou_odds", "away_ou_odds"
    ]

    for col in odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Implied probabilities from moneyline
    df["prob_home_ml"] = df["pinnacle_ml_home"].apply(american_to_implied_prob)
    df["prob_away_ml"] = df["pinnacle_ml_away"].apply(american_to_implied_prob)
    df["vig_ml"] = df["prob_home_ml"] + df["prob_away_ml"] - 1.0

    # Implied probabilities from spread
    df["prob_home_spread"] = df["home_spread_odds"].apply(american_to_implied_prob)
    df["prob_away_spread"] = df["away_spread_odds"].apply(american_to_implied_prob)
    df["vig_spread"] = df["prob_home_spread"].fillna(0) + df["prob_away_spread"].fillna(0) - 1.0

    # Implied probabilities from O/U
    df["prob_over"] = df["home_ou_odds"].apply(american_to_implied_prob)
    df["prob_under"] = df["away_ou_odds"].apply(american_to_implied_prob)
    df["vig_ou"] = df["prob_over"].fillna(0) + df["prob_under"].fillna(0) - 1.0

    # Public betting percentages (already in data)
    # These show where the public money is

    logger.info(f"✓ Created betting features")
    return df


def engineer_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for ML models.

    Targets:
    - home_win: Did home team win?
    - home_spread_cover: Did home team cover the spread?
    - ou_over_win: Did the total go over?
    """
    logger.info("Creating target variables...")

    # Convert points to numeric
    df["HOME_PTS"] = pd.to_numeric(df["HOME_PTS"], errors="coerce")
    df["AWAY_PTS"] = pd.to_numeric(df["AWAY_PTS"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    # Use HOME_PTS/AWAY_PTS from scores if available, else fall back to odds file
    df["final_home_pts"] = df["HOME_PTS"].fillna(df["home_pts"])
    df["final_away_pts"] = df["AWAY_PTS"].fillna(df["away_pts"])

    # Target 1: Home team win
    df["home_win"] = (df["final_home_pts"] > df["final_away_pts"]).astype(int)

    # Target 2: Home team covers spread
    df["spread_margin"] = df["final_home_pts"] - df["final_away_pts"]
    df["home_spread_cover"] = ((df["spread_margin"] + df["home_spread"]) > 0).astype(int)

    # Target 3: Over/Under
    df["total_pts"] = df["final_home_pts"] + df["final_away_pts"]
    df["ou_over_win"] = (df["total_pts"] > df["pinnacle_total"]).astype(int)

    logger.info("✓ Created target variables")
    return df


def engineer_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from team statistics.

    Features:
    - Field goal percentages (FG%, 3P%, FT%)
    - Rebounds, assists, turnovers
    - Points scored
    """
    logger.info("Engineering team features...")

    # Numeric columns to preserve
    stat_cols = [
        "HOME_FG_PCT", "HOME_FT_PCT", "HOME_FG3_PCT", "HOME_AST", "HOME_REB", "HOME_TOV",
        "AWAY_FG_PCT", "AWAY_FT_PCT", "AWAY_FG3_PCT", "AWAY_AST", "AWAY_REB", "AWAY_TOV"
    ]

    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("✓ Team features engineered")
    return df


def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced derived features.

    Features:
    - Pace estimation
    - Efficiency metrics
    - Matchup advantages
    """
    logger.info("Creating advanced features...")

    # Estimated pace (total possessions)
    df["estimated_pace"] = (df["HOME_AST"] + df["AWAY_AST"] + df["HOME_TOV"] + df["AWAY_TOV"]) / 2

    # Shooting efficiency differential
    df["fg_pct_diff"] = df["HOME_FG_PCT"] - df["AWAY_FG_PCT"]
    df["fg3_pct_diff"] = df["HOME_FG3_PCT"] - df["AWAY_FG3_PCT"]

    # Rebound differential
    df["reb_diff"] = df["HOME_REB"] - df["AWAY_REB"]

    # Assist-to-turnover ratio
    df["home_ast_tov_ratio"] = df["HOME_AST"] / (df["HOME_TOV"] + 1)
    df["away_ast_tov_ratio"] = df["AWAY_AST"] / (df["AWAY_TOV"] + 1)

    logger.info("✓ Advanced features created")
    return df


def clean_and_finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset and prepare for ML.

    Steps:
    - Remove text/identifier columns
    - Handle missing values
    - Select final feature set
    """
    logger.info("Finalizing dataset...")

    # Drop identifier columns (not useful for ML)
    drop_cols = [
        "Date", "season", "home_team", "away_team",
        "HOME_TEAM_CITY_NAME", "AWAY_TEAM_CITY_NAME",
        "HOME_TEAM_WINS_LOSSES", "AWAY_TEAM_WINS_LOSSES",
        "GAME_DATE_EST"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Fill missing values
    df = df.fillna(0)

    # Ensure all remaining columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    logger.info(f"✓ Final dataset shape: {df.shape}")
    return df


def save_to_db(df: pd.DataFrame):
    """Save dataset to SQLite database."""
    logger.info("Saving to SQLite database...")

    conn = sqlite3.connect(DB_PATH)

    # Save games table
    df.to_sql("games", conn, if_exists="replace", index=False)

    # Create betting outcomes table
    outcomes = df[[
        "GAME_ID", "home_win", "home_spread_cover", "ou_over_win",
        "home_spread", "pinnacle_total", "total_pts"
    ]].copy()

    outcomes.to_sql("betting_outcomes", conn, if_exists="replace", index=False)

    conn.close()
    logger.info(f"✓ Saved to database: {DB_PATH}")


def main():
    """Run full feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    try:
        # Load merged data
        df = pd.read_csv(os.path.join(DATA_DIR, "games_merged.csv"))
        logger.info(f"Loaded merged data: {df.shape}")

        # Engineer features
        df = engineer_betting_features(df)
        df = engineer_target_variables(df)
        df = engineer_team_features(df)
        df = engineer_advanced_features(df)
        df = clean_and_finalize(df)

        # Save outputs
        out_path = os.path.join(DATA_DIR, "games_master_2018_19.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"✓ Saved master dataset: {out_path}")

        # Save to database
        save_to_db(df)

        logger.info("=" * 60)
        logger.info("✓ FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
