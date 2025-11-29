"""
ETL Step 1: Data Ingestion
Loads raw data files from data/raw and performs initial cleaning.
"""
import os
import pandas as pd
from datetime import datetime
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def detect_season(date_str: str) -> str:
    """Detect NBA season from date (e.g., '2018-10-01' -> '2018-19')."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    if dt.month >= 7:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"


def ingest_odds() -> pd.DataFrame:
    """
    Load and clean betting odds data from vegas.txt and vegas_playoff.txt.

    Returns:
        DataFrame with cleaned odds data
    """
    logger.info("Loading betting odds...")

    paths = [
        os.path.join(DATA_DIR, "vegas.txt"),
        os.path.join(DATA_DIR, "vegas_playoff.txt"),
    ]

    dfs = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
            logger.info(f"  ✓ Loaded {p} ({len(df)} rows)")

    if not dfs:
        raise FileNotFoundError("No vegas odds files found in data/raw/")

    odds = pd.concat(dfs, ignore_index=True)
    odds.columns = [c.strip() for c in odds.columns]

    # Filter 2018-19 season
    odds["Date"] = pd.to_datetime(odds["Date"])
    mask = (odds["Date"] >= "2018-07-01") & (odds["Date"] <= "2019-06-30")
    odds = odds.loc[mask].copy()

    # Add season column
    odds["season"] = odds["Date"].dt.strftime("%Y-%m-%d").apply(detect_season)

    # Ensure GameId is standardized
    odds["GameId"] = odds["GameId"].astype(str).str.zfill(10)

    logger.info(f"✓ Odds ingested: {len(odds)} rows")
    return odds


def ingest_scores() -> pd.DataFrame:
    """
    Load and clean game scores/box scores from raw_scores.txt.

    Returns:
        DataFrame with cleaned score data
    """
    logger.info("Loading game scores...")

    raw_path = os.path.join(DATA_DIR, "raw_scores.txt")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Score file not found: {raw_path}")

    # Column names
    cols = [
        "GAME_DATE_EST", "GAME_SEQUENCE", "GAME_ID", "TEAM_ID",
        "TEAM_ABBREVIATION", "TEAM_CITY_NAME", "TEAM_WINS_LOSSES",
        "PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4",
        "PTS_OT1", "PTS_OT2", "PTS_OT3", "PTS_OT4", "PTS_OT5",
        "PTS_OT6", "PTS_OT7", "PTS_OT8", "PTS_OT9", "PTS_OT10",
        "PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB", "TOV"
    ]

    scores = pd.read_csv(raw_path, header=0, names=cols, dtype=str)

    # Convert numeric columns
    numeric_cols = [
        "GAME_SEQUENCE", "GAME_ID", "TEAM_ID",
        "PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4",
        "PTS_OT1", "PTS_OT2", "PTS_OT3", "PTS_OT4", "PTS_OT5",
        "PTS_OT6", "PTS_OT7", "PTS_OT8", "PTS_OT9", "PTS_OT10",
        "PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB", "TOV"
    ]

    for col in numeric_cols:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")

    # Parse dates
    scores["GAME_DATE_EST"] = pd.to_datetime(scores["GAME_DATE_EST"], errors="coerce")

    # Filter 2018-19 season
    mask = (scores["GAME_DATE_EST"] >= "2018-10-01") & \
           (scores["GAME_DATE_EST"] <= "2019-06-30")
    scores = scores[mask]

    # Drop invalid rows
    scores = scores.dropna(subset=["TEAM_ABBREVIATION", "PTS"])

    logger.info(f"✓ Scores ingested: {len(scores)} rows")
    return scores


def ingest_shots() -> pd.DataFrame:
    """
    Load and clean shot-level data from NBA_2019_Shots.csv.

    Returns:
        DataFrame with cleaned shot data
    """
    logger.info("Loading shot data...")

    raw_path = os.path.join(DATA_DIR, "NBA_2019_Shots.csv")

    if not os.path.exists(raw_path):
        logger.warning(f"Shot file not found: {raw_path}. Skipping shots.")
        return pd.DataFrame()

    shots = pd.read_csv(raw_path, low_memory=False)

    # Convert numeric columns
    numeric_cols = ["SHOT_DISTANCE", "LOC_X", "LOC_Y"]
    for col in numeric_cols:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")

    # Parse dates
    if "GAME_DATE" in shots.columns:
        shots["GAME_DATE"] = pd.to_datetime(shots["GAME_DATE"], errors="coerce")

    # Drop invalid rows
    shots = shots.dropna(subset=["GAME_ID", "TEAM_ID"])

    # Standardize GAME_ID
    shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)

    logger.info(f"✓ Shots ingested: {len(shots)} rows")
    return shots


def main():
    """Run full ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)

    try:
        # Ingest data
        odds = ingest_odds()
        scores = ingest_scores()
        shots = ingest_shots()

        # Save to intermediate files
        odds.to_csv(os.path.join(OUT_DIR, "odds_raw.csv"), index=False)
        scores.to_csv(os.path.join(OUT_DIR, "scores_raw.csv"), index=False)

        if not shots.empty:
            shots.to_csv(os.path.join(OUT_DIR, "shots_raw.csv"), index=False)

        logger.info("=" * 60)
        logger.info("✓ INGESTION COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
