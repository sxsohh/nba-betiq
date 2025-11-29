"""
ETL Step 2: Clean and Merge
Cleans raw data and merges betting odds with game scores.
"""
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"

# Team name mapping (odds files use full names, scores use abbreviations)
TEAM_NAME_TO_ABBR = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "L.A. Clippers": "LAC", "L.A. Lakers": "LAL", "LA Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS"
}


def pivot_odds_to_game_level(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot odds from team-level rows to game-level rows.
    Each game should have one row with home/away columns.
    """
    logger.info("Pivoting odds to game level...")

    # Separate home and away rows
    home_odds = odds[odds["Location"].str.lower() == "home"].copy()
    away_odds = odds[odds["Location"].str.lower() == "away"].copy()

    # Rename columns with home/away prefixes
    home_odds = home_odds.rename(columns={
        "Team": "home_team",
        "Pinnacle_ML": "pinnacle_ml_home",
        "Pinnacle_Line_Spread": "home_spread",
        "Pinnacle_Odds_Spread": "home_spread_odds",
        "Pinnacle_Line_OU": "pinnacle_total",
        "Pinnacle_Odds_OU": "home_ou_odds",
        "PercentBet_ML": "percent_bet_ml_home",
        "PercentBet_Spread": "percent_bet_spread_home",
        "PercentBet_OU": "percent_bet_ou_home",
        "Pts": "home_pts",
        "Total": "total_pts"
    })

    away_odds = away_odds.rename(columns={
        "Team": "away_team",
        "Pinnacle_ML": "pinnacle_ml_away",
        "Pinnacle_Line_Spread": "away_spread",
        "Pinnacle_Odds_Spread": "away_spread_odds",
        "Pinnacle_Odds_OU": "away_ou_odds",
        "PercentBet_ML": "percent_bet_ml_away",
        "PercentBet_Spread": "percent_bet_spread_away",
        "PercentBet_OU": "percent_bet_ou_away",
        "Pts": "away_pts"
    })

    # Merge home and away
    merged = pd.merge(
        home_odds[["Date", "season", "GameId", "home_team", "pinnacle_ml_home",
                   "home_spread", "home_spread_odds", "pinnacle_total",
                   "home_ou_odds", "percent_bet_ml_home", "percent_bet_spread_home",
                   "percent_bet_ou_home", "home_pts", "total_pts"]],
        away_odds[["GameId", "away_team", "pinnacle_ml_away", "away_spread",
                   "away_spread_odds", "away_ou_odds", "percent_bet_ml_away",
                   "percent_bet_spread_away", "percent_bet_ou_away", "away_pts"]],
        on="GameId",
        how="inner"
    )

    # Map team names to abbreviations
    merged["HOME_ABBR"] = merged["home_team"].map(TEAM_NAME_TO_ABBR)
    merged["AWAY_ABBR"] = merged["away_team"].map(TEAM_NAME_TO_ABBR)

    # Check for missing mappings
    missing = merged[merged["HOME_ABBR"].isna() | merged["AWAY_ABBR"].isna()]
    if len(missing) > 0:
        logger.warning(f"Missing team name mappings for {len(missing)} rows")

    logger.info(f"✓ Pivoted odds: {len(merged)} games")
    return merged


def pivot_scores_to_game_level(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot scores from team-level rows to game-level rows.
    """
    logger.info("Pivoting scores to game level...")

    # Standardize GAME_ID
    scores["GAME_ID"] = scores["GAME_ID"].astype(str).str.zfill(10)

    # Separate home and away (home teams have GAME_SEQUENCE = 1)
    home_scores = scores[scores["GAME_SEQUENCE"] == 1].copy()
    away_scores = scores[scores["GAME_SEQUENCE"] == 2].copy()

    # Prefix columns
    home_cols_rename = {col: f"HOME_{col}" for col in home_scores.columns
                        if col not in ["GAME_ID", "GAME_DATE_EST"]}
    away_cols_rename = {col: f"AWAY_{col}" for col in away_scores.columns
                        if col not in ["GAME_ID"]}

    home_scores = home_scores.rename(columns=home_cols_rename)
    away_scores = away_scores.rename(columns=away_cols_rename)

    # Merge
    merged_scores = pd.merge(
        home_scores,
        away_scores,
        on="GAME_ID",
        how="inner"
    )

    logger.info(f"✓ Pivoted scores: {len(merged_scores)} games")
    return merged_scores


def merge_odds_and_scores(odds: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge odds and scores on GAME_ID and team abbreviations.
    """
    logger.info("Merging odds and scores...")

    # Rename GameId to GAME_ID for consistency
    odds = odds.rename(columns={"GameId": "GAME_ID"})

    # Merge on GAME_ID, HOME_ABBR, AWAY_ABBR
    merged = pd.merge(
        odds,
        scores,
        left_on=["GAME_ID", "HOME_ABBR", "AWAY_ABBR"],
        right_on=["GAME_ID", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"],
        how="inner"
    )

    logger.info(f"✓ Merged dataset: {len(merged)} games")
    return merged


def main():
    """Run clean and merge pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: CLEAN AND MERGE")
    logger.info("=" * 60)

    try:
        # Load raw data
        odds = pd.read_csv(os.path.join(DATA_DIR, "odds_raw.csv"))
        scores = pd.read_csv(os.path.join(DATA_DIR, "scores_raw.csv"))

        # Pivot to game level
        odds_games = pivot_odds_to_game_level(odds)
        scores_games = pivot_scores_to_game_level(scores)

        # Merge
        merged = merge_odds_and_scores(odds_games, scores_games)

        # Save
        out_path = os.path.join(DATA_DIR, "games_merged.csv")
        merged.to_csv(out_path, index=False)

        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE: {out_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Clean/merge failed: {e}")
        raise


if __name__ == "__main__":
    main()
