import pandas as pd
import numpy as np
import sqlite3
import os

CLEAN_DIR = "data/clean"
OUT_GAMES = f"{CLEAN_DIR}/games_master_2018_19.csv"
OUT_OUTCOMES = f"{CLEAN_DIR}/betting_outcomes_2018_19.csv"
DB_PATH = "db/nba_betting.db"


# ---------------------------------------------------
# TEAM NAME → ABBR MAP (FULL NAMES IN ODDS FILE)
# ---------------------------------------------------
TEAM_NAME_TO_ABBR = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "L.A. Clippers": "LAC",
    "L.A. Lakers": "LAL",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "LA Clippers": "LAC",
    "Los Angeles": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS"
}


# ---------------------------------------------------
# LOAD CLEANED FILES + FIX GAME_ID + MAP TEAM NAMES
# ---------------------------------------------------
def load_clean_data():
    print("📥 Loading cleaned files...")

    odds = pd.read_csv(f"{CLEAN_DIR}/odds_2018_19_clean.csv", dtype=str)
    scores = pd.read_csv(f"{CLEAN_DIR}/scores_2018_19_clean.csv", dtype=str)
    shots = pd.read_csv(f"{CLEAN_DIR}/shots_team_game_2018_19.csv", dtype=str)

    print(f"✔ odds:   {odds.shape}")
    print(f"✔ scores: {scores.shape}")
    print(f"✔ shots:  {shots.shape}")

    # Fix GAME_ID in odds
    if "GameId" in odds.columns:
        odds = odds.rename(columns={"GameId": "GAME_ID"})

    odds["GAME_ID"] = (
        odds["GAME_ID"].astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(10)
    )

    # Fix GAME_ID in scores/shots
    for df in [scores, shots]:
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)

    # Fix team names in odds
    odds["HOME_ABBR"] = odds["home_team"].map(TEAM_NAME_TO_ABBR)
    odds["AWAY_ABBR"] = odds["away_team"].map(TEAM_NAME_TO_ABBR)

    missing = odds[odds["HOME_ABBR"].isna() | odds["AWAY_ABBR"].isna()]
    if len(missing) > 0:
        print("❌ Missing mappings:")
        print(missing[["home_team","away_team"]])
        raise SystemExit()

    return odds, scores, shots


# ---------------------------------------------------
# BUILD TEAM_ABBREVIATION → TEAM_ID MAP
# ---------------------------------------------------
def build_team_map(scores):
    print("🗺  Building team map...")

    scores["TEAM_ID"] = scores["TEAM_ID"].astype(str)
    team_map = (
        scores.groupby("TEAM_ABBREVIATION")["TEAM_ID"]
        .first()
        .to_dict()
    )

    print("✔ Team map entries:", len(team_map))
    return team_map


# ---------------------------------------------------
# ATTACH HOME/AWAY TEAM_ID TO ODDS
# ---------------------------------------------------
def attach_home_away_ids(odds, team_map):
    print("🏠 Attaching HOME/AWAY TEAM_ID...")

    odds["HOME_TEAM_ID"] = odds["HOME_ABBR"].map(team_map)
    odds["AWAY_TEAM_ID"] = odds["AWAY_ABBR"].map(team_map)

    missing = odds[odds["HOME_TEAM_ID"].isna() | odds["AWAY_TEAM_ID"].isna()]
    if len(missing) > 0:
        print("⚠ Missing team IDs:")
        print(missing[["home_team", "away_team"]].head())

    return odds


# ---------------------------------------------------
# BUILD HOME/AWAY SCORE TABLES
# ---------------------------------------------------
def build_home_away_scores(scores, odds):
    print("📊 Building home/away score tables...")

    scores["TEAM_ID"] = scores["TEAM_ID"].astype(str)

    # HOME
    home = scores.merge(
        odds[["GAME_ID", "HOME_TEAM_ID"]],
        left_on=["GAME_ID", "TEAM_ID"],
        right_on=["GAME_ID", "HOME_TEAM_ID"],
        how="inner"
    ).add_prefix("HOME_")

    # AWAY
    away = scores.merge(
        odds[["GAME_ID", "AWAY_TEAM_ID"]],
        left_on=["GAME_ID", "TEAM_ID"],
        right_on=["GAME_ID", "AWAY_TEAM_ID"],
        how="inner"
    ).add_prefix("AWAY_")

    print("✔ home:", home.shape)
    print("✔ away:", away.shape)

    games = home.merge(
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        how="inner"
    )

    print("✔ game rows:", games.shape)
    return games


# ---------------------------------------------------
# MERGE ODDS
# ---------------------------------------------------
def merge_odds_into_games(games, odds):
    print("📈 Merging odds...")

    # The games df has HOME_GAME_ID from the prefix, rename to GAME_ID
    if "HOME_GAME_ID" in games.columns and "GAME_ID" not in games.columns:
        games = games.rename(columns={"HOME_GAME_ID": "GAME_ID"})

    if "GameId" in odds.columns:
        odds = odds.rename(columns={"GameId": "GAME_ID"})

    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    odds["GAME_ID"] = odds["GAME_ID"].astype(str).str.zfill(10)

    merged = games.merge(odds, on="GAME_ID", how="inner")

    print("✔ rows after odds merge:", merged.shape)
    return merged


# ---------------------------------------------------
# MERGE SHOT FEATURES
# ---------------------------------------------------
def merge_shots(games, shots):
    print("🎯 Merging shots...")

    # Use HOME_TEAM_ID_y / AWAY_TEAM_ID_y from odds merge, or fall back
    home_team_col = "HOME_TEAM_ID_y" if "HOME_TEAM_ID_y" in games.columns else "HOME_TEAM_ID"
    away_team_col = "AWAY_TEAM_ID_y" if "AWAY_TEAM_ID_y" in games.columns else "AWAY_TEAM_ID"

    # HOME SHOTS
    home = shots.rename(columns={
        "TEAM_ID": home_team_col,
        "shots_attempts": "HOME_shots_attempts",
        "shots_made": "HOME_shots_made",
        "threes_attempts": "HOME_threes_attempts",
        "threes_made": "HOME_threes_made",
        "avg_shot_distance": "HOME_avg_shot_distance",
        "at_rim_attempts": "HOME_at_rim_attempts",
        "midrange_attempts": "HOME_midrange_attempts",
        "three_plus_attempts": "HOME_three_plus_attempts",
        "fg_pct": "HOME_fg_pct",
        "three_pct": "HOME_three_pct",
        "rim_freq": "HOME_rim_freq",
        "mid_freq": "HOME_mid_freq",
        "three_plus_freq": "HOME_three_plus_freq"
    })

    # AWAY SHOTS
    away = shots.rename(columns={
        "TEAM_ID": away_team_col,
        "shots_attempts": "AWAY_shots_attempts",
        "shots_made": "AWAY_shots_made",
        "threes_attempts": "AWAY_threes_attempts",
        "threes_made": "AWAY_threes_made",
        "avg_shot_distance": "AWAY_avg_shot_distance",
        "at_rim_attempts": "AWAY_at_rim_attempts",
        "midrange_attempts": "AWAY_midrange_attempts",
        "three_plus_attempts": "AWAY_three_plus_attempts",
        "fg_pct": "AWAY_fg_pct",
        "three_pct": "AWAY_three_pct",
        "rim_freq": "AWAY_rim_freq",
        "mid_freq": "AWAY_mid_freq",
        "three_plus_freq": "AWAY_three_plus_freq"
    })

    # MERGE HOME
    merged = games.merge(
        home,
        left_on=["GAME_ID", home_team_col],
        right_on=["GAME_ID", home_team_col],
        how="left"
    )

    # MERGE AWAY
    merged = merged.merge(
        away,
        left_on=["GAME_ID", away_team_col],
        right_on=["GAME_ID", away_team_col],
        how="left"
    )

    print("✔ rows after merging shots:", merged.shape)
    return merged

# ---------------------------------------------------
# BUILD BETTING OUTCOMES
# ---------------------------------------------------
def build_betting_outcomes(games):
    print("💰 Building betting outcomes...")

    df = games.copy()

    # Find the correct column names for team IDs
    home_team_col = "HOME_TEAM_ID_y" if "HOME_TEAM_ID_y" in df.columns else "HOME_TEAM_ID"
    away_team_col = "AWAY_TEAM_ID_y" if "AWAY_TEAM_ID_y" in df.columns else "AWAY_TEAM_ID"

    df["HOME_PTS"] = pd.to_numeric(df["HOME_PTS"], errors="coerce")
    df["AWAY_PTS"] = pd.to_numeric(df["AWAY_PTS"], errors="coerce")

    df["actual_margin"] = df["HOME_PTS"] - df["AWAY_PTS"]
    df["home_win"] = (df["HOME_PTS"] > df["AWAY_PTS"]).astype(int)
    df["away_win"] = 1 - df["home_win"]

    # Rename to standard names for output
    df = df.rename(columns={home_team_col: "HOME_TEAM_ID", away_team_col: "AWAY_TEAM_ID"})

    return df[[
        "GAME_ID",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "HOME_PTS",
        "AWAY_PTS",
        "actual_margin",
        "home_win",
        "away_win"
    ]]


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    odds, scores, shots = load_clean_data()

    team_map = build_team_map(scores)
    odds = attach_home_away_ids(odds, team_map)

    games = build_home_away_scores(scores, odds)
    games = merge_odds_into_games(games, odds)
    games = merge_shots(games, shots)

    outcomes = build_betting_outcomes(games)

    games.to_csv(OUT_GAMES, index=False)
    outcomes.to_csv(OUT_OUTCOMES, index=False)

    print("✔ Saved:", OUT_GAMES)
    print("✔ Saved:", OUT_OUTCOMES)

    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Remove duplicate columns (case-insensitive) before saving to SQLite
    games_dedup = games.loc[:, ~games.columns.str.lower().duplicated()]
    games_dedup.to_sql("games", conn, if_exists="replace", index=False)
    outcomes.to_sql("betting_outcomes", conn, if_exists="replace", index=False)
    conn.close()

    print("🎉 DONE!")


if __name__ == "__main__":
    main()
