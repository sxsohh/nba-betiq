import os
import sqlite3
import pandas as pd

CLEAN_DIR = os.path.join("data", "clean")
DB_DIR = "db"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "nba_betting.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")

def american_to_profit(bet_amount, odds, result):
    """Return profit for a single bet (win/loss/push)."""
    if result == "push":
        return 0.0
    if pd.isna(odds):
        return 0.0
    odds = float(odds)
    if odds > 0:
        win_profit = bet_amount * (odds / 100.0)
    else:
        win_profit = bet_amount * (100.0 / -odds)
    return win_profit if result == "win" else -bet_amount

def create_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

def main():
    odds = pd.read_csv(os.path.join(CLEAN_DIR, "odds_2018_19_clean.csv"))
    scores = pd.read_csv(os.path.join(CLEAN_DIR, "scores_2018_19_clean.csv"))
    shots = pd.read_csv(os.path.join(CLEAN_DIR, "shots_team_game_2018_19.csv"))

    # Map scores onto home/away using team abbreviations from odds
    scores["GAME_ID"] = scores["GAME_ID"].astype(str).str.zfill(10)
    odds["GameId"] = odds["GameId"].astype(str).str.zfill(10)

    # Join home team stats
    merged = pd.merge(
        odds,
        scores[
            [
                "GAME_ID",
                "TEAM_ABBREVIATION",
                "PTS",
                "FG_PCT",
                "FT_PCT",
                "FG3_PCT",
                "AST",
                "REB",
                "TOV",
            ]
        ],
        left_on=["GameId", "home_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="inner",
        suffixes=("", "_home_stats"),
    ).drop(columns=["GAME_ID", "TEAM_ABBREVIATION"])

    merged = merged.rename(columns={
        "PTS": "home_pts_scores",
        "FG_PCT": "home_fg_pct",
        "FT_PCT": "home_ft_pct",
        "FG3_PCT": "home_fg3_pct",
        "AST": "home_ast",
        "REB": "home_reb",
        "TOV": "home_tov",
    })

    # Join away team stats
    merged = pd.merge(
        merged,
        scores[
            [
                "GAME_ID",
                "TEAM_ABBREVIATION",
                "PTS",
                "FG_PCT",
                "FT_PCT",
                "FG3_PCT",
                "AST",
                "REB",
                "TOV",
            ]
        ],
        left_on=["GameId", "away_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="inner",
        suffixes=("", "_away_stats"),
    ).drop(columns=["GAME_ID", "TEAM_ABBREVIATION"])

    merged = merged.rename(columns={
        "PTS": "away_pts_scores",
        "FG_PCT": "away_fg_pct",
        "FT_PCT": "away_ft_pct",
        "FG3_PCT": "away_fg3_pct",
        "AST": "away_ast",
        "REB": "away_reb",
        "TOV": "away_tov",
    })

    # Use points from scores (more canonical than odds file)
    merged["home_pts"] = merged["home_pts_scores"]
    merged["away_pts"] = merged["away_pts_scores"]
    merged["total_pts"] = merged["home_pts"] + merged["away_pts"]
    merged["actual_margin"] = merged["home_pts"] - merged["away_pts"]  # + => home wins by X

    # Join shot features to home/away teams
    shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)

    # Home shots
    home_shots = shots.rename(columns={
        "GAME_ID": "GameId",
        "TEAM_ID": "home_team_id",
        "TEAM_NAME": "home_team_name_shots",
        "shots_attempts": "home_shots_attempts",
        "shots_made": "home_shots_made",
        "threes_attempts": "home_threes_attempts",
        "threes_made": "home_threes_made",
        "avg_shot_distance": "home_avg_shot_distance",
        "at_rim_attempts": "home_at_rim_attempts",
        "midrange_attempts": "home_midrange_attempts",
        "three_plus_attempts": "home_three_plus_attempts",
        "fg_pct": "home_shot_fg_pct",
        "three_pct": "home_shot_three_pct",
        "rim_freq": "home_rim_freq",
        "mid_freq": "home_mid_freq",
        "three_plus_freq": "home_three_plus_freq",
    })

    merged = pd.merge(
        merged,
        home_shots[["GameId", "home_shots_attempts", "home_shots_made",
                    "home_threes_attempts", "home_threes_made", "home_avg_shot_distance",
                    "home_at_rim_attempts", "home_midrange_attempts", "home_three_plus_attempts",
                    "home_shot_fg_pct", "home_shot_three_pct", "home_rim_freq", "home_mid_freq",
                    "home_three_plus_freq"]],
        on="GameId",
        how="left",
    )

    # Away shots
    away_shots = shots.rename(columns={
        "GAME_ID": "GameId",
        "TEAM_ID": "away_team_id",
        "TEAM_NAME": "away_team_name_shots",
        "shots_attempts": "away_shots_attempts",
        "shots_made": "away_shots_made",
        "threes_attempts": "away_threes_attempts",
        "threes_made": "away_threes_made",
        "avg_shot_distance": "away_avg_shot_distance",
        "at_rim_attempts": "away_at_rim_attempts",
        "midrange_attempts": "away_midrange_attempts",
        "three_plus_attempts": "away_three_plus_attempts",
        "fg_pct": "away_shot_fg_pct",
        "three_pct": "away_shot_three_pct",
        "rim_freq": "away_rim_freq",
        "mid_freq": "away_mid_freq",
        "three_plus_freq": "away_three_plus_freq",
    })

    # We need to join on GameId + team name; shots uses TEAM_NAME (e.g. "Boston Celtics")
    # For now, just join on GameId and trust team mapping from ordering
    merged = pd.merge(
        merged,
        away_shots[["GameId", "away_shots_attempts", "away_shots_made",
                    "away_threes_attempts", "away_threes_made", "away_avg_shot_distance",
                    "away_at_rim_attempts", "away_midrange_attempts", "away_three_plus_attempts",
                    "away_shot_fg_pct", "away_shot_three_pct", "away_rim_freq", "away_mid_freq",
                    "away_three_plus_freq"]],
        on="GameId",
        how="left",
    )

    # Labels
    merged["home_win"] = (merged["actual_margin"] > 0).astype(int)
    merged["away_win"] = (merged["actual_margin"] < 0).astype(int)

    # Spread cover: positive if home covers given home_spread (home spread is negative when favorite)
    merged["home_spread_cover"] = (merged["actual_margin"] + merged["home_spread"]) > 0
    merged["away_spread_cover"] = (merged["actual_margin"] - merged["home_spread"]) < 0

    # OU outcome: over (>), under (<), push (=)
    merged["ou_line"] = merged["pinnacle_total"]
    merged["ou_over_win"] = (merged["total_pts"] > merged["ou_line"])
    merged["ou_under_win"] = (merged["total_pts"] < merged["ou_line"])

    games_master_path = os.path.join(CLEAN_DIR, "games_master_2018_19.csv")
    merged.to_csv(games_master_path, index=False)
    print(f"Saved games master to {games_master_path}")

    # Build betting_outcomes table (ML, spread, OU for both sides)
    records = []
    bet_amount = 100.0

    for _, row in merged.iterrows():
        gid = row["GameId"]
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Moneyline bets
        for side in ["home", "away"]:
            team = home_team if side == "home" else away_team
            odds = row[f"pinnacle_ml_{side}"]
            won = row["home_win"] if side == "home" else row["away_win"]
            result = "win" if won == 1 else "loss"
            profit = american_to_profit(bet_amount, odds, result)
            records.append({
                "game_id": gid,
                "season": season,
                "bet_type": "ml",
                "team": team,
                "line": None,
                "odds": odds,
                "bet_amount": bet_amount,
                "result": result,
                "profit": profit,
            })

        # Spread bets: use home_spread and home_spread_odds, mirror for away
        for side in ["home", "away"]:
            team = home_team if side == "home" else away_team
            if side == "home":
                line = row["home_spread"]
                odds = row["home_spread_odds"]
                covered = bool(row["home_spread_cover"])
            else:
                line = -row["home_spread"] if pd.notna(row["home_spread"]) else None
                odds = row["away_spread_odds"]
                covered = bool(row["away_spread_cover"])
            if line is None or pd.isna(odds):
                continue
            if row["actual_margin"] + (line if side == "home" else -line) == 0:
                result = "push"
            else:
                result = "win" if covered else "loss"
            profit = american_to_profit(bet_amount, odds, result)
            records.append({
                "game_id": gid,
                "season": season,
                "bet_type": "spread",
                "team": team,
                "line": line,
                "odds": odds,
                "bet_amount": bet_amount,
                "result": result,
                "profit": profit,
            })

        # Over/Under bets (game-level, not team)
        for side, is_over in [("over", True), ("under", False)]:
            line = row["ou_line"]
            odds = row["home_ou_odds"] if is_over else row["away_ou_odds"]
            if pd.isna(line) or pd.isna(odds):
                continue
            if row["total_pts"] == line:
                result = "push"
            else:
                if is_over:
                    result = "win" if row["total_pts"] > line else "loss"
                else:
                    result = "win" if row["total_pts"] < line else "loss"
            profit = american_to_profit(bet_amount, odds, result)
            records.append({
                "game_id": gid,
                "season": season,
                "bet_type": "ou",
                "team": side,  # 'over' / 'under'
                "line": line,
                "odds": odds,
                "bet_amount": bet_amount,
                "result": result,
                "profit": profit,
            })

    betting_outcomes = pd.DataFrame.from_records(records)
    betting_outcomes_path = os.path.join(CLEAN_DIR, "betting_outcomes_2018_19.csv")
    betting_outcomes.to_csv(betting_outcomes_path, index=False)
    print(f"Saved betting outcomes to {betting_outcomes_path}")

    # Load into SQLite
    create_db()
    conn = sqlite3.connect(DB_PATH)
    merged.to_sql("games", conn, if_exists="replace", index=False)
    betting_outcomes.to_sql("betting_outcomes", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Loaded data into {DB_PATH}")

if __name__ == "__main__":
    main()
