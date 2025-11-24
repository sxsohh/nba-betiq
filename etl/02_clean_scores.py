import os
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "clean")
os.makedirs(OUT_DIR, exist_ok=True)

def detect_season_from_game_id(game_id: str) -> str:
    # Standard NBA ID format: 00218xxxxx => 2018-19 regular season
    game_id = str(game_id)
    year_code = game_id[3:5]  # '18'
    start_year = 2000 + int(year_code)
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"

def main():
    scores_path = os.path.join(DATA_DIR, "raw_scores.txt")
    scores = pd.read_csv(scores_path)

    scores.columns = [c.strip() for c in scores.columns]

    # Standardize
    scores["GAME_ID"] = scores["GAME_ID"].astype(str).str.zfill(10)
    scores["GAME_DATE_EST"] = pd.to_datetime(scores["GAME_DATE_EST"])

    # Only keep 2018-19 games
    scores["season"] = scores["GAME_ID"].apply(detect_season_from_game_id)
    scores = scores[scores["season"] == "2018-19"].copy()

    # Identify home vs away: we don't have explicit flag, so:
    # We'll infer home team by comparing to odds later.
    # For now, create per-game aggregated table.
    # Two rows per game: we can assign home/away later via join with odds.

    scores_out = scores[
        [
            "GAME_DATE_EST",
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_CITY_NAME",
            "TEAM_WINS_LOSSES",
            "PTS",
            "FG_PCT",
            "FT_PCT",
            "FG3_PCT",
            "AST",
            "REB",
            "TOV",
            "season",
        ]
    ].copy()

    out_path = os.path.join(OUT_DIR, "scores_2018_19_clean.csv")
    scores_out.to_csv(out_path, index=False)
    print(f"Saved cleaned scores to {out_path}")

if __name__ == "__main__":
    main()
