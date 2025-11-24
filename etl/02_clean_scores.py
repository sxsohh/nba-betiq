import pandas as pd
import os

RAW_PATH = "data/raw/raw_scores.txt"
OUT_PATH = "data/clean/scores_2018_19_clean.csv"

def main():

    # Columns EXACTLY matching your file
    cols = [
        "GAME_DATE_EST",
        "GAME_SEQUENCE",
        "GAME_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_CITY_NAME",
        "TEAM_WINS_LOSSES",
        "PTS_QTR1",
        "PTS_QTR2",
        "PTS_QTR3",
        "PTS_QTR4",
        "PTS_OT1",
        "PTS_OT2",
        "PTS_OT3",
        "PTS_OT4",
        "PTS_OT5",
        "PTS_OT6",
        "PTS_OT7",
        "PTS_OT8",
        "PTS_OT9",
        "PTS_OT10",
        "PTS",
        "FG_PCT",
        "FT_PCT",
        "FG3_PCT",
        "AST",
        "REB",
        "TOV"
    ]

    print("📥 Loading raw scores...")
    scores = pd.read_csv(RAW_PATH, header=0, names=cols, dtype=str)

    # Convert correct columns to numeric
    numeric_cols = [
        "GAME_SEQUENCE","GAME_ID","TEAM_ID",
        "PTS_QTR1","PTS_QTR2","PTS_QTR3","PTS_QTR4",
        "PTS_OT1","PTS_OT2","PTS_OT3","PTS_OT4","PTS_OT5",
        "PTS_OT6","PTS_OT7","PTS_OT8","PTS_OT9","PTS_OT10",
        "PTS","FG_PCT","FT_PCT","FG3_PCT","AST","REB","TOV"
    ]

    for col in numeric_cols:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")

    # Fix date column
    scores["GAME_DATE_EST"] = pd.to_datetime(scores["GAME_DATE_EST"], errors="coerce")

    print("📅 Filtering for 2018–19 season using date range...")
    mask = (scores["GAME_DATE_EST"] >= "2018-10-01") & \
           (scores["GAME_DATE_EST"] <= "2019-06-30")
    scores = scores[mask]

    print("🧹 Dropping invalid rows...")
    scores = scores.dropna(subset=["TEAM_ABBREVIATION", "PTS"])

    # Add total_pts (final score)
    scores["TOTAL_PTS"] = scores["PTS"]

    os.makedirs("data/clean", exist_ok=True)
    scores.to_csv(OUT_PATH, index=False)

    print("✅ Saved cleaned scores →", OUT_PATH)
    print("📝 Total rows:", len(scores))


if __name__ == "__main__":
    main()
