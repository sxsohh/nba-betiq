import pandas as pd
import os

RAW_18 = "data/raw/NBA_2018_Shots.csv"   # 2017-18 season (not used)
RAW_19 = "data/raw/NBA_2019_Shots.csv"   # 2018-19 season
OUT_PATH = "data/clean/shots_2018_19_clean.csv"

def main():

    print("📥 Loading raw shots for 2018–19...")

    # Load only the correct season (2018-19)
    shots = pd.read_csv(RAW_19, low_memory=False)

    print("✔ Loaded", len(shots), "raw shots")

    # Fix types
    numeric_cols = ["SHOT_DISTANCE", "LOC_X", "LOC_Y"]
    for col in numeric_cols:
        shots[col] = pd.to_numeric(shots[col], errors="coerce")

    # Convert date
    shots["GAME_DATE"] = pd.to_datetime(shots["GAME_DATE"], errors="coerce")

    # Keep only valid rows (drop garbage)
    shots = shots.dropna(subset=["GAME_ID", "TEAM_ID", "PLAYER_ID"])

    # GAME_ID must be string
    shots["GAME_ID"] = shots["GAME_ID"].astype(str)

    os.makedirs("data/clean", exist_ok=True)
    shots.to_csv(OUT_PATH, index=False)

    print("✅ Saved 2018–19 cleaned shots →", OUT_PATH)
    print("📝 Total rows:", len(shots))


if __name__ == "__main__":
    main()
