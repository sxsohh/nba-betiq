import os
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "clean")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    shots_path = os.path.join(DATA_DIR, "NBA_2019_Shots.csv")
    if not os.path.exists(shots_path):
        raise FileNotFoundError("NBA_2019_Shots.csv not found in data/raw")

    shots = pd.read_csv(shots_path)

    # Normalize column names
    shots.columns = [c.strip() for c in shots.columns]

    # Keep only 2018-19 season rows if season is encoded
    # Many shot datasets have SEASON_2 like "2018-19"
    if "SEASON_2" in shots.columns:
        shots = shots[shots["SEASON_2"] == "2018-19"].copy()

    # Standardize GAME_ID to match odds/scores
    if "GAME_ID" in shots.columns:
        shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)
    else:
        raise ValueError("Expected GAME_ID column in shots file")

    # Basic flags
    shots["is_made"] = shots["SHOT_MADE"].astype(str).str.upper().isin(["TRUE", "1"])

    if "SHOT_TYPE" in shots.columns:
        shots["is_three"] = shots["SHOT_TYPE"].astype(str).str.contains("3PT")
    else:
        shots["is_three"] = False

    # Distance
    if "SHOT_DISTANCE" in shots.columns:
        shots["SHOT_DISTANCE"] = pd.to_numeric(shots["SHOT_DISTANCE"], errors="coerce")
    else:
        shots["SHOT_DISTANCE"] = None

    # Simple zones: at-rim vs mid vs three
    def zone_group(row):
        dist = row["SHOT_DISTANCE"]
        if pd.isna(dist):
            return "unknown"
        if dist <= 8:
            return "at_rim"
        elif dist <= 22:
            return "midrange"
        else:
            return "three_plus"

    shots["zone_group"] = shots.apply(zone_group, axis=1)

    # Group to team-game level
    group_cols = ["GAME_ID", "TEAM_ID", "TEAM_NAME"]
    agg = shots.groupby(group_cols).agg(
        shots_attempts=("is_made", "size"),
        shots_made=("is_made", "sum"),
        threes_attempts=("is_three", "sum"),
        threes_made=("is_three", lambda x: ((x) & shots.loc[x.index, "is_made"]).sum()),
        avg_shot_distance=("SHOT_DISTANCE", "mean"),
        at_rim_attempts=("zone_group", lambda x: (x == "at_rim").sum()),
        midrange_attempts=("zone_group", lambda x: (x == "midrange").sum()),
        three_plus_attempts=("zone_group", lambda x: (x == "three_plus").sum()),
    ).reset_index()

    # Rates
    agg["fg_pct"] = agg["shots_made"] / agg["shots_attempts"]
    agg["three_pct"] = agg["threes_made"] / agg["threes_attempts"].replace(0, pd.NA)
    agg["rim_freq"] = agg["at_rim_attempts"] / agg["shots_attempts"]
    agg["mid_freq"] = agg["midrange_attempts"] / agg["shots_attempts"]
    agg["three_plus_freq"] = agg["three_plus_attempts"] / agg["shots_attempts"]

    out_path = os.path.join(OUT_DIR, "shots_team_game_2018_19.csv")
    agg.to_csv(out_path, index=False)
    print(f"Saved team-game shot features to {out_path}")

if __name__ == "__main__":
    main()
