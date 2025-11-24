import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "clean")
os.makedirs(OUT_DIR, exist_ok=True)

def american_to_prob(odds):
    if pd.isna(odds):
        return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    else:
        return 100.0 / (odds + 100.0)

def detect_season(date_str):
    # Date is YYYY-MM-DD
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    # NBA season crosses years. 2018-10 to 2019-06 => 2018-19
    if dt.month >= 7:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"

def load_and_concat_odds():
    paths = [
        os.path.join(DATA_DIR, "vegas.txt"),
        os.path.join(DATA_DIR, "vegas_playoff.txt"),
    ]
    dfs = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No vegas odds files found.")
    odds = pd.concat(dfs, ignore_index=True)
    return odds

def main():
    odds = load_and_concat_odds()

    # Standardize column names just in case
    odds.columns = [c.strip() for c in odds.columns]

    # Filter 2018-19 season via Date (2018-07-01 to 2019-06-30)
    odds["Date"] = pd.to_datetime(odds["Date"])
    mask = (odds["Date"] >= "2018-07-01") & (odds["Date"] <= "2019-06-30")
    odds = odds.loc[mask].copy()

    # Season string
    odds["season"] = odds["Date"].dt.strftime("%Y-%m-%d").apply(detect_season)

    # Ensure GameId is string with leading zeros
    odds["GameId"] = odds["GameId"].astype(str).str.zfill(10)

    # Pivot to game-level: one row per game, home+away columns
    # We assume Location is "home" or "away"
    def side_prefix(row):
        return "home" if row["Location"].lower() == "home" else "away"

    odds["side"] = odds["Location"].str.lower().map({"home": "home", "away": "away"})

    # Keep a subset of columns to pivot
    cols_keep = [
        "Date", "season", "GameId", "Team", "OppTeam", "side",
        "Pinnacle_ML", "Average_Line_ML",
        "Pinnacle_Line_Spread", "Pinnacle_Odds_Spread",
        "Average_Line_Spread", "Average_Odds_Spread",
        "Pinnacle_Line_OU", "Pinnacle_Odds_OU",
        "Average_Line_OU", "Average_Odds_OU",
        "PercentBet_ML", "PercentBet_Spread", "PercentBet_OU",
        "Pts", "Total"
    ]

    odds_sub = odds[cols_keep].copy()

    # We'll pivot manually
    def pivot_side(df, side):
        s = df[df["side"] == side]
        s = s.rename(columns={
            "Team": f"{side}_team",
            "OppTeam": f"{'home' if side=='away' else 'away'}_team",
            "Pinnacle_ML": f"pinnacle_ml_{side}",
            "Average_Line_ML": f"avg_ml_{side}",
            "Pinnacle_Line_Spread": f"{side}_spread",
            "Pinnacle_Odds_Spread": f"{side}_spread_odds",
            "Average_Line_Spread": f"{side}_avg_spread",
            "Average_Odds_Spread": f"{side}_avg_spread_odds",
            "Pinnacle_Line_OU": "pinnacle_total",  # same for both sides
            "Pinnacle_Odds_OU": f"{side}_ou_odds",
            "Average_Line_OU": "avg_total",
            "Average_Odds_OU": f"{side}_avg_ou_odds",
            "PercentBet_ML": f"percent_bet_ml_{side}",
            "PercentBet_Spread": f"percent_bet_spread_{side}",
            "PercentBet_OU": f"percent_bet_ou_{side}",
            "Pts": f"{side}_pts",
            "Total": "total_pts"  # same for both
        })
        s = s.drop(columns=["side"])
        return s

    home = pivot_side(odds_sub, "home")
    away = pivot_side(odds_sub, "away")

    merged = pd.merge(
        home,
        away[
            [
                "GameId",
                "away_team",
                "pinnacle_ml_away",
                "avg_ml_away",
                "away_spread",
                "away_spread_odds",
                "away_avg_spread",
                "away_avg_spread_odds",
                "away_ou_odds",
                "away_avg_ou_odds",
                "percent_bet_ml_away",
                "percent_bet_spread_away",
                "percent_bet_ou_away",
                "away_pts",
            ]
        ],
        on="GameId",
        how="inner",
        suffixes=("", "_y"),
    )

    # Compute vigs using Pinnacle ML odds
    merged["prob_home_ml"] = merged["pinnacle_ml_home"].apply(american_to_prob)
    merged["prob_away_ml"] = merged["pinnacle_ml_away"].apply(american_to_prob)
    merged["vig_ml"] = merged["prob_home_ml"] + merged["prob_away_ml"] - 1.0

    # Spread vig (assume home vs away spread odds)
    merged["prob_home_spread"] = merged["home_spread_odds"].apply(american_to_prob)
    merged["prob_away_spread"] = merged["away_spread_odds"].apply(american_to_prob)
    merged["vig_spread"] = (
        merged["prob_home_spread"].fillna(0) + merged["prob_away_spread"].fillna(0) - 1.0
    )

    # OU vig: use home as "over", away as "under"
    merged["prob_over"] = merged["home_ou_odds"].apply(american_to_prob)
    merged["prob_under"] = merged["away_ou_odds"].apply(american_to_prob)
    merged["vig_ou"] = (
        merged["prob_over"].fillna(0) + merged["prob_under"].fillna(0) - 1.0
    )

    out_path = os.path.join(OUT_DIR, "odds_2018_19_clean.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved cleaned odds to {out_path}")

if __name__ == "__main__":
    main()
