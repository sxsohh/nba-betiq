import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

DB_PATH = os.path.join("db", "nba_betting.db")
MODELS_DIR = os.path.join("ml", "models")

def load_games():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()
    return df

def evaluate_spread_model():
    df = load_games()
    model = joblib.load(os.path.join(MODELS_DIR, "spread_model.pkl"))

    feature_cols = [
        "home_spread",
        "home_spread_odds",
        "vig_spread",
        "percent_bet_spread_home",
        "percent_bet_spread_away",
        "home_fg_pct", "home_fg3_pct", "home_ast", "home_reb", "home_tov",
        "away_fg_pct", "away_fg3_pct", "away_ast", "away_reb", "away_tov",
        "home_avg_shot_distance",
        "home_rim_freq", "home_mid_freq", "home_three_plus_freq",
        "away_avg_shot_distance",
        "away_rim_freq", "away_mid_freq", "away_three_plus_freq",
    ]

    X = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True))
    y = df["home_spread_cover"].astype(int)

    y_proba = model.predict_proba(X)[:, 1]

    frac_pos, mean_pred = calibration_curve(y, y_proba, n_bins=10, strategy="uniform")

    print("Calibration (spread model):")
    for p_hat, freq in zip(mean_pred, frac_pos):
        print(f"Bucket predicted ~{p_hat:.2f}, actual cover rate {freq:.2f}")

    # ROI vs threshold, assuming -110 odds (~0.91 net on wins)
    thresholds = np.linspace(0.5, 0.7, 9)
    for t in thresholds:
        mask = y_proba >= t
        if mask.sum() < 50:
            continue
        win_rate = y[mask].mean()
        expected_roi = (win_rate * 0.91) - (1 - win_rate)
        print(f"Thresh {t:.2f}: games={mask.sum()}, win_rate={win_rate:.3f}, exp_ROI={expected_roi:.3f}")

def main():
    evaluate_spread_model()

if __name__ == "__main__":
    main()
