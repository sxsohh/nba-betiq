import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

DB_PATH = os.path.join("db", "nba_betting.db")
MODELS_DIR = os.path.join("ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM games", conn)
    conn.close()

    # Target: did home cover vs book's spread?
    y = df["home_spread_cover"].astype(int)

    # Features: betting + team stats + shot profile
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

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"Spread model AUC: {auc:.3f}, Acc: {acc:.3f}")

    model_path = os.path.join(MODELS_DIR, "spread_model.pkl")
    joblib.dump(pipe, model_path)
    print(f"Saved spread model to {model_path}")

if __name__ == "__main__":
    main()
