import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib


def load_games():
    # Try different paths for data file
    paths = [
        "data/clean/games_master_2018_19.csv",
        "data/processed/games_master_2018_19.csv",
        "../data/processed/games_master_2018_19.csv"
    ]

    df = None
    for path in paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError(f"Could not find games_master_2018_19.csv in any of: {paths}")

    # Standardize numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Moneyline outcome
    df["home_win"] = (df["HOME_PTS"] > df["AWAY_PTS"]).astype(int)

    # Spread outcome
    df["spread_margin"] = df["HOME_PTS"] - df["AWAY_PTS"]
    df["home_spread_cover"] = ((df["spread_margin"] + df["home_spread"]) > 0).astype(int)

    # Over/Under outcome
    df["ou_over_win"] = (df["total_pts"] > df["pinnacle_total"]).astype(int)

    # Remove identifier / text columns (bad for ML)
    drop_cols = [
        "Date", "season", "home_team", "away_team",
        "TEAM_NAME_x", "TEAM_NAME_y", "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ABBREVIATION"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert everything possible to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def build_model(X_train, y_train):

    # --- FIX: eliminate NaNs ---
    X_train = X_train.fillna(0)
    y_train = y_train.fillna(0)

    # XGBoost model
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    # Calibrated Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=2000)
    calib_lr = CalibratedClassifierCV(lr, cv=5)
    calib_lr.fit(X_train_scaled, y_train)

    return xgb, calib_lr, scaler


def save_model(model, name):
    import os
    # Handle both running from project root and from ml/ directory
    paths = ["ml/models", "models", "../ml/models"]
    model_dir = None

    for path in paths:
        if os.path.exists(path):
            model_dir = path
            break

    if model_dir is None:
        # Create the directory
        os.makedirs("models", exist_ok=True)
        model_dir = "models"

    joblib.dump(model, f"{model_dir}/{name}.pkl")
