"""
ML Model Training Pipeline
Trains XGBoost and Calibrated Logistic Regression models for NBA betting predictions.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/processed/games_master_2018_19.csv"
MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare_data():
    """
    Load master dataset and prepare features/targets.

    Returns:
        Tuple of (features DataFrame, targets dict)
    """
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Create targets
    targets = {
        "home_win": df["home_win"],
        "spread": df["home_spread_cover"],
        "ou": df["ou_over_win"]
    }

    # Drop target columns and identifiers
    drop_cols = [
        "home_win", "home_spread_cover", "ou_over_win",
        "GAME_ID", "HOME_ABBR", "AWAY_ABBR",
        "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
        "home_pts", "away_pts", "total_pts",
        "final_home_pts", "final_away_pts", "spread_margin"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Fill NaN with 0
    X = X.fillna(0)

    logger.info(f"Features: {X.shape[1]} columns, {X.shape[0]} samples")
    return X, targets


def train_xgboost(X_train, y_train, X_test, y_test, model_name: str):
    """
    Train XGBoost classifier with optimized hyperparameters.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for saving model

    Returns:
        Trained XGBoost model
    """
    logger.info(f"Training XGBoost for {model_name}...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    logger.info(f"  XGB Accuracy: {acc:.4f}")
    logger.info(f"  XGB ROC AUC:  {auc:.4f}")
    logger.info(f"  XGB Log Loss: {logloss:.4f}")
    logger.info(f"  XGB Brier:    {brier:.4f}")

    return model


def train_calibrated_lr(X_train, y_train, X_test, y_test, model_name: str):
    """
    Train calibrated logistic regression.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for saving model

    Returns:
        Tuple of (scaler, calibrated model)
    """
    logger.info(f"Training Calibrated LR for {model_name}...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    lr = LogisticRegression(max_iter=2000, random_state=42)

    # Calibrate with cross-validation
    calibrated = CalibratedClassifierCV(lr, cv=5, method='sigmoid')
    calibrated.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = calibrated.predict(X_test_scaled)
    y_proba = calibrated.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    logger.info(f"  LR Accuracy:  {acc:.4f}")
    logger.info(f"  LR ROC AUC:   {auc:.4f}")
    logger.info(f"  LR Log Loss:  {logloss:.4f}")
    logger.info(f"  LR Brier:     {brier:.4f}")

    return scaler, calibrated


def save_models(model, scaler, calibrated, prefix: str):
    """
    Save trained models to disk.

    Args:
        model: XGBoost model
        scaler: StandardScaler
        calibrated: Calibrated logistic regression
        prefix: Prefix for model files (e.g., 'home_win', 'spread', 'ou')
    """
    joblib.dump(model, os.path.join(MODEL_DIR, f"{prefix}_xgb.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{prefix}_scaler.pkl"))
    joblib.dump(calibrated, os.path.join(MODEL_DIR, f"{prefix}_logreg_calibrated.pkl"))

    logger.info(f"✓ Saved {prefix} models")


def train_all_models():
    """
    Train all three types of models: home_win, spread, ou.
    """
    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    X, targets = load_and_prepare_data()

    # Train models for each target
    for target_name, y in targets.items():
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"TRAINING: {target_name.upper()}")
        logger.info("-" * 60)

        # Handle missing values in target
        y = y.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Train XGBoost
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test, target_name)

        # Train Calibrated LR
        scaler, calibrated = train_calibrated_lr(X_train, y_train, X_test, y_test, target_name)

        # Save models
        save_models(xgb_model, scaler, calibrated, target_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)


def main():
    """Run full training pipeline."""
    try:
        train_all_models()

        # Save feature columns for production use
        X, _ = load_and_prepare_data()
        feature_cols = X.columns.tolist()
        joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))
        logger.info(f"✓ Saved feature columns ({len(feature_cols)} features)")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
