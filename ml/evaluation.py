"""
Model Evaluation and Metrics
Comprehensive evaluation of trained models including calibration analysis.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/processed/games_master_2018_19.csv"
MODEL_DIR = "ml/models"


def load_data():
    """Load and prepare test data."""
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)

    targets = {
        "home_win": df["home_win"],
        "spread": df["home_spread_cover"],
        "ou": df["ou_over_win"]
    }

    drop_cols = [
        "home_win", "home_spread_cover", "ou_over_win",
        "GAME_ID", "HOME_ABBR", "AWAY_ABBR",
        "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
        "home_pts", "away_pts", "total_pts",
        "final_home_pts", "final_away_pts", "spread_margin"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    return X, targets


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for logging

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Probability metrics
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0

    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "log_loss": logloss,
        "brier_score": brier,
        "confusion_matrix": cm
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print(f"MODEL: {metrics['model'].upper()}")
    print("=" * 60)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"ROC AUC:       {metrics['roc_auc']:.4f}")
    print(f"Log Loss:      {metrics['log_loss']:.4f}")
    print(f"Brier Score:   {metrics['brier_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("=" * 60)


def calculate_calibration(y_true, y_prob, n_bins=10):
    """
    Calculate calibration curve data.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (prob_true, prob_pred)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return prob_true, prob_pred


def evaluate_calibration(model, X_test, y_test, model_name: str):
    """
    Evaluate probability calibration.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of model

    Returns:
        Calibration data
    """
    logger.info(f"Evaluating calibration for {model_name}...")

    y_proba = model.predict_proba(X_test)[:, 1]

    prob_true, prob_pred = calculate_calibration(y_test, y_proba, n_bins=10)

    # Calculate Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))

    logger.info(f"  Expected Calibration Error: {ece:.4f}")

    return {
        "model": model_name,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "ece": ece
    }


def compare_models():
    """
    Compare XGBoost vs Calibrated LR for all prediction types.
    """
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    X, targets = load_data()

    all_metrics = []

    for target_name, y in targets.items():
        logger.info(f"\n--- {target_name.upper()} ---")

        y = y.fillna(0)

        # Split data
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Load models
        xgb_model = joblib.load(os.path.join(MODEL_DIR, f"{target_name}_xgb.pkl"))
        calib_model = joblib.load(os.path.join(MODEL_DIR, f"{target_name}_logreg_calibrated.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, f"{target_name}_scaler.pkl"))

        # Evaluate XGBoost
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, f"{target_name}_xgb")
        print_metrics(xgb_metrics)
        all_metrics.append(xgb_metrics)

        # Evaluate Calibrated LR
        X_test_scaled = scaler.transform(X_test)
        lr_metrics = evaluate_model(calib_model, X_test_scaled, y_test, f"{target_name}_lr")
        print_metrics(lr_metrics)
        all_metrics.append(lr_metrics)

        # Calibration analysis
        xgb_calib = evaluate_calibration(xgb_model, X_test, y_test, f"{target_name}_xgb")
        lr_calib = evaluate_calibration(calib_model, X_test_scaled, y_test, f"{target_name}_lr")

    # Summary table
    df_metrics = pd.DataFrame([
        {
            "model": m["model"],
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"],
            "brier_score": m["brier_score"]
        }
        for m in all_metrics
    ])

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(df_metrics.to_string(index=False))
    print("=" * 60)

    # Save metrics
    df_metrics.to_csv(os.path.join(MODEL_DIR, "evaluation_metrics.csv"), index=False)
    logger.info("✓ Saved evaluation metrics to ml/models/evaluation_metrics.csv")


def main():
    """Run full evaluation pipeline."""
    try:
        compare_models()

        logger.info("\n" + "=" * 60)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
