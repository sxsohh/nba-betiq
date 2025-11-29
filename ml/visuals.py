"""
Visualization Functions for NBA BetIQ
Generates all plots for model evaluation and analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Output directory
VISUALS_DIR = "visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)


def plot_calibration_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plot probability calibration curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of model for title
        save_path: Path to save figure (optional)
    """
    logger.info(f"Generating calibration curve for {model_name}...")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', label=f'{model_name}', linewidth=2)

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)

    # Calculate Expected Calibration Error
    ece = np.mean(np.abs(prob_true - prob_pred))

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curve: {model_name}\nECE: {ece:.4f}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of model
        save_path: Path to save figure
    """
    logger.info(f"Generating ROC curve for {model_name}...")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {model_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_public_betting_distribution(df, save_path=None):
    """
    Plot distribution of public betting percentages.

    Args:
        df: DataFrame with public betting data
        save_path: Path to save figure
    """
    logger.info("Generating public betting distribution...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Moneyline
    if "percent_bet_ml_home" in df.columns:
        axes[0].hist(df["percent_bet_ml_home"].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(50, color='red', linestyle='--', linewidth=2, label='50% (Even Split)')
        axes[0].set_xlabel('Public Bet % on Home', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Moneyline Public Betting', fontsize=12)
        axes[0].legend()

    # Spread
    if "percent_bet_spread_home" in df.columns:
        axes[1].hist(df["percent_bet_spread_home"].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(50, color='red', linestyle='--', linewidth=2, label='50% (Even Split)')
        axes[1].set_xlabel('Public Bet % on Home Spread', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Spread Public Betting', fontsize=12)
        axes[1].legend()

    # Over/Under
    if "percent_bet_ou_home" in df.columns:
        axes[2].hist(df["percent_bet_ou_home"].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(50, color='red', linestyle='--', linewidth=2, label='50% (Even Split)')
        axes[2].set_xlabel('Public Bet % on Over', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('O/U Public Betting', fontsize=12)
        axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_odds_movement_distribution(df, save_path=None):
    """
    Plot distribution of odds line movement.

    Args:
        df: DataFrame with opening/closing lines
        save_path: Path to save figure
    """
    logger.info("Generating odds movement distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate line movement (if we had opening lines, we'd use those)
    # For now, plot distribution of spreads
    if "home_spread" in df.columns:
        ax.hist(df["home_spread"].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Pick-em (0)')
        ax.set_xlabel('Home Team Spread', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Point Spreads', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_implied_vs_model_scatter(y_true, y_prob, implied_prob, save_path=None):
    """
    Scatter plot of implied probability vs model probability.

    Args:
        y_true: True outcomes
        y_prob: Model predicted probabilities
        implied_prob: Implied probabilities from odds
        save_path: Path to save figure
    """
    logger.info("Generating implied vs model probability scatter...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by outcome
    colors = ['red' if y == 0 else 'green' for y in y_true]

    ax.scatter(implied_prob, y_prob, alpha=0.5, c=colors, s=20)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Equal Probabilities')

    ax.set_xlabel('Implied Probability (from odds)', fontsize=12)
    ax.set_ylabel('Model Probability', fontsize=12)
    ax.set_title('Model vs Market: Probability Comparison', fontsize=14)
    ax.legend(['Equal', 'Loss', 'Win'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_ev_over_time(dates, cumulative_ev, save_path=None):
    """
    Plot expected value over time.

    Args:
        dates: List of dates
        cumulative_ev: Cumulative expected value
        save_path: Path to save figure
    """
    logger.info("Generating EV over time plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, cumulative_ev, linewidth=2, color='blue')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Break-even')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Expected Value ($)', fontsize=12)
    ax.set_title('Expected Value Over Time (2018-19 Season)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance from tree-based model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_path: Path to save figure
    """
    logger.info("Generating feature importance plot...")

    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(top_n), importance[indices], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def plot_vig_distribution(df, save_path=None):
    """
    Plot distribution of house edge (vig).

    Args:
        df: DataFrame with vig columns
        save_path: Path to save figure
    """
    logger.info("Generating vig distribution plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vig_cols = ["vig_ml", "vig_spread", "vig_ou"]
    titles = ["Moneyline Vig", "Spread Vig", "O/U Vig"]

    for i, (col, title) in enumerate(zip(vig_cols, titles)):
        if col in df.columns:
            data = df[col].dropna()
            axes[i].hist(data, bins=30, edgecolor='black', alpha=0.7)
            axes[i].axvline(data.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {data.mean():.4f}')
            axes[i].set_xlabel('Vig (House Edge)', fontsize=11)
            axes[i].set_ylabel('Frequency', fontsize=11)
            axes[i].set_title(title, fontsize=12)
            axes[i].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved to {save_path}")

    plt.close()


def generate_all_visuals():
    """
    Generate all visualization files.
    Requires trained models and processed data.
    """
    import joblib
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("GENERATING ALL VISUALIZATIONS")
    logger.info("=" * 60)

    # Load data
    df = pd.read_csv("data/processed/games_master_2018_19.csv")

    # Data visualizations
    plot_public_betting_distribution(
        df,
        save_path=os.path.join(VISUALS_DIR, "public_betting_distribution.png")
    )

    plot_odds_movement_distribution(
        df,
        save_path=os.path.join(VISUALS_DIR, "odds_movement_distribution.png")
    )

    plot_vig_distribution(
        df,
        save_path=os.path.join(VISUALS_DIR, "vig_distribution.png")
    )

    # Load model and generate model-specific plots
    try:
        model = joblib.load("ml/models/home_win_xgb.pkl")

        # Prepare data
        y = df["home_win"].fillna(0)
        drop_cols = [
            "home_win", "home_spread_cover", "ou_over_win",
            "GAME_ID", "HOME_ABBR", "AWAY_ABBR",
            "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
            "home_pts", "away_pts", "total_pts",
            "final_home_pts", "final_away_pts", "spread_margin"
        ]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        X = X.select_dtypes(include=[np.number]).fillna(0)

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC Curve
        plot_roc_curve(
            y_test, y_prob, model_name="Home Win XGBoost",
            save_path=os.path.join(VISUALS_DIR, "roc_curve.png")
        )

        # Calibration Curve
        plot_calibration_curve(
            y_test, y_prob, model_name="Home Win XGBoost",
            save_path=os.path.join(VISUALS_DIR, "prob_calibration_curve.png")
        )

        # Feature Importance
        plot_feature_importance(
            model, X.columns.tolist(),
            save_path=os.path.join(VISUALS_DIR, "feature_importance.png")
        )

        # Implied vs Model scatter
        if "prob_home_ml" in df.columns:
            implied = df.loc[X_test.index, "prob_home_ml"].fillna(0.5)
            plot_implied_vs_model_scatter(
                y_test, y_prob, implied,
                save_path=os.path.join(VISUALS_DIR, "implied_vs_model_scatter.png")
            )

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL VISUALIZATIONS GENERATED")
        logger.info(f"✓ Saved to {VISUALS_DIR}/")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.warning("Models not found. Train models first with: python ml/training.py")


if __name__ == "__main__":
    generate_all_visuals()
