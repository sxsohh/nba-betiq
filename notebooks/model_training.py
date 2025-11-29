import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib


games = pd.read_csv("data/clean/games_master_2018_19.csv")
outcomes = pd.read_csv("data/clean/betting_outcomes_2018_19.csv")

print("games:", games.shape)
print("outcomes:", outcomes.shape)

# Sanity check
games.head()


target_cols = ["home_win", "home_spread_cover", "ou_over_win"]
for c in target_cols:
    print(c, "in games:", c in games.columns)

# Features: team strength, shot profile, market info
feature_cols = [
    # Score / box score
    "home_pts_scores", "home_fg_pct", "home_fg3_pct", "home_ast", "home_reb", "home_tov",
    "away_pts_scores", "away_fg_pct", "away_fg3_pct", "away_ast", "away_reb", "away_tov",

    # Shot profile
    "home_shots_attempts", "home_shot_fg_pct", "home_avg_shot_distance", "home_rim_freq", "home_three_plus_freq",
    "away_shots_attempts", "away_shot_fg_pct", "away_avg_shot_distance", "away_rim_freq", "away_three_plus_freq",

    # Market signals
    "home_spread", "pinnacle_total", "avg_total",
    "percent_bet_ml_home", "percent_bet_spread_home", "percent_bet_ou_home",
    "vig_ml", "vig_spread", "vig_ou",
]

missing = [c for c in feature_cols if c not in games.columns]
print("Missing feature columns:", missing)

def train_eval_binary_model(df, feature_cols, target_col, model_name, save_path):
    print(f"\n==============================")
    print(f"Training model: {model_name} (target = {target_col})")
    print(f"==============================")

    data = df.dropna(subset=feature_cols + [target_col]).copy()

    X = data[feature_cols]
    y = data[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_test, y_prob)

    print(f"Accuracy: {acc:.3f}")
    print(f"AUC:      {auc:.3f}")
    print(f"Brier:    {brier:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipe, save_path)
    print(f"\n💾 Saved model to {save_path}")

    return {
        "model": pipe,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {"acc": acc, "auc": auc, "brier": brier},
    }


results = {}

os.makedirs("ml/models", exist_ok=True)

results["home_win"] = train_eval_binary_model(
    games,
    feature_cols,
    target_col="home_win",
    model_name="Home Moneyline Winner",
    save_path="ml/models/home_win_logreg.pkl"
)

results["spread"] = train_eval_binary_model(
    games,
    feature_cols,
    target_col="home_spread_cover",
    model_name="Home Spread Cover",
    save_path="ml/models/home_spread_logreg.pkl"
)

results["ou"] = train_eval_binary_model(
    games,
    feature_cols,
    target_col="ou_over_win",
    model_name="Over/Under Over",
    save_path="ml/models/ou_over_logreg.pkl"
)


from sklearn.model_selection import StratifiedKFold

data = games.dropna(subset=feature_cols + ["home_win"]).copy()
X = data[feature_cols]
y = data["home_win"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
])

param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
    "clf__penalty": ["l2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(best_model, "ml/models/home_win_logreg_tuned.pkl")
print("💾 Saved tuned home_win model → ml/models/home_win_logreg_tuned.pkl")
