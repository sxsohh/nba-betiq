import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from utils import load_games, build_model, save_model

def main():
    df = load_games()

    y = df["home_win"]
    X = df.drop(columns=["home_win", "home_spread_cover", "ou_over_win"])

    # Only keep numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    xgb, calib_lr, scaler = build_model(X_train, y_train)

    # Evaluate
    preds = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]

    print("Moneyline Accuracy:", accuracy_score(y_test, preds))
    print("Moneyline ROC AUC:", roc_auc_score(y_test, proba))

    # Save
    save_model(xgb, "home_win_xgb")
    save_model(calib_lr, "home_win_logreg_calibrated")
    save_model(scaler, "home_win_scaler")

    print("✔ Saved home win models.")


if __name__ == "__main__":
    main()
