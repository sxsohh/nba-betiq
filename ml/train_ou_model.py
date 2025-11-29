import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from utils import load_games, build_model, save_model

def main():
    df = load_games()

    y = df["ou_over_win"]
    X = df.drop(columns=["home_win", "home_spread_cover", "ou_over_win"])

    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    xgb, calib_lr, scaler = build_model(X_train, y_train)

    preds = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]

    print("OU Accuracy:", accuracy_score(y_test, preds))
    print("OU ROC AUC:", roc_auc_score(y_test, proba))

    save_model(xgb, "ou_xgb")
    save_model(calib_lr, "ou_logreg_calibrated")
    save_model(scaler, "ou_scaler")

    print("✔ Saved OU models.")


if __name__ == "__main__":
    main()
