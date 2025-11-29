import os
import sqlite3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from . import sql_templates
from .llm_query import build_gambling_harm_prompt

DB_PATH = os.path.join("db", "nba_betting.db")

app = FastAPI(title="NBA BetIQ API")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NBA BetIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-vercel-domain.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "NBA BetIQ API is running"}


class TeamSeasonQuery(BaseModel):
    team: str
    season: str
    stake: float = 100.0

class HouseEdgeQuery(BaseModel):
    pass

from typing import Optional, Dict

def run_sql(query: str, params: Optional[Dict] = None) -> pd.DataFrame:

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows returned."
    return df.to_markdown(index=False)

@app.post("/team-profit")
def team_profit(q: TeamSeasonQuery):
    df = run_sql(sql_templates.TEAM_PROFIT_TEMPLATE, {"team": q.team, "season": q.season})
    md = df_to_markdown(df)
    # In real usage: call your LLM here with build_gambling_harm_prompt(md, user_question)
    explanation_prompt = build_gambling_harm_prompt(
        md,
        f"If I bet {q.stake} on every {q.team} game in {q.season}, how would I do?"
    )
    return {
        "sql_results": df.to_dict(orient="records"),
        "llm_prompt_example": explanation_prompt,
    }

@app.post("/house-edge")
def house_edge(q: HouseEdgeQuery):
    df = run_sql(sql_templates.HOUSE_EDGE_TEMPLATE, {})
    md = df_to_markdown(df)
    prompt = build_gambling_harm_prompt(
        md,
        "Explain the average house edge (vig) on moneylines, spreads, and totals, and why that means long-term losses."
    )
    return {
        "sql_results": df.to_dict(orient="records"),
        "llm_prompt_example": prompt,
    }

def simulate_moneyline(home_team: str, away_team: str, home_ml: int, bet_amount: float):
    # Convert ML to implied probability
    if home_ml < 0:
        p = (-home_ml) / ((-home_ml) + 100)
    else:
        p = 100 / (home_ml + 100)

    # Your model predicted win probability
    model_prob = 0.58  # placeholder

    fair_line = int(-(model_prob / (1 - model_prob)) * 100)
    house_edge = (p - model_prob) * 100

    expected_profit_curve = []
    bankroll = 0
    for i in range(1, 51):
        expected_profit = (model_prob * bet_amount) - ((1 - model_prob) * bet_amount)
        bankroll += expected_profit
        expected_profit_curve.append(bankroll)

    return {
        "win_prob": model_prob,
        "fair_line": fair_line,
        "house_edge": house_edge,
        "bet_numbers": list(range(1, 51)),
        "expected_profit_curve": expected_profit_curve,
    }


from pydantic import BaseModel
import joblib

# ================================
#  NEW: Load ML models
# ================================
MODEL_DIR = "ml/models"

home_win_model = joblib.load(os.path.join(MODEL_DIR, "home_win_xgb.pkl"))
home_win_scaler = joblib.load(os.path.join(MODEL_DIR, "home_win_scaler.pkl"))
home_win_calib = joblib.load(os.path.join(MODEL_DIR, "home_win_logreg_calibrated.pkl"))

spread_model = joblib.load(os.path.join(MODEL_DIR, "spread_xgb.pkl"))
spread_scaler = joblib.load(os.path.join(MODEL_DIR, "spread_scaler.pkl"))
spread_calib = joblib.load(os.path.join(MODEL_DIR, "spread_logreg_calibrated.pkl"))

ou_model = joblib.load(os.path.join(MODEL_DIR, "ou_xgb.pkl"))
ou_scaler = joblib.load(os.path.join(MODEL_DIR, "ou_scaler.pkl"))
ou_calib = joblib.load(os.path.join(MODEL_DIR, "ou_logreg_calibrated.pkl"))




# ================================
#  NEW: Request schema
# ================================
class FeaturePayload(BaseModel):
    features: dict   # keys MUST match your training feature column names


# ================================
#  NEW: Helper
# ================================
def payload_to_df(payload: FeaturePayload) -> pd.DataFrame:
    return pd.DataFrame([payload.features])


# ================================
#  NEW: Prediction routes
# ================================

@app.post("/predict/home-win")
def predict_home_win(payload: FeaturePayload):
    X = payload_to_df(payload)
    prob = float(home_win_model.predict_proba(X)[0, 1])
    pred = int(home_win_model.predict(X)[0])
    return {
        "prediction": pred,
        "prob_home_win": prob
    }


@app.post("/predict/spread")
def predict_spread(payload: FeaturePayload):
    X = payload_to_df(payload)
    prob = float(spread_model.predict_proba(X)[0, 1])
    pred = int(spread_model.predict(X)[0])
    return {
        "prediction": pred,
        "prob_home_covers": prob
    }


@app.post("/predict/ou")
def predict_ou(payload: FeaturePayload):
    X = payload_to_df(payload)
    prob = float(ou_model.predict_proba(X)[0, 1])
    pred = int(ou_model.predict(X)[0])
    return {
        "prediction": pred,
        "prob_over": prob
    }
