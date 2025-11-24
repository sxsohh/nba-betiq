import os
import sqlite3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from . import sql_templates
from .llm_query import build_gambling_harm_prompt

DB_PATH = os.path.join("db", "nba_betting.db")

app = FastAPI(title="NBA BetIQ API")

class TeamSeasonQuery(BaseModel):
    team: str
    season: str
    stake: float = 100.0

class HouseEdgeQuery(BaseModel):
    pass

def run_sql(query: str, params: dict | None = None) -> pd.DataFrame:
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
