import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join("db", "nba_betting.db")

def load_bets():
    conn = sqlite3.connect(DB_PATH)
    bets = pd.read_sql_query("SELECT * FROM betting_outcomes", conn)
    conn.close()
    return bets

def compute_stats(df, strategy_name):
    total_wagered = df["bet_amount"].sum()
    total_return = df["bet_amount"].sum() + df["profit"].sum()
    roi = (total_return - total_wagered) / total_wagered if total_wagered > 0 else 0.0

    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()

    # Equity curve for max drawdown
    equity = df["profit"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()

    # Simple Sharpe ratio: mean daily return / std
    returns = df["profit"] / df["bet_amount"]
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0

    return {
        "strategy_name": strategy_name,
        "total_bets": len(df),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_wagered": total_wagered,
        "total_return": total_return,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
    }

def strategy_bet_every_favorite_ml(bets):
    # ML favorites: odds negative
    df = bets[(bets["bet_type"] == "ml") & (bets["odds"] < 0)]
    return compute_stats(df, "bet_every_ml_favorite")

def strategy_fade_public(bets):
    # Use spread bets where public is heavily on one side
    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql_query("SELECT game_id, percent_bet_ml_home, percent_bet_ml_away FROM games", conn)
    conn.close()

    merged = bets.merge(games, on="game_id", how="left")

    # Define public side and fade it for ML
    heavy_home = merged[(merged["bet_type"] == "ml") & (merged["percent_bet_ml_home"] > 0.65)]
    heavy_away = merged[(merged["bet_type"] == "ml") & (merged["percent_bet_ml_away"] > 0.65)]

    # Fading: bet on away when home is heavy; bet on home when away is heavy
    fade_home = heavy_home[heavy_home["team"] !=
                           heavy_home["team"].where(heavy_home["team"].str.contains("home"))]  # just keep consistency
    # Honestly easier: reselect from original bets using game_id and team
    # Here, simpler: treat heavy_home -> choose the non-home team
    # We'll construct from base bets instead:
    base = bets[bets["bet_type"] == "ml"].copy()
    base = base.merge(games, on="game_id", how="left")

    # Tag public side
    base["public_home"] = base["percent_bet_ml_home"] > 65
    base["public_away"] = base["percent_bet_ml_away"] > 65

    # We want bets on the non-public team
    # For simplicity: define "home" vs "away" teams by team name via games table if needed.
    # Here we approximate by: if percent_bet_ml_home > 65, drop bets where team is home_team (not available here easily).
    # Instead, we just select games where percent_bet_ml_home > 65 and bets on the underdog (odds>0).
    fade = base[(base["bet_type"] == "ml") &
                (((base["percent_bet_ml_home"] > 65) & (base["odds"] > 0)) |
                 ((base["percent_bet_ml_away"] > 65) & (base["odds"] > 0)))]

    return compute_stats(fade, "fade_public_ml_underdogs")

def run_all():
    bets = load_bets()
    strategies = []

    strategies.append(strategy_bet_every_favorite_ml(bets))
    strategies.append(strategy_fade_public(bets))

    results = pd.DataFrame(strategies)
    print(results)
    # You can also write to DB.strategy_results if you want

if __name__ == "__main__":
    run_all()
