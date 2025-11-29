import pandas as pd

games = pd.read_csv("data/clean/games_master_2018_19.csv")
outcomes = pd.read_csv("data/clean/betting_outcomes_2018_19.csv")

df = games.merge(outcomes, on="GAME_ID", how="inner")
