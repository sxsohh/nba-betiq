import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Example: visualize home shot chart feature distributions
conn = sqlite3.connect(os.path.join("db", "nba_betting.db"))
games = pd.read_sql_query("SELECT * FROM games", conn)
conn.close()

# Histogram of home_avg_shot_distance
plt.figure()
plt.hist(games["home_avg_shot_distance"].dropna(), bins=30)
plt.xlabel("Home average shot distance (ft)")
plt.ylabel("Games")
plt.title("Distribution of home shot distance, 2018-19")
plt.show()

# Rim vs mid vs three freq scatter
plt.figure()
plt.scatter(games["home_rim_freq"], games["home_three_plus_freq"])
plt.xlabel("Rim freq (share of attempts at rim)")
plt.ylabel("3+ freq (share of 3s)")
plt.title("Team shot profile (home games), 2018-19")
plt.show()
