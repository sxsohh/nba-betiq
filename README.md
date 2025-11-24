# NBA BetIQ – Why the House Always Wins

This project analyzes NBA betting data from the **2018–2019** season to show how
the sportsbook's **house edge (vig)** makes it extremely difficult to profit long term,
even with smart strategies and decent models.

## Data

- `data/raw/vegas.txt` – regular season odds + public betting
- `data/raw/vegas_playoff.txt` – playoff odds
- `data/raw/raw_scores.txt` – box score / final scores
- `data/raw/NBA_2019_Shots.csv` – shot-level data for 2018–19

All ETL steps produce:

- `data/clean/games_master_2018_19.csv`
- `data/clean/betting_outcomes_2018_19.csv`
- `db/nba_betting.db` (SQLite)

## Pipeline

1. **ETL**

```bash
python etl/01_clean_odds.py
python etl/02_clean_scores.py
python etl/03_clean_shots.py
python etl/04_feature_engineering.py
