NBA BetIQ

A transparency project on how sportsbooks maintain their advantage

NBA BetIQ is an open-source analytics tool designed to help people understand why sports bettors lose money in the long run. The goal is not to encourage betting. It is to show, through data, probability, and machine learning, why the house consistently wins.

This project combines multi-season NBA game data with real sportsbook odds to model true win probabilities and compare them to sportsbook implied odds. The result is a clear view into vig, mispricing, and ROI decay for everyday bettors.

Core Features

Full ETL pipeline merging NBA box scores, play-by-play data, odds movement, and public betting percentages

Moneyline, spread, and total machine learning models with probability calibration

FastAPI backend with endpoints for simulation and probability estimates

Interactive frontend for exploring ROI, house edge, and long-term loss curves

Designed to educate athletes, students, and fans about the realities of sports betting

Why This Project Exists

Sports gambling has become a national crisis for young people.
Millions of students and athletes are exposed to constant betting ads and “guaranteed picks,” even though the math shows that long-term profit is extremely unlikely.

NBA BetIQ is built to:

reveal how vig eats into bettor ROI

demonstrate that “50-50 bets” are not actually even

teach probability and statistical reasoning

help people understand the risks before money is on the line

This is a data transparency effort, not a betting tool.

How It Works

Raw NBA data and betting odds are cleaned and merged into a master dataset

Machine learning models estimate true probabilities for home win, spread cover, and total results

The system compares fair probabilities to sportsbook-implied probabilities

Users can run simulations to see how bankrolls change over time

Running Locally
1. Install requirements
pip install -r requirements.txt

2. Train models

Run these once to generate model files:

python ml/train_moneyline_model.py
python ml/train_spread_model.py
python ml/train_ou_model.py

3. Start the FastAPI backend
uvicorn backend.main:app --reload

4. Visit the API

http://127.0.0.1:8000

5. Deploying the Backend

The backend can be deployed on Render, Railway, or Fly.io.
Once deployed, set the environment variable in Vercel:

BETIQ_API = https://nba-betiq.onrender.com

Frontend Integration

The Next.js frontend calls:

/api/betiq/team-profit
/api/betiq/house-edge
/api/betiq/simulate


These provide:

probability estimates

expected value calculations

bankroll simulation results

Folder Structure
backend/
  main.py
  llm_query.py
  sql_templates.py

ml/
  train_moneyline_model.py
  train_spread_model.py
  train_ou_model.py
  models/

simulations/
  backtest_strategies.py

notebooks/
  analysis_template.py

db/
  database.db

data/
  raw/
  clean/

License

MIT License. This project is for educational and research purposes only.
