DROP TABLE IF EXISTS games;
DROP TABLE IF EXISTS betting_outcomes;
DROP TABLE IF EXISTS strategy_results;

CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    Date TEXT,
    season TEXT,
    home_team TEXT,
    away_team TEXT,
    home_pts INT,
    away_pts INT,
    total_pts INT,
    actual_margin REAL,

    pinnacle_ml_home REAL,
    pinnacle_ml_away REAL,
    avg_ml_home REAL,
    avg_ml_away REAL,

    home_spread REAL,
    home_spread_odds REAL,
    away_spread REAL,
    away_spread_odds REAL,

    pinnacle_total REAL,
    ou_line REAL,
    home_ou_odds REAL,
    away_ou_odds REAL,
    avg_total REAL,

    percent_bet_ml_home REAL,
    percent_bet_ml_away REAL,
    percent_bet_spread_home REAL,
    percent_bet_spread_away REAL,
    percent_bet_ou_home REAL,
    percent_bet_ou_away REAL,

    vig_ml REAL,
    vig_spread REAL,
    vig_ou REAL,

    home_fg_pct REAL,
    home_ft_pct REAL,
    home_fg3_pct REAL,
    home_ast REAL,
    home_reb REAL,
    home_tov REAL,

    away_fg_pct REAL,
    away_ft_pct REAL,
    away_fg3_pct REAL,
    away_ast REAL,
    away_reb REAL,
    away_tov REAL,

    -- Shot features
    home_shots_attempts REAL,
    home_shots_made REAL,
    home_threes_attempts REAL,
    home_threes_made REAL,
    home_avg_shot_distance REAL,
    home_at_rim_attempts REAL,
    home_midrange_attempts REAL,
    home_three_plus_attempts REAL,
    home_shot_fg_pct REAL,
    home_shot_three_pct REAL,
    home_rim_freq REAL,
    home_mid_freq REAL,
    home_three_plus_freq REAL,

    away_shots_attempts REAL,
    away_shots_made REAL,
    away_threes_attempts REAL,
    away_threes_made REAL,
    away_avg_shot_distance REAL,
    away_at_rim_attempts REAL,
    away_midrange_attempts REAL,
    away_three_plus_attempts REAL,
    away_shot_fg_pct REAL,
    away_shot_three_pct REAL,
    away_rim_freq REAL,
    away_mid_freq REAL,
    away_three_plus_freq REAL,

    home_win INT,
    away_win INT,
    home_spread_cover INT,
    away_spread_cover INT,
    ou_over_win INT,
    ou_under_win INT
);

CREATE TABLE betting_outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    season TEXT,
    bet_type TEXT,  -- 'ml', 'spread', 'ou'
    team TEXT,
    line REAL,
    odds REAL,
    bet_amount REAL,
    result TEXT,    -- 'win', 'loss', 'push'
    profit REAL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE TABLE strategy_results (
    strategy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT,
    season TEXT,
    total_bets INT,
    wins INT,
    losses INT,
    pushes INT,
    total_wagered REAL,
    total_return REAL,
    roi REAL,
    max_drawdown REAL,
    sharpe_ratio REAL
);
