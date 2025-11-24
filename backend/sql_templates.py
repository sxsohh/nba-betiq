# Common SQL templates for the LLM to use (filled by code, not by user directly)

TEAM_PROFIT_TEMPLATE = """
SELECT
    team,
    season,
    bet_type,
    COUNT(*) AS total_bets,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
    SUM(profit) AS total_profit,
    AVG(profit) AS avg_profit_per_bet
FROM betting_outcomes
WHERE team = :team
  AND season = :season
GROUP BY team, season, bet_type
ORDER BY bet_type;
"""

PUBLIC_TRAP_TEMPLATE = """
WITH public_games AS (
    SELECT
        g.game_id,
        g.home_team,
        g.away_team,
        g.percent_bet_ml_home,
        g.actual_margin
    FROM games g
    WHERE g.percent_bet_ml_home IS NOT NULL
)
SELECT
    CASE
        WHEN percent_bet_ml_home > 70 THEN 'heavy_home'
        WHEN percent_bet_ml_home < 30 THEN 'heavy_away'
        ELSE 'balanced'
    END AS public_sentiment,
    COUNT(*) AS games,
    AVG(CASE WHEN actual_margin > 0 THEN 1 ELSE 0 END) AS home_win_rate,
    AVG(actual_margin) AS avg_margin
FROM public_games
GROUP BY public_sentiment;
"""

HOUSE_EDGE_TEMPLATE = """
SELECT
    AVG(vig_ml) AS avg_vig_ml,
    AVG(vig_spread) AS avg_vig_spread,
    AVG(vig_ou) AS avg_vig_ou
FROM games;
"""

BEST_ATS_TEMPLATE = """
WITH team_performance AS (
    SELECT
        home_team AS team,
        season,
        COUNT(*) AS games,
        AVG(home_spread_cover) AS cover_rate
    FROM games
    GROUP BY home_team, season
)
SELECT
    team,
    season,
    games,
    cover_rate,
    cover_rate - 0.5 AS edge_over_coin_flip,
    (cover_rate * 0.91) - (1 - cover_rate) AS edge_after_vig
FROM team_performance
WHERE games >= 20
ORDER BY edge_after_vig DESC;
"""
