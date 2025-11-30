import os
import sqlite3
import pytest

DB_PATH = os.path.join("db", "nba_betting.db")

def test_db_has_tables():
    if not os.path.exists(DB_PATH):
        pytest.skip("Database file not found (expected in CI/CD)")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "games" in tables
    assert "betting_outcomes" in tables
