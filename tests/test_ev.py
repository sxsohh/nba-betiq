"""
Test suite for Expected Value calculations.
"""
import pytest
from backend.utils import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_ev,
    kelly_criterion,
    calculate_roi
)


def test_american_to_decimal():
    """Test American odds conversion to decimal."""
    assert american_to_decimal(-110) == pytest.approx(1.909, abs=0.01)
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-200) == 1.5


def test_american_to_implied_prob():
    """Test implied probability from American odds."""
    assert american_to_implied_prob(-110) == pytest.approx(0.5238, abs=0.001)
    assert american_to_implied_prob(150) == pytest.approx(0.4, abs=0.001)
    assert american_to_implied_prob(100) == 0.5
    assert american_to_implied_prob(-200) == pytest.approx(0.6667, abs=0.001)


def test_calculate_ev_positive():
    """Test EV calculation with positive expected value."""
    result = calculate_ev(win_prob=0.58, odds=-110, stake=100)

    assert result["ev"] > 0
    assert result["ev_percent"] > 0
    assert result["recommendation"] == "BET"
    assert "implied_prob" in result
    assert "edge" in result


def test_calculate_ev_negative():
    """Test EV calculation with negative expected value."""
    result = calculate_ev(win_prob=0.45, odds=-110, stake=100)

    assert result["ev"] < 0
    assert result["ev_percent"] < 0
    assert result["recommendation"] == "PASS"


def test_calculate_ev_zero():
    """Test EV calculation at break-even."""
    # -110 odds implies ~52.38% probability
    result = calculate_ev(win_prob=0.5238, odds=-110, stake=100)

    assert result["ev"] == pytest.approx(0, abs=0.5)
    assert result["edge"] == pytest.approx(0, abs=0.01)


def test_kelly_criterion():
    """Test Kelly Criterion bet sizing."""
    # With edge, should recommend betting
    kelly = kelly_criterion(win_prob=0.58, odds=-110, kelly_fraction=0.25)
    assert kelly > 0

    # Without edge, should recommend 0
    kelly_no_edge = kelly_criterion(win_prob=0.45, odds=-110, kelly_fraction=0.25)
    assert kelly_no_edge == 0


def test_calculate_roi():
    """Test ROI calculation."""
    roi = calculate_roi(wins=55, losses=45, avg_odds=-110, stake=100)

    assert roi["total_bets"] == 100
    assert roi["win_rate"] == 0.55
    assert roi["total_staked"] == 10000
    assert roi["profit"] != 0
    assert "roi_percent" in roi


def test_ev_with_underdog_odds():
    """Test EV calculation with plus odds (underdog)."""
    result = calculate_ev(win_prob=0.5, odds=200, stake=100)

    # 50% prob at +200 should have positive EV
    assert result["ev"] > 0
    assert result["implied_prob"] == pytest.approx(0.3333, abs=0.001)


def test_ev_with_favorite_odds():
    """Test EV calculation with minus odds (favorite)."""
    result = calculate_ev(win_prob=0.7, odds=-300, stake=100)

    # 70% prob at -300 (implies 75%) should have negative EV (no edge)
    assert result["ev"] < 0
    assert result["implied_prob"] == pytest.approx(0.75, abs=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
