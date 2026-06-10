"""Sharp-vs-soft value finder: sharp selection, best-price edge, fallbacks."""

import pytest

from app.edge.value import find_value_bets


def test_flags_value_when_soft_beats_pinnacle_fair() -> None:
    # Pinnacle ~ fair (low margin); SoftBook offers a generous home price.
    prices = {
        "home": {"Pinnacle": 2.00, "SoftBook": 2.30, "OtherBook": 2.05},
        "away": {"Pinnacle": 2.00, "SoftBook": 1.80, "OtherBook": 1.95},
    }
    bets = find_value_bets(prices, min_edge=0.01)
    assert len(bets) == 1
    v = bets[0]
    assert v.selection == "home"
    assert v.best_book == "SoftBook"
    assert v.best_odds == 2.30
    assert v.sharp_book == "Pinnacle"
    # Pinnacle 2.00/2.00 -> fair 0.5; soft 2.30 -> implied 0.4348; edge ~0.065
    assert v.sharp_fair_prob == pytest.approx(0.5, abs=1e-9)
    assert v.edge == pytest.approx(0.5 - 1 / 2.30, abs=1e-9)
    assert v.ev > 0


def test_no_value_when_soft_prices_are_tight() -> None:
    prices = {
        "home": {"Pinnacle": 2.00, "SoftBook": 1.95},
        "away": {"Pinnacle": 2.00, "SoftBook": 1.95},
    }
    assert find_value_bets(prices, min_edge=0.01) == []


def test_sharp_book_excluded_from_best_price() -> None:
    # Even though Pinnacle has the highest home odds, it can't be the value book.
    prices = {
        "home": {"Pinnacle": 2.40, "SoftBook": 2.10},
        "away": {"Pinnacle": 1.70, "SoftBook": 1.80},
    }
    bets = find_value_bets(prices, min_edge=0.01)
    # sharp fair from Pinnacle (2.40/1.70); best OTHER book for home is SoftBook 2.10
    assert all(b.best_book != "Pinnacle" for b in bets)


def test_fallback_to_lowest_overround_when_no_named_sharp() -> None:
    # No Pinnacle; TightBook has the lowest overround -> used as fair source.
    prices = {
        "home": {"TightBook": 2.02, "WideBook": 2.40},
        "away": {"TightBook": 1.98, "WideBook": 1.70},
    }
    bets = find_value_bets(prices, min_edge=0.01)
    assert len(bets) == 1
    assert bets[0].sharp_book == "TightBook"
    assert bets[0].best_book == "WideBook"
    assert bets[0].selection == "home"


def test_requires_full_market_pricing() -> None:
    # A book pricing only one side cannot anchor fair value.
    prices = {"home": {"OnlyHome": 2.0}, "away": {"OtherBook": 2.0}}
    assert find_value_bets(prices, min_edge=0.0) == []


def test_single_selection_returns_nothing() -> None:
    assert find_value_bets({"home": {"Pinnacle": 2.0}}, min_edge=0.0) == []
