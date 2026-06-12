"""ValuePolicy pure helpers: the empty policy is a strict no-op; overrides
match the line-qualified market key first, then the market family."""

from app.edge.value_policy import (
    ValuePolicy,
    distinct_book_count,
    market_lookup_keys,
    min_books_for,
    min_edge_for,
    odds_in_bands,
)


def test_empty_policy_is_a_strict_noop() -> None:
    policy = ValuePolicy()
    assert min_edge_for(policy, "h2h", "1x2", default=0.03) == 0.03
    assert min_books_for(policy, "h2h", "1x2") == 0
    assert odds_in_bands(1.01, policy.odds_bands) is True
    assert odds_in_bands(1000.0, policy.odds_bands) is True


def test_detail_key_beats_family_key() -> None:
    policy = ValuePolicy(min_edge_by_market=(("totals", 0.04), ("over_under_2_5", 0.06)))
    assert min_edge_for(policy, "totals", "over_under_2_5", default=0.03) == 0.06
    # a different line of the same family falls back to the family entry
    assert min_edge_for(policy, "totals", "over_under_3_5", default=0.03) == 0.04


def test_family_key_applies_when_detail_is_absent() -> None:
    policy = ValuePolicy(min_edge_by_market=(("h2h", 0.05),))
    assert min_edge_for(policy, "h2h", None, default=0.03) == 0.05
    # unlisted market keeps the global default
    assert min_edge_for(policy, "btts", None, default=0.03) == 0.03


def test_lookup_keys_are_normalized_and_ordered() -> None:
    assert market_lookup_keys("H2H", " 1X2 ") == ("1x2", "h2h")
    assert market_lookup_keys("totals", None) == ("totals",)
    assert market_lookup_keys("totals", "  ") == ("totals",)
    # detail equal to family is deduplicated
    assert market_lookup_keys("h2h", "h2h") == ("h2h",)


def test_min_books_lookup_mirrors_min_edge_lookup() -> None:
    policy = ValuePolicy(min_books_by_market=(("over_under_1_5", 5), ("totals", 3)))
    assert min_books_for(policy, "totals", "over_under_1_5") == 5
    assert min_books_for(policy, "totals", "over_under_3_5") == 3
    assert min_books_for(policy, "h2h", "1x2") == 0


def test_odds_bands_are_inclusive_and_unioned() -> None:
    bands = ((1.8, 2.6), (3.0, 4.2))
    assert odds_in_bands(1.8, bands) is True  # inclusive lo
    assert odds_in_bands(2.6, bands) is True  # inclusive hi
    assert odds_in_bands(2.9, bands) is False  # between bands
    assert odds_in_bands(3.5, bands) is True  # second band
    assert odds_in_bands(4.3, bands) is False  # above all bands
    assert odds_in_bands(1.6, bands) is False  # below all bands


def test_distinct_book_count_normalizes_names_across_selections() -> None:
    prices = {
        "Home": {"Pinnacle": 2.5, "SoftBook": 2.9},
        "Draw": {"pinnacle ": 3.3},  # same book, different casing/spacing
        "Away": {"OtherBook": 3.1},
    }
    assert distinct_book_count(prices) == 3
