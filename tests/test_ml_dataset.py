"""Leakage classification + candidate-rule tests for the value-dataset builder.

Loads scripts/ml/build_value_dataset.py by path (scripts/ is not a package)
and exercises it on synthetic CSV rows — no network, no files written.
"""

import importlib.util
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.probabilities.devig import DevigMethod, devig

# build_value_dataset.py imports pandas at module level; without the `ml`
# extra a bare exec below would abort COLLECTION instead of skipping.
pytest.importorskip("pandas")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "build_value_dataset.py"
_spec = importlib.util.spec_from_file_location("build_value_dataset", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bvd: Any = importlib.util.module_from_spec(_spec)
# dataclasses resolves sys.modules[cls.__module__] at class creation — the
# module must be registered BEFORE exec_module (importlib docs pattern).
sys.modules["build_value_dataset"] = bvd
_spec.loader.exec_module(bvd)


def _row_maxavg(**over: str) -> dict[str, str]:
    """Synthetic 2019/20+-era row (Max*/close columns present)."""
    r = {
        "Div": "E0",
        "Date": "16/08/2024",  # a Friday, BST
        "Time": "20:00",
        "HomeTeam": "Alpha",
        "AwayTeam": "Beta",
        "FTHG": "2",
        "FTAG": "1",
        "FTR": "H",
        "PSH": "2.00",
        "PSD": "3.50",
        "PSA": "4.00",
        "MaxH": "2.20",
        "MaxD": "3.55",
        "MaxA": "4.05",
        "PSCH": "1.90",
        "PSCD": "3.60",
        "PSCA": "4.40",
        "MaxCH": "2.00",
        "MaxCD": "3.70",
        "MaxCA": "4.50",
        "P>2.5": "1.90",
        "P<2.5": "2.00",
        "Max>2.5": "2.30",
        "Max<2.5": "2.02",
        "PC>2.5": "1.85",
        "PC<2.5": "2.05",
        "MaxC>2.5": "1.95",
        "MaxC<2.5": "2.10",
        "B365H": "2.05",
        "BWH": "2.02",
        "IWH": "2.00",
    }
    r.update(over)
    return r


def _row_betbrain(**over: str) -> dict[str, str]:
    """Synthetic 2012/13-2018/19-era row (BbMx* best price, no Max*/MaxC*)."""
    r = {
        "Div": "E0",
        "Date": "17/08/13",  # two-digit year form; no Time column in this era
        "HomeTeam": "Gamma",
        "AwayTeam": "Delta",
        "FTHG": "0",
        "FTAG": "0",
        "FTR": "D",
        "PSH": "2.50",
        "PSD": "3.30",
        "PSA": "3.10",
        "BbMxH": "2.65",
        "BbMxD": "3.40",
        "BbMxA": "3.15",
        "Bb1X2": "45",
        "PSCH": "2.60",
        "PSCD": "3.35",
        "PSCA": "2.95",
    }
    r.update(over)
    return r


def test_schema_classifies_every_close_and_outcome_column_as_label() -> None:
    bvd.assert_no_label_leak()  # the build-breaking gate itself
    feats = set(bvd.FEATURE_COLUMNS)
    labels = set(bvd.LABEL_COLUMNS)
    assert not feats & labels
    assert labels == {
        "won",
        "bet_odds",
        "pinn_close_fair",
        "max_close_fair",
        "clv_pinn",
        "clv_max",
        "profit_units",
    }
    # nothing close-derived hides among the features
    assert not {c for c in feats if "close" in c or "clv" in c or c == "won"}
    # every dataset column is classified exactly once
    assert set(bvd.SCHEMA) == feats | labels | set(bvd.ID_COLUMNS)


def test_candidate_rule_emits_only_selections_beating_fair_by_min_edge() -> None:
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()], min_edge=0.005)
    assert cands, "qualifying selections must be emitted"
    fair_1x2 = devig([2.00, 3.50, 4.00], method=DevigMethod.DIFFERENTIAL_MARGIN)
    for c in cands:
        assert c.edge >= 0.005
        if c.market == "1x2":
            i = ("H", "D", "A").index(c.selection)
            assert c.edge == pytest.approx(fair_1x2[i] - 1.0 / c.best_price)
            assert c.fair_prob == pytest.approx(fair_1x2[i])
    # home is a candidate here (2.20 beats fair), and is the market argmax
    one_x_two = [c for c in cands if c.market == "1x2"]
    assert any(c.selection == "H" and c.is_argmax_edge for c in one_x_two)
    assert sum(c.is_argmax_edge for c in one_x_two) <= 1


def test_no_candidates_when_best_price_equals_sharp_price() -> None:
    # best == Pinnacle -> 1/odds keeps the vig, fair removes it -> all edges < 0
    row = _row_maxavg(
        MaxH="2.00",
        MaxD="3.50",
        MaxA="4.00",
        **{"Max>2.5": "1.90", "Max<2.5": "2.00"},
    )
    assert bvd.candidates_from_rows("E0", "2425", [row], min_edge=0.005) == []


def test_close_columns_affect_only_labels_never_features() -> None:
    base = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    moved = bvd.candidates_from_rows(
        "E0",
        "2425",
        [_row_maxavg(PSCH="2.40", PSCD="3.40", PSCA="3.30", MaxCH="2.50")],
    )
    assert len(base) == len(moved)
    assert len(base) >= 1
    for a, b in zip(base, moved, strict=True):
        for col in bvd.FEATURE_COLUMNS:
            assert getattr(a, col) == getattr(b, col), f"close move leaked into {col}"
        assert a.won == b.won
        assert a.bet_odds == b.bet_odds
    pair = next(
        (a, b) for a, b in zip(base, moved, strict=True) if a.market == "1x2" and a.selection == "H"
    )
    assert pair[0].clv_pinn != pair[1].clv_pinn  # labels DID move


def test_ou25_settlement_and_clv_labels() -> None:
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    over = next(c for c in cands if c.market == "ou25" and c.selection == "over")
    assert over.won is True  # FTHG+FTAG = 3 -> over 2.5 wins
    assert over.profit_units == pytest.approx(over.bet_odds - 1.0)
    close_fair = devig([1.85, 2.05], method=DevigMethod.DIFFERENTIAL_MARGIN)
    assert over.pinn_close_fair == pytest.approx(close_fair[0])
    assert over.clv_pinn == pytest.approx(math.log(over.bet_odds * close_fair[0]))
    # under loses on 3 goals
    under_row = _row_maxavg(**{"Max<2.5": "2.30", "Max>2.5": "1.90"})
    unders = bvd.candidates_from_rows("E0", "2425", [under_row])
    under = next(c for c in unders if c.market == "ou25" and c.selection == "under")
    assert under.won is False
    assert under.profit_units == -1.0


def test_betbrain_era_uses_bbmx_and_has_no_max_close_label() -> None:
    cands = bvd.candidates_from_rows("E0", "1314", [_row_betbrain()])
    assert cands, "BbMx beating Pinnacle fair must qualify"
    for c in cands:
        assert c.market == "1x2"  # no Pinnacle pre-match OU in this era
        assert c.era == "betbrain"
        assert c.max_close_fair is None
        assert c.clv_max is None  # MaxC* starts 2019/20
        assert c.clv_pinn is not None  # PSC* exists from 1213
        assert c.book_count == 45  # explicit Bb1X2 count
        assert c.kickoff_utc is None  # no Time column pre-2019/20
    home = next(c for c in cands if c.selection == "H")
    assert home.best_price == pytest.approx(2.65)  # BbMxH, not PSH/B365H


def test_kickoff_utc_is_tz_aware_and_converted_from_uk_local() -> None:
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    k = cands[0].kickoff_utc
    assert k is not None
    assert k.tzinfo is not None
    # 2024-08-16 is BST (UTC+1): 20:00 UK -> 19:00 UTC
    assert k == datetime(2024, 8, 16, 19, 0, tzinfo=UTC)
    assert cands[0].day_of_week == 4  # Friday


# ---------------------------------------------------------------------------
# v2 — schema versioning, FLB, anchor disagreement, rolling form, xG
# ---------------------------------------------------------------------------
def _stat_row(date: str, home: str, away: str, **over: str) -> dict[str, str]:
    """maxavg row with stats columns (HS/HST/HC) for rolling-form tests."""
    r = _row_maxavg(Date=date, HomeTeam=home, AwayTeam=away)
    r.update(over)
    return r


def test_v1_schema_is_exact_frozen_subset() -> None:
    assert len(bvd.SCHEMA_V1) == 29
    assert set(bvd.SCHEMA_V1) <= set(bvd.SCHEMA)
    # frozen v1 ordering — the --v1 artifact must keep the original layout
    assert bvd.SCHEMA_V1[:8] == (
        "league",
        "season",
        "match_date",
        "kickoff_utc",
        "home_team",
        "away_team",
        "market",
        "selection",
    )
    assert bvd.SCHEMA_V1[-7:] == (
        "won",
        "bet_odds",
        "pinn_close_fair",
        "max_close_fair",
        "clv_pinn",
        "clv_max",
        "profit_units",
    )
    # v1 columns keep their original ID/SIGNAL/LABEL classification
    for col in bvd.SCHEMA_V1:
        assert col in bvd.SCHEMA


def test_to_dataframe_v1_flag_reproduces_v1_columns() -> None:
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    df_v1 = bvd._to_dataframe(cands, schema_version=1)
    assert list(df_v1.columns) == list(bvd.SCHEMA_V1)
    df_v2 = bvd._to_dataframe(cands)
    assert list(df_v2.columns) == list(bvd.SCHEMA)
    assert df_v2["consulted_before"].dtype == bool
    assert str(df_v2["fair_prob_rank"].dtype) == "int32"
    # v2 nullable feature columns land as float64 (LightGBM-native NaN)
    assert str(df_v2["home_xg_for_r5"].dtype) == "float64"
    assert str(df_v2["away_corners_against_r5"].dtype) == "float64"


def test_rolling_form_uses_only_strictly_prior_matches() -> None:
    rows = [
        _stat_row(
            "16/08/2024",
            "Alpha",
            "Beta",
            FTHG="2",
            FTAG="1",
            HS="10",
            AS="4",
            HST="5",
            AST="2",
            HC="6",
            AC="3",
        ),
        _stat_row(
            "23/08/2024",
            "Beta",
            "Alpha",
            FTHG="0",
            FTAG="3",
            HS="7",
            AS="12",
            HST="3",
            AST="6",
            HC="2",
            AC="8",
        ),
        _stat_row(
            "30/08/2024",
            "Alpha",
            "Beta",
            FTHG="5",
            FTAG="5",
            FTR="D",
            HS="99",
            AS="99",
            HST="99",
            AST="99",
            HC="99",
            AC="99",
        ),
    ]
    cands = bvd.candidates_from_rows("E0", "2425", rows)
    first = next(c for c in cands if c.match_date.day == 16)
    # first match of the season: no prior matches -> all form features null
    for col in (f"home_{f}" for f in bvd._FORM_FEATS):
        assert getattr(first, col) is None, f"{col} must be null on matchday 1"
    third = next(c for c in cands if c.match_date.day == 30)
    # Alpha (home): m1 home (gf 2, ga 1, sf 10, sa 4), m2 away (gf 3, ga 0, sf 12, sa 7)
    assert third.home_gf_r5 == pytest.approx(2.5)
    assert third.home_ga_r5 == pytest.approx(0.5)
    assert third.home_gf_r10 == pytest.approx(2.5)
    assert third.home_shots_for_r5 == pytest.approx(11.0)
    assert third.home_shots_against_r5 == pytest.approx(5.5)
    assert third.home_sot_for_r5 == pytest.approx(5.5)
    assert third.home_sot_against_r5 == pytest.approx(2.5)
    assert third.home_corners_for_r5 == pytest.approx(7.0)
    assert third.home_corners_against_r5 == pytest.approx(2.5)
    # Beta (away): mirror image
    assert third.away_gf_r5 == pytest.approx(0.5)
    assert third.away_ga_r5 == pytest.approx(2.5)
    assert third.away_shots_for_r5 == pytest.approx(5.5)
    assert third.away_shots_against_r5 == pytest.approx(11.0)
    # the current match's own 99s did NOT enter its features (shift(1) proof):
    # had they leaked, home_shots_for_r5 would be (10+12+99)/3 = 40.33
    assert third.home_shots_for_r5 < 12.0


def test_rolling_form_stats_nullable_but_goals_present() -> None:
    rows = [
        _stat_row("16/08/2024", "Alpha", "Beta", FTHG="2", FTAG="1"),  # no HS/HC cols
        _stat_row("23/08/2024", "Alpha", "Beta", FTHG="0", FTAG="0", FTR="D"),
    ]
    cands = bvd.candidates_from_rows("E0", "2425", rows)
    second = next(c for c in cands if c.match_date.day == 23)
    assert second.home_gf_r5 == pytest.approx(2.0)  # goals always present
    assert second.away_ga_r5 == pytest.approx(2.0)
    assert second.home_shots_for_r5 is None  # stats columns absent -> null
    assert second.away_corners_against_r5 is None


def test_flb_odds_band_and_fair_prob_rank() -> None:
    assert bvd._odds_band(1.5) == "(1.0,1.5]"
    assert bvd._odds_band(1.51) == "(1.5,2.0]"
    assert bvd._odds_band(2.2) == "(2.0,3.0]"
    assert bvd._odds_band(10.0) == "(5.0,10.0]"
    assert bvd._odds_band(10.01) == "(10.0,inf)"
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    fair = devig([2.00, 3.50, 4.00], method=DevigMethod.DIFFERENTIAL_MARGIN)
    for c in cands:
        assert c.odds_band == bvd._odds_band(c.best_price)
        if c.market == "1x2":
            i = ("H", "D", "A").index(c.selection)
            assert c.fair_prob_rank == 1 + sum(1 for p in fair if p > fair[i])
    home = next(c for c in cands if c.market == "1x2" and c.selection == "H")
    assert home.fair_prob_rank == 1  # shortest price = highest fair prob
    assert home.odds_band == "(2.0,3.0]"  # best price 2.20


def test_anchor_disagreement_v2_features() -> None:
    cands = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    home = next(c for c in cands if c.market == "1x2" and c.selection == "H")
    assert home.price_ratio_best_pinn == pytest.approx(2.20 / 2.00)
    assert home.prob_gap_pinn_best == pytest.approx(1 / 2.00 - 1 / 2.20)
    ps = [2.00, 3.50, 4.00]
    canon = devig(ps, method=DevigMethod.DIFFERENTIAL_MARGIN)
    assert home.fair_mult_delta == pytest.approx(
        devig(ps, method=DevigMethod.MULTIPLICATIVE)[0] - canon[0]
    )
    assert home.fair_shin_delta == pytest.approx(devig(ps, method=DevigMethod.SHIN)[0] - canon[0])
    assert home.fair_power_delta == pytest.approx(devig(ps, method=DevigMethod.POWER)[0] - canon[0])


def test_consulted_before_tags_the_18_league_universe() -> None:
    assert len(bvd.CONSULTED_LEAGUES) == 18
    for code in ("EC", "SC1", "SC2", "SC3"):
        assert code not in bvd.CONSULTED_LEAGUES
    consulted = bvd.candidates_from_rows("E0", "2425", [_row_maxavg()])
    assert all(c.consulted_before for c in consulted)
    fresh = bvd.candidates_from_rows("SC2", "2425", [_row_maxavg()])
    assert fresh and all(not c.consulted_before for c in fresh)


def _us_fixture(
    dt: str, home: str, away: str, gh: int, ga: int, xgh: float, xga: float
) -> dict[str, str]:
    return {
        "datetime": dt,
        "team_home": home,
        "team_away": away,
        "goals_home": str(gh),
        "goals_away": str(ga),
        "xg_home": str(xgh),
        "xg_away": str(xga),
    }


def test_xg_lookup_rolls_strictly_prior_and_maps_names() -> None:
    fixtures = [
        _us_fixture("2024-08-10 12:00:00", "Manchester City", "Alpha", 2, 1, 1.8, 0.7),
        _us_fixture("2024-08-17 12:00:00", "Alpha", "Manchester City", 0, 1, 0.5, 2.2),
        _us_fixture("2024-08-24 12:00:00", "Manchester City", "Beta", 3, 0, 2.5, 0.4),
    ]
    lookup = bvd.build_xg_lookup(fixtures)
    # join keys are MAPPED football-data names (Manchester City -> Man City)
    h1, a1 = lookup[("Man City", "Alpha")]
    assert all(v is None for v in h1.values())  # first match: no prior xG
    assert all(v is None for v in a1.values())
    h3, _ = lookup[("Man City", "Beta")]
    assert h3["xg_for_r5"] == pytest.approx((1.8 + 2.2) / 2)
    assert h3["xg_against_r5"] == pytest.approx((0.7 + 0.5) / 2)
    assert h3["xg_for_r10"] == pytest.approx((1.8 + 2.2) / 2)
    # overperformance = goals_for - xg_for over priors: (2-1.8), (1-2.2)
    assert h3["xg_overperf_r10"] == pytest.approx((0.2 - 1.2) / 2)
    # the current fixture's own 2.5 xG did not leak into its features
    assert h3["xg_for_r5"] < 2.2


def test_xg_features_join_candidates_and_default_null() -> None:
    fixtures = [
        _us_fixture("2024-08-10 12:00:00", "Manchester City", "Beta", 2, 1, 1.8, 0.7),
        _us_fixture("2024-08-17 12:00:00", "Manchester City", "Alpha", 1, 1, 1.4, 1.1),
    ]
    lookup = bvd.build_xg_lookup(fixtures)
    row = _row_maxavg(HomeTeam="Man City", AwayTeam="Alpha", Date="17/08/2024")
    cands = bvd.candidates_from_rows("E0", "2425", [row], 0.005, lookup)
    assert cands
    for c in cands:
        assert c.home_xg_for_r5 == pytest.approx(1.8)
        assert c.home_xg_against_r5 == pytest.approx(0.7)
        assert c.home_xg_overperf_r10 == pytest.approx(0.2)
        assert c.away_xg_for_r5 is None  # Alpha's first appearance
    # no lookup (non-big-5 league / offline) -> all xG features null
    plain = bvd.candidates_from_rows("E1", "2425", [_row_maxavg()])
    for c in plain:
        for col in (f"{s}_{f}" for s in ("home", "away") for f in bvd._XG_FEATS):
            assert getattr(c, col) is None


def test_unmatched_understat_names_quarantine_with_log(capsys: pytest.CaptureFixture) -> None:
    fixtures = [
        _us_fixture("2024-08-10 12:00:00", "Phantom FC", "Manchester City", 1, 1, 1.0, 1.0),
        _us_fixture("2024-08-17 12:00:00", "Manchester City", "Phantom FC", 1, 1, 1.0, 1.0),
    ]
    lookup = bvd.build_xg_lookup(fixtures)
    rows = [_row_maxavg(HomeTeam="Man City", AwayTeam="Beta")]
    bvd._quarantine_unmatched_names("E0", "2425", rows, lookup)
    captured = capsys.readouterr().out
    assert "quarantine understat names" in captured
    assert "Phantom FC" in captured  # explicit log, no fuzzy auto-merge
