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
