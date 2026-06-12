"""Value-filter meta-model: scope gates, feature parity, calibrator math.

Everything here runs on CORE deps only (numpy/stdlib) — the lazy-import
design of app/models/value_filter.py is part of what is under test: loading
and scope logic must work (and fail soft) without the `ml` extra installed.
Scoring with a real LightGBM booster lives in tests/test_value_filter_ml.py
(importorskip-guarded).
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.edge.value import CONSENSUS_ANCHOR
from app.models.value_filter import (
    ValueFilterModel,
    calibrate,
    league_code,
    live_features,
)
from app.schemas.base import Market

NOW = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)

PRICES_1X2 = {
    "Home FC": {"Pinnacle": 2.50, "SoftBook": 2.90, "OtherBook": 2.60},
    "Draw": {"Pinnacle": 3.30, "SoftBook": 3.20, "OtherBook": 3.25},
    "Away FC": {"Pinnacle": 3.10, "SoftBook": 2.95, "OtherBook": 3.00},
}
FAIR_1X2 = {"Home FC": 0.40, "Draw": 0.295, "Away FC": 0.305}


def feats_1x2(**over: object) -> dict | None:
    kwargs: dict = {
        "market": Market.H2H,
        "market_detail": None,
        "selection": "Home FC",
        "prices": PRICES_1X2,
        "fair_by_sel": FAIR_1X2,
        "anchor_book": "Pinnacle",
        "league": "england-premier-league",
        "kickoff_utc": None,
        "now": NOW,
        "min_odds": 1.6,
    }
    kwargs.update(over)
    return live_features(**kwargs)


# ---------------------------------------------------------------------------
# League normalization (trained vocabulary only — never guessed)
# ---------------------------------------------------------------------------
def test_league_code_maps_slugs_and_display_names() -> None:
    assert league_code("england-premier-league") == "E0"
    assert league_code("Premier League") == "E0"
    assert league_code("germany-bundesliga") == "D1"
    assert league_code("Serie A") == "I1"
    assert league_code("LaLiga2") == "SP2"


def test_league_code_unknown_is_none() -> None:
    # Out-of-vocabulary leagues are unscored, never approximated: the model
    # never saw them and its evidence does not transfer.
    assert league_code("brazil-serie-a") is None
    assert league_code("nba") is None
    assert league_code("") is None


# ---------------------------------------------------------------------------
# Scope gates — out-of-scope candidates must be UNSCORED (None), not vetoed
# ---------------------------------------------------------------------------
def test_consensus_anchor_is_out_of_scope() -> None:
    # Trained fair side is devig(Pinnacle pre-match); a median consensus
    # anchor is a different estimator — unscored.
    assert feats_1x2(anchor_book=CONSENSUS_ANCHOR) is None


def test_unknown_league_is_out_of_scope() -> None:
    assert feats_1x2(league="brazil-serie-a") is None


def test_below_training_odds_floor_is_out_of_scope() -> None:
    assert feats_1x2(min_odds=3.0) is None  # Home best price 2.90 < 3.0


def test_two_way_h2h_is_out_of_scope() -> None:
    # Basketball moneyline: 2-way h2h was never in the training pool.
    prices = {k: v for k, v in PRICES_1X2.items() if k != "Draw"}
    fair = {"Home FC": 0.55, "Away FC": 0.45}
    assert feats_1x2(prices=prices, fair_by_sel=fair) is None


def test_only_the_2_5_totals_line_is_in_scope() -> None:
    prices = {
        "Over 2.5": {"Pinnacle": 1.95, "SoftBook": 2.05},
        "Under 2.5": {"Pinnacle": 1.95, "SoftBook": 1.90},
    }
    fair = {"Over 2.5": 0.51, "Under 2.5": 0.49}
    ok = feats_1x2(
        market=Market.TOTALS,
        market_detail="over_under_2_5",
        selection="Over 2.5",
        prices=prices,
        fair_by_sel=fair,
    )
    assert ok is not None
    assert ok["market"] == "ou25"
    assert (
        feats_1x2(
            market=Market.TOTALS,
            market_detail="over_under_3_5",  # untrained line
            selection="Over 3.5",
            prices=prices,
            fair_by_sel=fair,
        )
        is None
    )


def test_other_markets_are_out_of_scope() -> None:
    for market in (Market.BTTS, Market.DNB, Market.SPREADS, Market.DOUBLE_CHANCE):
        assert feats_1x2(market=market) is None


def test_missing_selection_or_anchor_quote_is_unscored() -> None:
    assert feats_1x2(selection="Phantom FC") is None
    prices = {k: dict(v) for k, v in PRICES_1X2.items()}
    del prices["Draw"]["Pinnacle"]  # anchor no longer quotes the full market
    assert feats_1x2(prices=prices) is None


# ---------------------------------------------------------------------------
# Feature parity with the dataset builder's definitions
# ---------------------------------------------------------------------------
def test_feature_values_match_hand_computation() -> None:
    f = feats_1x2()
    assert f is not None
    # edge = fair - 1/best (RAW price, dataset definition)
    assert f["edge"] == pytest.approx(0.40 - 1 / 2.90)
    assert f["fair_prob"] == pytest.approx(0.40)
    assert f["best_price"] == pytest.approx(2.90)  # max across ALL books
    assert f["pinn_price"] == pytest.approx(2.50)  # the anchor's own quote
    assert f["overround_pinn"] == pytest.approx(1 / 2.5 + 1 / 3.3 + 1 / 3.1 - 1)
    assert f["overround_best"] == pytest.approx(1 / 2.9 + 1 / 3.3 + 1 / 3.1 - 1)
    assert f["book_count"] == 3  # books quoting the FULL market
    assert f["devig_spread"] >= 0.0
    assert f["is_argmax_edge"] is True  # H edge dominates D (-0.008) and A (-0.018)
    assert f["league"] == "E0"
    assert f["market"] == "1x2"
    assert f["selection_type"] == "fav"  # 0.40 >= 0.305


def test_selection_type_draw_and_dog() -> None:
    draw = feats_1x2(selection="Draw")
    assert draw is not None and draw["selection_type"] == "draw"
    dog = feats_1x2(selection="Away FC")
    assert dog is not None and dog["selection_type"] == "dog"  # 0.305 < 0.40
    assert dog["is_argmax_edge"] is False


def test_day_and_season_end_use_kickoff_date() -> None:
    # June 30 convention (build_value_dataset._season_end): Jul-Dec roll to
    # NEXT year's June 30; Jan-Jun use the same year.
    aug = feats_1x2(kickoff_utc=datetime(2026, 8, 15, 14, 0, tzinfo=UTC))
    assert aug is not None
    assert aug["day_of_week"] == 5  # 2026-08-15 is a Saturday
    assert aug["days_to_season_end"] == (datetime(2027, 6, 30) - datetime(2026, 8, 15)).days
    mar = feats_1x2(kickoff_utc=datetime(2026, 3, 1, 20, 0, tzinfo=UTC))
    assert mar is not None
    assert mar["days_to_season_end"] == (datetime(2026, 6, 30) - datetime(2026, 3, 1)).days
    # kickoff unknown -> the cycle's `now` stands in (dated scrapes are
    # today..+1, so the drift is bounded by the scrape window)
    fallback = feats_1x2(kickoff_utc=None)
    assert fallback is not None
    assert fallback["day_of_week"] == NOW.weekday()


# ---------------------------------------------------------------------------
# Calibrator replay (manifest parameters only — no sklearn at runtime)
# ---------------------------------------------------------------------------
def test_isotonic_calibration_interpolates_and_clips() -> None:
    cal = {"kind": "isotonic", "x_thresholds": [0.2, 0.4, 0.8], "y_thresholds": [0.1, 0.5, 0.9]}
    p = calibrate(cal, np.array([0.3, 0.4, 0.05, 0.95]))
    assert p[0] == pytest.approx(0.3)  # midpoint of (0.2,0.1)-(0.4,0.5)
    assert p[1] == pytest.approx(0.5)
    assert p[2] == pytest.approx(0.1)  # out_of_bounds="clip" (left)
    assert p[3] == pytest.approx(0.9)  # out_of_bounds="clip" (right)


def test_platt_calibration_is_the_sigmoid() -> None:
    cal = {"kind": "platt", "coef": 2.0, "intercept": -1.0}
    p = calibrate(cal, np.array([0.5]))
    assert p[0] == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * 0.5 - 1.0))))


def test_calibration_output_never_reaches_0_or_1() -> None:
    cal = {"kind": "isotonic", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}
    p = calibrate(cal, np.array([0.0, 1.0]))
    assert 0.0 < p[0] < p[1] < 1.0


def test_unknown_calibrator_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown calibrator kind"):
        calibrate({"kind": "beta"}, np.array([0.5]))


# ---------------------------------------------------------------------------
# Loader fail-soft contract (no ML deps needed: refusal precedes any import)
# ---------------------------------------------------------------------------
def _write_artifacts(tmp_path: Path, manifest: dict) -> Path:
    (tmp_path / "value_filter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "value_filter_model.txt").write_text("placeholder", encoding="utf-8")
    return tmp_path


def test_load_returns_none_when_artifacts_missing(tmp_path: Path) -> None:
    assert ValueFilterModel.load(tmp_path) is None


def test_load_refuses_non_adopt_verdict(tmp_path: Path) -> None:
    # An unvalidated (or rejected) artifact must never be able to demote
    # live picks — the verdict gate runs BEFORE any ML import.
    manifest = {
        "verdict": "REJECT",
        "operating_point": {"q": 0.7},
        "features": ["edge"],
        "calibrator": {"kind": "platt", "coef": 1.0, "intercept": 0.0},
        "model": {"kind": "lgbm"},
    }
    assert ValueFilterModel.load(_write_artifacts(tmp_path, manifest)) is None


def test_allow_shadow_still_refuses_incomplete_manifest(tmp_path: Path) -> None:
    # VALUE_ML_MANIFEST_ALLOW_SHADOW relaxes ONLY the verdict gate — a
    # shadow candidate without a frozen operating point is still refused
    # (this check precedes any ML import, no lightgbm needed).
    manifest = {
        "verdict": "CANDIDATE (binding verdict: live shadow CLV + fresh 2627 season)",
        "operating_point": None,
        "features": ["edge"],
        "calibrator": {"kind": "platt", "coef": 1.0, "intercept": 0.0},
        "model": {"kind": "lgbm"},
    }
    assert ValueFilterModel.load(_write_artifacts(tmp_path, manifest), allow_shadow=True) is None


def test_load_honors_custom_artifact_filenames(tmp_path: Path) -> None:
    # Only the default-named artifacts exist: pointing the loader at the v2
    # filenames must be a clean miss (None), never a fallback to v1 files.
    manifest = {
        "verdict": "ADOPT",
        "operating_point": {"q": 0.7},
        "features": ["edge"],
        "calibrator": {"kind": "platt", "coef": 1.0, "intercept": 0.0},
        "model": {"kind": "lgbm"},
    }
    assert (
        ValueFilterModel.load(
            _write_artifacts(tmp_path, manifest),
            manifest_filename="value_filter_manifest_v2.json",
            model_filename="value_filter_model_v2.txt",
        )
        is None
    )


def test_load_refuses_manifest_without_operating_point(tmp_path: Path) -> None:
    manifest = {
        "verdict": "ADOPT",
        "operating_point": None,  # trainer writes null when nothing qualified
        "features": ["edge"],
        "calibrator": {"kind": "platt", "coef": 1.0, "intercept": 0.0},
        "model": {"kind": "lgbm"},
    }
    assert ValueFilterModel.load(_write_artifacts(tmp_path, manifest)) is None


def test_load_refuses_unsupported_model_kind(tmp_path: Path) -> None:
    manifest = {
        "verdict": "ADOPT",
        "operating_point": {"q": 0.7},
        "features": ["edge"],
        "calibrator": {"kind": "platt", "coef": 1.0, "intercept": 0.0},
        "model": {"kind": "logreg"},
    }
    assert ValueFilterModel.load(_write_artifacts(tmp_path, manifest)) is None


def test_load_survives_unreadable_manifest(tmp_path: Path) -> None:
    (tmp_path / "value_filter_manifest.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "value_filter_model.txt").write_text("placeholder", encoding="utf-8")
    assert ValueFilterModel.load(tmp_path) is None
