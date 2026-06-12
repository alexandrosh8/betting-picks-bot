"""Synthetic tests for the value-filter meta-model trainer.

Loads scripts/ml/train_value_filter.py by path (scripts/ is not a package).
ML deps are optional (`uv sync --extra ml`); everything here skips cleanly
when lightgbm/sklearn are absent so CI and the Docker image stay unaffected.
No network, no files written, no real data.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# pandas guard FIRST: a bare `import pandas` above would abort COLLECTION in
# a no-extras env instead of skipping (the importorskip-then-bind pattern
# keeps the module import-safe without the `ml` extra).
pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "train_value_filter.py"
_spec = importlib.util.spec_from_file_location("train_value_filter", _SCRIPT)
assert _spec is not None and _spec.loader is not None
tvf: Any = importlib.util.module_from_spec(_spec)
sys.modules["train_value_filter"] = tvf  # register BEFORE exec (dataclasses)
_spec.loader.exec_module(tvf)


# ---------------------------------------------------------------------------
# Walk-forward split correctness
# ---------------------------------------------------------------------------
def test_folds_are_strictly_walkforward() -> None:
    folds = tvf.make_folds(tvf.TRAIN_SEASONS)
    assert [f.predict_season for f in folds] == ["2122", "2223", "2324"]
    for f in folds:
        past = [*f.fit_seasons, f.calib_season]
        # chronological: every fit/calib season strictly precedes predict
        assert all(s < f.predict_season for s in past)
        assert f.calib_season == max(past), "calibration tail must be the most recent"
        # disjoint slices
        assert f.predict_season not in past
        assert len(set(f.fit_seasons)) == len(f.fit_seasons)
        assert f.calib_season not in f.fit_seasons


def test_no_test_season_rows_in_any_train_fold() -> None:
    test_seasons = set(tvf.TEST_SEASONS)
    for f in tvf.make_folds(tvf.TRAIN_SEASONS):
        fold_seasons = {*f.fit_seasons, f.calib_season, f.predict_season}
        assert fold_seasons <= set(tvf.TRAIN_SEASONS)
        assert not fold_seasons & test_seasons


def test_oof_predictions_only_cover_predict_seasons() -> None:
    df = _synthetic_frame(n=900, seasons=("1920", "2021", "2122", "2223", "2324", "2425"))
    folds = tvf.make_folds(tvf.TRAIN_SEASONS)
    spec = tvf.ModelSpec("lgbm_tiny", "lgbm", {"min_child_samples": 20, "n_estimators": 30})
    p = tvf.oof_calibrated(df, spec, folds, "y")
    covered = set(df.loc[p.notna(), "season"].unique())
    assert covered == {"2122", "2223", "2324"}  # never the holdout, never fit/calib seasons
    assert df.loc[p.notna(), "season"].isin(tvf.TEST_SEASONS).sum() == 0


# ---------------------------------------------------------------------------
# Feature hygiene (leakage gate)
# ---------------------------------------------------------------------------
def test_trainer_features_contain_no_label_columns() -> None:
    tvf.assert_feature_hygiene()  # build-breaking gate must pass
    bvd = sys.modules["build_value_dataset"]
    assert not set(tvf.FEATURES) & set(bvd.LABEL_COLUMNS)
    for forbidden in (
        "clv_max",
        "clv_pinn",
        "won",
        "profit_units",
        "pinn_close_fair",
        "max_close_fair",
        "bet_odds",
    ):
        assert forbidden not in tvf.FEATURES


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------
def _bet_row(match: str, market: str, sel: str, edge: float, price: float) -> dict[str, Any]:
    return {
        "league": "E0",
        "season": "2122",
        "match_date": "2021-10-02",
        "home_team": match,
        "away_team": "X",
        "market": market,
        "selection": sel,
        "edge": edge,
        "best_price": price,
        "won": True,
        "profit_units": price - 1.0,
        "clv_pinn": 0.01,
        "clv_max": 0.005,
    }


def test_select_bets_one_per_match_market_argmax_and_odds_floor() -> None:
    df = pd.DataFrame(
        [
            _bet_row("m1", "1x2", "H", 0.04, 2.0),
            _bet_row("m1", "1x2", "A", 0.06, 3.0),  # argmax for m1/1x2
            _bet_row("m1", "ou25", "over", 0.03, 1.9),
            _bet_row("m2", "1x2", "H", 0.09, 1.4),  # below the 1.6 odds floor
            _bet_row("m2", "1x2", "D", 0.02, 3.4),  # survives the floor
            _bet_row("m3", "1x2", "H", 0.001, 2.2),  # below thr
        ]
    )
    out = tvf.select_bets(df, "edge", 0.01, min_odds=1.6)
    assert len(out) == 3  # m1/1x2, m1/ou25, m2/1x2 — one bet per (match, market)
    m1_1x2 = out[(out["home_team"] == "m1") & (out["market"] == "1x2")]
    assert m1_1x2["selection"].tolist() == ["A"]  # argmax edge
    assert (out["best_price"] >= 1.6).all()
    assert "m3" not in out["home_team"].tolist()


# ---------------------------------------------------------------------------
# Monotone-edge sanity
# ---------------------------------------------------------------------------
def _synthetic_frame(
    n: int, seasons: tuple[str, ...]
) -> Any:  # pd.DataFrame (pd is bound via importorskip)
    rng = np.random.default_rng(11)
    edge = rng.uniform(0.0, 0.10, n)
    df = pd.DataFrame(
        {
            "season": rng.choice(list(seasons), n),
            "edge": edge,
            "fair_prob": rng.uniform(0.2, 0.7, n),
            "best_price": rng.uniform(1.6, 5.0, n),
            "pinn_price": rng.uniform(1.5, 4.5, n),
            "overround_pinn": rng.uniform(0.01, 0.06, n),
            "overround_best": rng.uniform(-0.02, 0.05, n),
            "devig_spread": rng.uniform(0.0, 0.02, n),
            "book_count": rng.integers(5, 16, n),
            "day_of_week": rng.integers(0, 7, n),
            "days_to_season_end": rng.integers(10, 300, n),
            "is_argmax_edge": rng.random(n) > 0.5,
            "league": rng.choice(["E0", "D1", "I1"], n),
            "market": rng.choice(["1x2", "ou25"], n),
            "selection_type": rng.choice(["fav", "dog", "draw"], n),
        }
    )
    # label depends positively on edge (plus noise) — monotone ground truth
    df["y"] = (rng.random(n) < 0.25 + 5.0 * edge).astype(float)
    return df


def test_lgbm_monotone_constraint_on_edge() -> None:
    df = _synthetic_frame(n=3000, seasons=("1920",))
    x, cats = tvf.prepare_matrix(df)
    spec = tvf.ModelSpec("lgbm_tiny", "lgbm", {"min_child_samples": 50, "n_estimators": 80})
    model = tvf.fit_model(spec, x, df["y"].to_numpy(dtype=int))
    probe = df.iloc[[0]].copy()
    probe = pd.concat([probe] * 60, ignore_index=True)
    probe["edge"] = np.linspace(0.0, 0.10, 60)
    p = tvf.predict_raw(model, tvf.prepare_matrix(probe, cats)[0])
    assert np.all(np.diff(p) >= -1e-9), "P(beat close) must be non-decreasing in edge"


# ---------------------------------------------------------------------------
# Calibration (chronological-tail, prefit pattern)
# ---------------------------------------------------------------------------
def test_calibrator_isotonic_above_floor_platt_below() -> None:
    rng = np.random.default_rng(3)
    p_raw = rng.uniform(0, 1, 1500)
    y = (rng.random(1500) < p_raw).astype(int)
    kind, _ = tvf.fit_calibrator(p_raw, y)
    assert kind == "isotonic"
    kind_small, _ = tvf.fit_calibrator(p_raw[:400], y[:400])
    assert kind_small == "platt"


def test_calibrated_model_output_is_monotone_in_raw_score() -> None:
    rng = np.random.default_rng(4)
    p_raw = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < p_raw).astype(int)
    cal = tvf.fit_calibrator(p_raw, y)
    grid = np.linspace(0.0, 1.0, 101)
    out = tvf.apply_calibrator(cal, grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Operating-point selection: pre-declared criterion, train-only inputs
# ---------------------------------------------------------------------------
def _op(q: float, n: int, roi: float, inc: float | None, se: float) -> Any:
    boot = None if inc is None else tvf.Boot(inc, se, inc - 2 * se, inc + 2 * se)
    stats = tvf.SetStats(
        label=f"q>={q}",
        n=n,
        n_matches=n,
        hit=0.5,
        roi=roi,
        roi_boot=None,
        clv_pinn=None,
        clv_pinn_se=None,
        clv_max=None,
        clv_max_se=None,
        beat_max=None,
        n_missing_pinn=0,
        n_missing_max=0,
        inc_pinn=None,
        inc_max=boot,
    )
    return tvf.OperatingPoint(q=q, stats=stats)


def test_operating_point_criterion_max_roi_subject_to_gates() -> None:
    points = [
        _op(0.40, n=1000, roi=0.010, inc=0.020, se=0.005),  # qualifies
        _op(0.50, n=500, roi=0.050, inc=0.030, se=0.005),  # qualifies, best ROI
        _op(0.60, n=200, roi=0.200, inc=0.050, se=0.001),  # fails n >= 300
        _op(0.70, n=400, roi=0.300, inc=0.010, se=0.020),  # fails inc - 2SE > 0
        _op(0.80, n=400, roi=0.250, inc=None, se=0.0),  # no increment computable
    ]
    chosen = tvf.choose_operating_point(points)
    assert chosen is not None
    assert chosen.q == 0.50  # max ROI among the QUALIFYING points only


def test_operating_point_returns_none_when_nothing_qualifies() -> None:
    points = [
        _op(0.40, n=200, roi=0.10, inc=0.05, se=0.001),  # n too small
        _op(0.50, n=900, roi=0.20, inc=0.01, se=0.02),  # inc not > 2SE
    ]
    assert tvf.choose_operating_point(points) is None


def test_operating_point_selection_consumes_only_train_sweep_stats() -> None:
    """The criterion function operates on OperatingPoint rows alone — it has
    no parameter through which holdout data could enter."""
    import inspect

    params = inspect.signature(tvf.choose_operating_point).parameters
    assert list(params) == ["points", "min_bets"]


# ---------------------------------------------------------------------------
# Match-clustered bootstrap sanity
# ---------------------------------------------------------------------------
def test_boot_increment_positive_when_selected_beats_null() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(300):
        rows.append(
            {
                "league": "E0",
                "season": "2122",
                "match_date": f"2021-{(i % 9) + 1:02d}-01",
                "home_team": f"team{i}",
                "away_team": "X",
                "market": "1x2",
                "selection": "H",
                "clv_max": rng.normal(0.0, 0.02),
                "clv_pinn": rng.normal(0.0, 0.02),
            }
        )
    null = pd.DataFrame(rows)
    sel = null.iloc[:80].copy()
    sel["clv_max"] = sel["clv_max"] + 0.05  # selected set clearly better
    boot = tvf.boot_increment(sel, null, "clv_max", n_boot=300, rng=rng)
    assert boot is not None
    assert boot.point > 0.03
    assert np.isfinite(boot.se)
    assert boot.lo > 0.0
