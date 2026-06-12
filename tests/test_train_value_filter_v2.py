"""v2 value-filter trainer: sweep sampling, ES fold fits, spent-season gates.

Loads scripts/ml/train_value_filter_v2.py by path (scripts/ is not a package)
and exercises the NEW v2 code paths on synthetic frames — no network, no real
data, nothing written outside tmp_path. importorskip-guarded: skips cleanly
without the `ml` extra; the XGBoost tests skip separately (challenger is an
optional dependency of the trainer too).
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "train_value_filter_v2.py"
_spec = importlib.util.spec_from_file_location("train_value_filter_v2", _SCRIPT)
assert _spec is not None and _spec.loader is not None
t2: Any = importlib.util.module_from_spec(_spec)
sys.modules["train_value_filter_v2"] = t2
_spec.loader.exec_module(t2)

SEED = 20260612


# ---------------------------------------------------------------------------
# Feature hygiene + constants (leakage gates for the v2 model feature sets)
# ---------------------------------------------------------------------------
def test_v2_model_feature_hygiene_passes() -> None:
    t2.assert_feature_hygiene_v2()  # build-breaking on any LABEL leak


def test_v2_model_features_extend_v1_without_labels() -> None:
    bvd = sys.modules["build_value_dataset"]
    assert set(t2.FS_V1.all) <= set(t2.FS_V2.all)
    assert t2.V2_NEW_FEATURES  # the v2 retrain actually adds features
    assert not set(t2.V2_NEW_FEATURES) & set(bvd.LABEL_COLUMNS)
    # era is constant in the maxavg window; drift is all-null for this source
    assert "era" not in t2.FS_V2.all
    assert "open_to_signal_drift" not in t2.FS_V2.all


def test_v2_artifact_names_never_collide_with_deployed_model_files() -> None:
    from app.models.value_filter import MANIFEST_FILENAME, MODEL_FILENAME

    assert MANIFEST_FILENAME != "value_filter_manifest_v2.json"
    assert MODEL_FILENAME != "value_filter_model_v2.txt"


# ---------------------------------------------------------------------------
# Sweep sampling (docs-recipe constraints)
# ---------------------------------------------------------------------------
def test_lgbm_model_sweep_draws_respect_documented_constraints() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        p = t2.sample_lgbm_params(rng)
        assert p["num_leaves"] < 2 ** p["max_depth"]  # LightGBM docs rule
        assert p["learning_rate"] in (0.02, 0.03, 0.05)
        assert p["min_child_samples"] in (100, 200, 350, 500, 750, 1000)
        assert 1.0 <= p["reg_lambda"] <= 30.0
        assert p["reg_alpha"] in (0.0, 0.1, 1.0)
        assert p["min_split_gain"] in (0.0, 0.01, 0.1)
        assert 0.6 <= p["colsample_bytree"] <= 1.0
        # bagging is only active with fraction<1 AND freq>0 (docs) — freq fixed 1
        assert 0.6 <= p["subsample"] <= 1.0
        assert p["subsample_freq"] == 1
        assert p["max_bin"] in (63, 127, 255)
        assert p["path_smooth"] in (0.0, 1.0, 10.0)
        assert isinstance(p["extra_trees"], bool)
        assert p["monotone_penalty"] in (0.0, 1.0, 2.0)


def test_xgb_model_sweep_draws_respect_documented_constraints() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        p = t2.sample_xgb_params(rng)
        assert p["max_depth"] in (2, 3, 4)
        assert p["min_child_weight"] in (25.0, 50.0, 100.0, 150.0)
        assert 1.0 <= p["reg_lambda"] <= 30.0
        assert p["reg_alpha"] in (0.0, 0.1, 1.0)
        assert p["gamma"] in (0.0, 0.05, 0.2)
        assert 0.6 <= p["subsample"] <= 1.0
        assert 0.6 <= p["colsample_bytree"] <= 1.0


# ---------------------------------------------------------------------------
# Synthetic fold data
# ---------------------------------------------------------------------------
_FS_TINY = t2.FeatureSet("tiny", ("edge", "best_price", "is_argmax_edge"), ("league",))


def _frame(n: int, season: str, rng: np.random.Generator) -> Any:
    edge = rng.uniform(-0.02, 0.08, n)
    return pd.DataFrame(
        {
            "season": season,
            "edge": edge,
            "best_price": rng.uniform(1.6, 4.0, n),
            "is_argmax_edge": rng.integers(0, 2, n).astype(bool),
            "league": rng.choice(["E0", "D1"], n),
            # label correlated with edge so the booster actually splits on it
            "y": (edge + rng.normal(0, 0.03, n) > 0.02).astype(float),
        }
    )


def _fold_data(rng: np.random.Generator) -> Any:
    train = pd.concat(
        [_frame(400, "1920", rng), _frame(400, "2021", rng), _frame(400, "2122", rng)],
        ignore_index=True,
    )
    tvf = sys.modules["train_value_filter"]
    folds = tvf.make_folds(["1920", "2021", "2122"])
    assert len(folds) == 1  # fit 1920 -> calib 2021 -> predict 2122
    return train, t2.make_fold_data(train, folds, _FS_TINY)


# ---------------------------------------------------------------------------
# Early-stopping fold fit (leak-free placement: eval set == calib season)
# ---------------------------------------------------------------------------
def test_lgbm_model_es_fit_stops_below_cap_and_respects_edge_monotone() -> None:
    rng = np.random.default_rng(SEED)
    _, fds = _fold_data(rng)
    fd = fds[0]
    params = {"learning_rate": 0.05, "max_depth": 3, "num_leaves": 7, "min_child_samples": 50}
    clf = t2.fit_lgbm_es(params, fd, _FS_TINY)
    assert clf.best_iteration_ is not None
    assert 1 <= clf.best_iteration_ <= t2.N_ESTIMATORS_CAP
    # monotone +1 on edge: raising edge alone never lowers the raw score
    probe = fd.x_prd.head(50).copy()
    lo = t2.predict_raw(clf, probe)
    probe["edge"] = probe["edge"] + 0.05
    hi = t2.predict_raw(clf, probe)
    assert (hi >= lo - 1e-12).all()


def test_oof_calibration_fills_predict_season_rows_only() -> None:
    rng = np.random.default_rng(SEED)
    train, fds = _fold_data(rng)
    res = t2.run_oof(train, fds, lambda fd: t2.fit_lgbm_es({"learning_rate": 0.05}, fd, _FS_TINY))
    filled = res.p_cal.notna()
    assert filled[train["season"] == "2122"].all()  # every predict row scored
    assert not filled[train["season"] != "2122"].any()  # fit/calib rows untouched
    assert 0.0 < res.log_loss < 1.5
    assert res.calib_kinds == ("platt",)  # n=400 < 1000 -> Platt fallback path
    assert len(res.best_iters) == 1


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")
def test_xgb_model_challenger_es_fit_and_monotone() -> None:
    rng = np.random.default_rng(SEED)
    _, fds = _fold_data(rng)
    fd = fds[0]
    clf = t2.fit_xgb_es({"max_depth": 3, "min_child_weight": 25.0}, fd)
    assert t2.best_iteration_of(clf) is not None
    probe = fd.x_prd.head(50).copy()
    lo = t2.predict_raw(clf, probe)
    probe["edge"] = probe["edge"] + 0.05
    hi = t2.predict_raw(clf, probe)
    assert (hi >= lo - 1e-12).all()


# ---------------------------------------------------------------------------
# Spent-holdout gates (2425/2526 must never enter any computation)
# ---------------------------------------------------------------------------
def _mini_candidates() -> Any:
    rows = []
    for league, consulted in (("E0", True), ("EC", False)):
        for season in ("1920", "2021", "2122", "2223", "2324", "2425", "2526"):
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "match_date": "2020-01-01",
                    "home_team": "A",
                    "away_team": "B",
                    "market": "1x2",
                    "selection": "H",
                    "era": "maxavg",
                    "consulted_before": consulted,
                    "clv_max": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_spent_seasons_are_dropped_at_load_for_model_and_fresh_slices(tmp_path: Path) -> None:
    p = tmp_path / "cand_v2.parquet"
    _mini_candidates().to_parquet(p, index=False)
    train, fresh = t2.load_candidates_v2(p)
    assert set(train["season"]) == {"1920", "2021", "2122", "2223", "2324"}
    assert not set(train["season"]) & set(t2.SPENT_SEASONS)
    assert not set(fresh["season"]) & set(t2.SPENT_SEASONS)
    assert set(train["league"]) == {"E0"}  # consulted universe only
    assert set(fresh["league"]) == {"EC"}  # never-consulted divisions only
    assert train["y"].notna().all()


def test_fresh_pool_builder_refuses_spent_seasons(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        t2.build_fresh_pool(tmp_path, ["EC"], ["2425"])


def test_fresh_pool_builder_tolerates_missing_cache_files(tmp_path: Path) -> None:
    pool = t2.build_fresh_pool(tmp_path, ["EC", "SC3"], ["1920", "2021"])
    assert len(pool) == 0  # nothing cached -> empty pool, no crash
