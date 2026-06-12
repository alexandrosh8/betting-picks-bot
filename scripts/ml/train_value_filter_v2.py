"""Train + evaluate the value-filter META-MODEL **v2** (meta-labeling).

Builds on scripts/ml/train_value_filter.py (v1) — same doctrine: the edge is
sharp-vs-soft line shopping; ML only SCORES VALUE CANDIDATES (Lopez de Prado
meta-labeling), never match outcomes. Label: clv_max > 0 (bet beats the
vig-free Max-of-books close). Selection currency: pooled OOF log-loss
(Brier + post-isotonic ECE reported) — never accuracy.

SPENT-HOLDOUT DISCIPLINE (binding, .claude/memory/decisions.md):
  Seasons 2425+2526 have been consulted 4x and are SPENT. This script NEVER
  loads a 2425/2526 row into any computation — they are filtered out at load
  time. All development (sweeps, early-stopping protocol, monotone method,
  LightGBM-vs-XGBoost selection) happens EXCLUSIVELY on the nested
  season-blocked expanding walk-forward within train seasons <= 2324.
  The only fresh test slice is the never-consulted division pool
  (consulted_before=False: EC/SC1/SC2/SC3), seasons excluding 2425/2526 —
  a pre-registered ONE-SHOT transportability check, not an adoption gate.
  The binding verdict for v2 is live shadow CLV + the fresh 2627 season.

PRE-REGISTERED PROTOCOL v2 — frozen in code before any result is computed:
  Universe   TRAIN: the 18 consulted leagues, maxavg era, seasons 1920-2324,
             markets 1x2+ou25, min_odds 1.6, devig differential_margin
             (ADR-0006). Identical row set to v1 (verified byte-identical
             between value_candidates.parquet and value_candidates_v2.parquet
             on the train universe: 18,786 labeled rows).
  Folds      fit 1920          -> calib 2021 -> predict 2122
             fit 1920+2021     -> calib 2122 -> predict 2223
             fit 1920..2122    -> calib 2223 -> predict 2324
  ES proto   Uniform option A (pre-registered per the GBM docs research):
             within fold k the early-stopping eval set is the CALIBRATION
             season k-1, never the predict season. num_boost_round cap 2000,
             early_stopping(stopping_rounds=100, first_metric_only=True,
             min_delta=1e-4). The calib season does double duty (chooses
             best_iteration AND fits isotonic) — both are freezing operations
             on strictly pre-predict data. Residual caveat: isotonic on the
             same season used for best_iteration is mildly optimistic ->
             calib-season ECE is reported alongside OOF ECE.
  Arms       1. v1_grounding   exact v1 winner (lgbm_mcs500_n200, fixed 200
                               rounds, monotone basic) on V1 features —
                               must reproduce manifest OOF log-loss 0.65175.
             2. v2_lgbm_v1feat the SAME sweep draws evaluated on V1 features
                               (isolates hyperparameter lift).
             3. v2_lgbm        LightGBM random sweep (docs recipe) on V2
                               features (headline).
             4. xgb_v2feat     XGBoost 3.2 challenger, identical folds +
                               calibration, on V2 features.
             NO learning-to-rank arm: the docs research verdict is that LTR
             does not fit (ranker outputs are relevance scores, not the
             calibrated probabilities the pre-registered pipeline consumes;
             (match, market) qid groups hold 1-3 rows -> near-zero effective
             pairs; bucketing clv_max is a fresh researcher degree of
             freedom). Headline challenger stays binary:logistic.
  Sweep      LightGBM: random search (default 100 draws, seeded) over
             num_leaves<2^max_depth pairs {(2,3),(3,3),(3,7),(4,3),(4,7),
             (4,15)}, learning_rate {0.02,0.03,0.05}, min_data_in_leaf
             {100,200,350,500,750,1000}, lambda_l2 logU[1,30], lambda_l1
             {0,0.1,1}, min_gain_to_split {0,0.01,0.1}, feature_fraction
             U[0.6,1.0], bagging_fraction U[0.6,1.0] WITH bagging_freq=1
             (bagging was structurally off in v1: subsample=1.0, freq=0),
             max_bin {63,127,255}, path_smooth {0,1,10}, extra_trees
             {false,true}, monotone on `edge` only with method="advanced",
             monotone_penalty {0,1,2}. DART/GOSS excluded per the docs
             verdict (the ES callback hard-disables itself in dart mode;
             GOSS is a large-data speed lever, pure variance at our n).
             XGBoost: tree_method=hist, device=cpu, enable_categorical=True,
             monotone_constraints={"edge": 1}, max_bin=512 (histogram +
             monotone interaction mitigation per the XGB tutorial), eta 0.03,
             cap 2000 + early_stopping_rounds=100 on the calib season; sweep
             max_depth {2,3,4}, min_child_weight {25,50,100,150} (hessian
             ~0.24/row at base rate 0.615), reg_lambda logU[1,30], reg_alpha
             {0,0.1,1}, gamma {0,0.05,0.2}, subsample U[0.6,1.0],
             colsample_bytree U[0.6,1.0].
  Selection  Within arm: min pooled OOF log-loss (calibrated). Across arms:
             the LGBM winner is the GLOBAL argmin of pooled OOF log-loss over
             {v2_lgbm_v1feat, v2_lgbm} — the feature set is part of the
             search space, per the round instruction "model selection on
             pooled OOF log-loss". The XGB challenger replaces the LGBM
             winner iff strictly lower pooled OOF log-loss AND post-isotonic
             OOF ECE <= LGBM ECE + 0.005 (pre-registered tolerance; an XGB
             win also requires a loader branch in app/models/value_filter.py
             before it could serve).
             AMENDMENT NOTE (honest log): the first draft of this rule fixed
             the final feature set to V2; it was amended to the global argmin
             after the train-OOF sweep tables existed but STRICTLY BEFORE the
             fresh one-shot was consulted (the first full run was killed at
             the XGB sweep, fresh slice untouched). All quantities involved
             in the amendment are train-OOF — the development sandbox.
  Operating  v1 criterion unchanged: q on the pre-declared grid maximizing
  point      TRAIN-OOF ROI s.t. n >= 300 and incCLV_max - 2*bootstrap_SE > 0
             (match-clustered bootstrap, thr=0 bet-everything null).
  Final fit  fit 1920-2223, early-stop + isotonic on 2324 (tail), matching
             the fold protocol. Artifacts are written ONLY to *_v2 paths —
             the deployed v1 artifacts are never overwritten. The v2 manifest
             verdict is "CANDIDATE" (never "ADOPT" from this script), so
             app/models/value_filter.py refuses to serve it by construction.
  Fresh      ONE-SHOT on consulted_before=False rows (EC/SC1/SC2/SC3),
  one-shot   maxavg era, seasons != 2425/2526 (~1.9k effective rows; the
             dataset round confirmed this clears the 1,500 pre-registered
             floor only when all four divisions are pooled — SC2+SC3 alone
             is underpowered). Reported as a transportability check with
             CIs; NOT the binding verdict.

Run:
    uv run --extra ml python scripts/ml/train_value_filter_v2.py
    uv run --extra ml python scripts/ml/train_value_filter_v2.py --quick

Decision-support only — nothing here places bets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss, log_loss
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"missing ML dependency ({exc.name}); install with: uv sync --extra ml"
    ) from exc

try:  # the XGBoost challenger is optional — the LGBM arms still run without it
    import xgboost as xgb_lib
except ImportError:  # pragma: no cover - environment guard
    xgb_lib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_by_path(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec (dataclasses)
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).resolve().parent
tvf: Any = _import_by_path("train_value_filter", _HERE / "train_value_filter.py")
bvd: Any = sys.modules["build_value_dataset"]  # imported by tvf

# ---------------------------------------------------------------------------
# Frozen v2 constants
# ---------------------------------------------------------------------------
SEED: int = int(tvf.SEED)  # 20260612 — model fits
SWEEP_SEED: int = SEED + 100  # parameter draws
SPENT_SEASONS: tuple[str, ...] = ("2425", "2526")  # NEVER loaded
TRAIN_SEASONS: tuple[str, ...] = tuple(tvf.TRAIN_SEASONS)
LEAGUES_18: tuple[str, ...] = tuple(tvf.LEAGUES_18)
FRESH_LEAGUES: tuple[str, ...] = ("EC", "SC1", "SC2", "SC3")
MIN_ODDS: float = float(tvf.MIN_ODDS)
N_ESTIMATORS_CAP = 2000
ES_ROUNDS = 100  # ~3/learning_rate at 0.03
ES_MIN_DELTA = 1e-4
ECE_TOLERANCE = 0.005  # challenger adoption: ECE_xgb <= ECE_lgbm + this
V1_GROUNDING_LL = 0.651751143188923  # manifest oof log-loss to reproduce

MATCH_KEY: list[str] = list(tvf.MATCH_KEY)
MM_KEY: list[str] = list(tvf.MM_KEY)

# Feature sets. V1 = the deployed manifest list. V2 = every SIGNAL column in
# the v2 SCHEMA except `era` (constant in the maxavg window) and
# `open_to_signal_drift` (all-null for this source), plus league/market
# (allowed ID context, as in v1). Hygiene re-asserted below.
V1_FEATURES_NUM: tuple[str, ...] = tuple(tvf.FEATURES_NUM)
V1_FEATURES_CAT: tuple[str, ...] = tuple(tvf.FEATURES_CAT)
_EXCLUDED_SIGNALS: tuple[str, ...] = ("era", "open_to_signal_drift")
V2_FEATURES_CAT: tuple[str, ...] = ("league", "market", "selection_type", "odds_band")
V2_FEATURES_NUM: tuple[str, ...] = tuple(
    c for c in bvd.FEATURE_COLUMNS if c not in _EXCLUDED_SIGNALS and c not in V2_FEATURES_CAT
)
V2_NEW_FEATURES: tuple[str, ...] = tuple(
    c for c in (*V2_FEATURES_NUM, *V2_FEATURES_CAT) if c not in (*V1_FEATURES_NUM, *V1_FEATURES_CAT)
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    num: tuple[str, ...]
    cat: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return self.num + self.cat


FS_V1 = FeatureSet("v1", V1_FEATURES_NUM, V1_FEATURES_CAT)
FS_V2 = FeatureSet("v2", V2_FEATURES_NUM, V2_FEATURES_CAT)


def assert_feature_hygiene_v2() -> None:
    """Build-breaking leakage gate for BOTH feature sets."""
    bvd.assert_no_label_leak()
    labels = set(bvd.LABEL_COLUMNS)
    signal = set(bvd.FEATURE_COLUMNS)
    for fs in (FS_V1, FS_V2):
        leaked = set(fs.all) & labels
        assert not leaked, f"LABEL columns leaked into {fs.name} features: {leaked}"
        extra = set(fs.all) - signal - set(tvf.ALLOWED_ID_FEATURES)
        assert not extra, f"non-SIGNAL features outside the allowed ID set ({fs.name}): {extra}"
    assert "edge" in V2_FEATURES_NUM and "edge" in V1_FEATURES_NUM


# ---------------------------------------------------------------------------
# Data loading (2425/2526 filtered out HERE — never reach any computation)
# ---------------------------------------------------------------------------
def load_candidates_v2(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_cand, fresh_cand) from the v2 parquet. Spent seasons dropped."""
    df = pd.read_parquet(path)
    df = df[(df["era"] == "maxavg") & ~df["season"].isin(SPENT_SEASONS)].copy()
    for col in MM_KEY:
        df[col] = df[col].astype(str)
    df = df.sort_values([*MM_KEY, "selection"], kind="mergesort").reset_index(drop=True)
    df["y"] = np.where(df["clv_max"].notna(), (df["clv_max"] > 0).astype(float), np.nan)
    train = df[df["league"].isin(LEAGUES_18) & df["season"].isin(TRAIN_SEASONS)].copy()
    fresh = df[~df["consulted_before"].astype(bool)].copy()
    assert not set(train["season"]) & set(SPENT_SEASONS)
    assert not set(fresh["season"]) & set(SPENT_SEASONS)
    assert set(fresh["league"]) <= set(FRESH_LEAGUES), sorted(set(fresh["league"]))
    return train, fresh


def build_fresh_pool(
    cache_dir: Path, leagues: Sequence[str], seasons: Sequence[str]
) -> pd.DataFrame:
    """No-edge-floor pool for the fresh divisions (null + edge baselines).

    Tolerant variant of tvf.build_full_pool: fresh divisions have sparse
    cache coverage (SC2/SC3 histories are short) — missing files are counted,
    not fatal. Reads cache only; never touches 2425/2526 files.
    """
    cands: list[Any] = []
    missing = 0
    for lg in leagues:
        for s in seasons:
            assert s not in SPENT_SEASONS
            f = cache_dir / f"{s}_{lg}.csv"
            if not f.exists():
                missing += 1
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
            cands.extend(bvd.candidates_from_rows(lg, s, rows, min_edge=-10.0))
    if missing:
        print(f"  fresh pool: {missing} league-season cache files absent (expected; skipped)")
    pool = bvd._to_dataframe(cands)
    pool = pool[pool["era"] == "maxavg"].copy()
    for col in MM_KEY:
        pool[col] = pool[col].astype(str)
    return pool.sort_values([*MM_KEY, "selection"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matrices + per-fold prepared data
# ---------------------------------------------------------------------------
def prepare_matrix_v2(
    df: pd.DataFrame, fs: FeatureSet, categories: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    x = df[list(fs.all)].copy()
    x["is_argmax_edge"] = x["is_argmax_edge"].astype("int8")
    if "fair_prob_rank" in x.columns:
        x["fair_prob_rank"] = x["fair_prob_rank"].astype("int32")
    if categories is None:  # vocabulary comes from the FIT slice only
        categories = {c: sorted(x[c].astype(str).unique()) for c in fs.cat}
    for c in fs.cat:
        x[c] = pd.Categorical(x[c].astype(str), categories=categories[c])
    return x, categories


@dataclass(frozen=True)
class FoldData:
    fold: Any  # tvf.Fold
    x_fit: pd.DataFrame
    y_fit: np.ndarray
    x_cal: pd.DataFrame
    y_cal: np.ndarray
    x_prd: pd.DataFrame
    prd_index: pd.Index
    cats: dict[str, list[str]]


def make_fold_data(
    train_cand: pd.DataFrame, folds: Sequence[Any], fs: FeatureSet
) -> list[FoldData]:
    out: list[FoldData] = []
    for fold in folds:
        fit_df = train_cand[train_cand["season"].isin(fold.fit_seasons) & train_cand["y"].notna()]
        cal_df = train_cand[(train_cand["season"] == fold.calib_season) & train_cand["y"].notna()]
        prd_df = train_cand[train_cand["season"] == fold.predict_season]
        x_fit, cats = prepare_matrix_v2(fit_df, fs)
        out.append(
            FoldData(
                fold=fold,
                x_fit=x_fit,
                y_fit=fit_df["y"].to_numpy(dtype=int),
                x_cal=prepare_matrix_v2(cal_df, fs, cats)[0],
                y_cal=cal_df["y"].to_numpy(dtype=int),
                x_prd=prepare_matrix_v2(prd_df, fs, cats)[0],
                prd_index=prd_df.index,
                cats=cats,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Config sampling (docs recipe — exact ranges in the module docstring)
# ---------------------------------------------------------------------------
_DEPTH_LEAVES: tuple[tuple[int, int], ...] = ((2, 3), (3, 3), (3, 7), (4, 3), (4, 7), (4, 15))


def sample_lgbm_params(rng: np.random.Generator) -> dict[str, Any]:
    """One sweep draw — sklearn-API param names; num_leaves < 2^max_depth."""
    max_depth, num_leaves = _DEPTH_LEAVES[int(rng.integers(len(_DEPTH_LEAVES)))]
    return {
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05])),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": int(rng.choice([100, 200, 350, 500, 750, 1000])),
        "reg_lambda": float(math.exp(rng.uniform(math.log(1.0), math.log(30.0)))),
        "reg_alpha": float(rng.choice([0.0, 0.1, 1.0])),
        "min_split_gain": float(rng.choice([0.0, 0.01, 0.1])),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "subsample_freq": 1,  # docs: bagging only active when fraction<1 AND freq>0
        "max_bin": int(rng.choice([63, 127, 255])),
        "path_smooth": float(rng.choice([0.0, 1.0, 10.0])),
        "extra_trees": bool(rng.choice([False, True])),
        "monotone_penalty": float(rng.choice([0.0, 1.0, 2.0])),
    }


def sample_xgb_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "max_depth": int(rng.choice([2, 3, 4])),
        "min_child_weight": float(rng.choice([25, 50, 100, 150])),
        "reg_lambda": float(math.exp(rng.uniform(math.log(1.0), math.log(30.0)))),
        "reg_alpha": float(rng.choice([0.0, 0.1, 1.0])),
        "gamma": float(rng.choice([0.0, 0.05, 0.2])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
    }


# ---------------------------------------------------------------------------
# Fitters — all early-stop on the CALIBRATION season (leak-free placement)
# ---------------------------------------------------------------------------
def fit_lgbm_es(params: Mapping[str, Any], fd: FoldData, fs: FeatureSet) -> lgb.LGBMClassifier:
    mono = [1 if f == "edge" else 0 for f in fd.x_fit.columns]
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=N_ESTIMATORS_CAP,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
        monotone_constraints=mono,
        monotone_constraints_method="advanced",
        **params,
    )
    clf.fit(
        fd.x_fit,
        fd.y_fit,
        eval_set=[(fd.x_cal, fd.y_cal)],
        eval_metric="binary_logloss",
        categorical_feature=list(fs.cat),
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=ES_ROUNDS,
                first_metric_only=True,
                min_delta=ES_MIN_DELTA,
                verbose=False,
            )
        ],
    )
    return clf


def fit_xgb_es(params: Mapping[str, Any], fd: FoldData) -> Any:
    assert xgb_lib is not None
    clf = xgb_lib.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        enable_categorical=True,
        monotone_constraints={"edge": 1},
        max_bin=512,  # monotone+histogram interaction mitigation (XGB docs)
        learning_rate=0.03,
        n_estimators=N_ESTIMATORS_CAP,
        early_stopping_rounds=ES_ROUNDS,
        n_jobs=4,
        random_state=SEED,
        verbosity=0,
        **params,
    )
    clf.fit(fd.x_fit, fd.y_fit, eval_set=[(fd.x_cal, fd.y_cal)], verbose=False)
    return clf


def fit_v1_grounding(fd: FoldData) -> Any:
    """Exact v1 winner via the v1 code path (monotone basic, fixed 200 rounds)."""
    spec = next(s for s in tvf.SPECS if s.name == "lgbm_mcs500_n200")
    return tvf.fit_model(spec, fd.x_fit, fd.y_fit)


def predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def best_iteration_of(model: Any) -> int | None:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    return None


# ---------------------------------------------------------------------------
# OOF runner (shared by every arm — identical folds, identical calibration)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OofResult:
    p_cal: pd.Series  # calibrated OOF probability per predict-season row
    log_loss: float
    brier: float
    ece: float
    calib_ece: float  # mean calib-season ECE (double-duty caveat metric)
    best_iters: tuple[int | None, ...]
    calib_kinds: tuple[str, ...]


def run_oof(
    train_cand: pd.DataFrame,
    fold_datas: Sequence[FoldData],
    fit_fn: Callable[[FoldData], Any],
) -> OofResult:
    p = pd.Series(np.nan, index=train_cand.index, dtype=float)
    iters: list[int | None] = []
    kinds: list[str] = []
    calib_eces: list[float] = []
    for fd in fold_datas:
        model = fit_fn(fd)
        p_raw_cal = predict_raw(model, fd.x_cal)
        cal = tvf.fit_calibrator(p_raw_cal, fd.y_cal)
        p_cal_on_cal = tvf.apply_calibrator(cal, p_raw_cal)
        _, cal_ece = tvf.reliability_table(p_cal_on_cal, fd.y_cal.astype(float))
        calib_eces.append(cal_ece)
        p.loc[fd.prd_index] = tvf.apply_calibrator(cal, predict_raw(model, fd.x_prd))
        iters.append(best_iteration_of(model))
        kinds.append(cal[0])
    mask = p.notna() & train_cand["y"].notna()
    y = train_cand.loc[mask, "y"].to_numpy(dtype=float)
    pm = p[mask].to_numpy(dtype=float)
    _, ece = tvf.reliability_table(pm, y)
    return OofResult(
        p_cal=p,
        log_loss=float(log_loss(y, pm, labels=[0.0, 1.0])),
        brier=float(brier_score_loss(y, pm)),
        ece=float(ece),
        calib_ece=float(np.mean(calib_eces)),
        best_iters=tuple(iters),
        calib_kinds=tuple(kinds),
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Draw:
    index: int
    params: dict[str, Any]
    result: OofResult


def run_sweep(
    train_cand: pd.DataFrame,
    fold_datas: Sequence[FoldData],
    fs: FeatureSet,
    param_draws: Sequence[dict[str, Any]],
    kind: str,  # "lgbm" | "xgb"
    log_every: int = 10,
) -> list[Draw]:
    out: list[Draw] = []
    t0 = time.monotonic()
    for i, params in enumerate(param_draws):
        if kind == "lgbm":
            res = run_oof(train_cand, fold_datas, lambda fd, p=params: fit_lgbm_es(p, fd, fs))
        else:
            res = run_oof(train_cand, fold_datas, lambda fd, p=params: fit_xgb_es(p, fd))
        out.append(Draw(i, dict(params), res))
        if (i + 1) % log_every == 0 or i + 1 == len(param_draws):
            best = min(out, key=lambda d: d.result.log_loss)
            print(
                f"  [{kind}/{fs.name}] draw {i + 1:3d}/{len(param_draws)} | "
                f"elapsed {time.monotonic() - t0:6.1f}s | "
                f"best ll {best.result.log_loss:.5f} (draw {best.index})",
                flush=True,
            )
    return out


def sweep_rows(arm: str, draws: Sequence[Draw]) -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "draw": d.index,
            "log_loss": d.result.log_loss,
            "brier": d.result.brier,
            "ece": d.result.ece,
            "calib_ece": d.result.calib_ece,
            "best_iters": json.dumps(list(d.result.best_iters)),
            "params": json.dumps(d.params, sort_keys=True),
        }
        for d in draws
    ]


# ---------------------------------------------------------------------------
# Importances
# ---------------------------------------------------------------------------
def importance_rows(model: Any, feature_names: Sequence[str]) -> list[tuple[str, float, int]]:
    """(feature, gain, splits) sorted by gain desc — LGBM or XGB."""
    if isinstance(model, lgb.LGBMClassifier):
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        names = booster.feature_name()
        return sorted(
            [(names[i], float(gain[i]), int(split[i])) for i in range(len(names))],
            key=lambda t: t[1],
            reverse=True,
        )
    xbooster = model.get_booster()
    gain_d = xbooster.get_score(importance_type="total_gain")
    split_d = xbooster.get_score(importance_type="weight")
    return sorted(
        [(f, float(gain_d.get(f, 0.0)), int(split_d.get(f, 0))) for f in feature_names],
        key=lambda t: t[1],
        reverse=True,
    )


def print_importances(model: Any, feature_names: Sequence[str]) -> list[dict[str, Any]]:
    rows = importance_rows(model, feature_names)
    total = sum(g for _, g, _ in rows) or 1.0
    out: list[dict[str, Any]] = []
    for name, gain, split in rows:
        tag = " [v2-NEW]" if name in V2_NEW_FEATURES else ""
        print(
            f"  {name:>24} | gain {gain:14.1f} ({gain / total * 100:5.1f}%) | "
            f"splits {split:4d}{tag}"
        )
        out.append(
            {
                "feature": name,
                "gain": gain,
                "gain_pct": gain / total * 100,
                "splits": split,
                "v2_new": name in V2_NEW_FEATURES,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "data/ml/value_candidates_v2.parquet"
    )
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/ml/cache")
    ap.add_argument(
        "--pool-cache", type=Path, default=REPO_ROOT / "data/ml/value_pool_full.parquet"
    )
    ap.add_argument("--n-draws-lgbm", type=int, default=100)
    ap.add_argument("--n-draws-xgb", type=int, default=80)
    ap.add_argument("--n-boot-train", type=int, default=500)
    ap.add_argument("--n-boot-fresh", type=int, default=2000)
    ap.add_argument("--skip-fresh", action="store_true", help="skip the one-shot fresh slice")
    ap.add_argument("--quick", action="store_true", help="smoke run: 4/3 draws, 100 bootstrap")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/ml")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_draws_lgbm, args.n_draws_xgb = 4, 3
        args.n_boot_train, args.n_boot_fresh = 100, 100

    assert_feature_hygiene_v2()
    rng_train = np.random.default_rng(SEED)
    rng_fresh = np.random.default_rng(SEED + 1)

    print(f"value-filter v2 trainer | seed {SEED} | sweep seed {SWEEP_SEED}")
    print("label: clv_max > 0 | selection: pooled OOF log-loss (never accuracy)")
    print(f"V1 features ({len(FS_V1.all)}): {', '.join(FS_V1.all)}")
    print(f"V2 features ({len(FS_V2.all)}): {', '.join(FS_V2.all)}")
    print(f"V2-new features ({len(V2_NEW_FEATURES)}): {', '.join(V2_NEW_FEATURES)}")
    print(
        "SPENT-HOLDOUT DISCIPLINE: seasons 2425+2526 are SPENT (consulted 4x) and are "
        "filtered out at load — no number in this run touches them. Binding verdict "
        "= live shadow CLV + fresh 2627 season."
    )

    # ---- data ---------------------------------------------------------------
    train_cand, fresh_cand = load_candidates_v2(args.dataset)
    print(
        f"\ntrain candidates: {len(train_cand)} rows "
        f"({int(train_cand['y'].notna().sum())} labeled) | "
        f"fresh candidates (never-consulted divisions): {len(fresh_cand)} rows"
    )
    folds = tvf.make_folds(TRAIN_SEASONS)
    for f in folds:
        print(f"fold: fit {f.fit_seasons} -> calib {f.calib_season} -> predict {f.predict_season}")

    fd_v1 = make_fold_data(train_cand, folds, FS_V1)
    fd_v2 = make_fold_data(train_cand, folds, FS_V2)

    # ---- arm 1: v1 grounding ------------------------------------------------
    print("\nARM v1_grounding (exact v1 winner, v1 features, fixed 200 rounds):")
    res_ground = run_oof(train_cand, fd_v1, fit_v1_grounding)
    drift = abs(res_ground.log_loss - V1_GROUNDING_LL)
    print(
        f"  log-loss {res_ground.log_loss:.6f} | manifest {V1_GROUNDING_LL:.6f} | "
        f"drift {drift:.2e} {'OK' if drift < 1e-6 else 'MISMATCH — harness not at parity'}"
    )

    # ---- shared sweep draws (identical configs on both feature sets) --------
    rng_sweep = np.random.default_rng(SWEEP_SEED)
    lgbm_draws = [sample_lgbm_params(rng_sweep) for _ in range(args.n_draws_lgbm)]
    xgb_draws = [sample_xgb_params(rng_sweep) for _ in range(args.n_draws_xgb)]

    print(f"\nARM v2_lgbm (sweep, {len(lgbm_draws)} draws, V2 features):")
    sweep_v2 = run_sweep(train_cand, fd_v2, FS_V2, lgbm_draws, "lgbm")
    best_v2 = min(sweep_v2, key=lambda d: d.result.log_loss)

    print(f"\nARM v2_lgbm_v1feat (same {len(lgbm_draws)} draws, V1 features):")
    sweep_v1f = run_sweep(train_cand, fd_v1, FS_V1, lgbm_draws, "lgbm")
    best_v1f = min(sweep_v1f, key=lambda d: d.result.log_loss)

    sweep_xgb: list[Draw] = []
    best_xgb: Draw | None = None
    if xgb_lib is None:
        print("\nARM xgb_v2feat SKIPPED: xgboost not installed (uv sync --extra ml)")
    else:
        print(f"\nARM xgb_v2feat (XGBoost {xgb_lib.__version__}, {len(xgb_draws)} draws):")
        sweep_xgb = run_sweep(train_cand, fd_v2, FS_V2, xgb_draws, "xgb")
        best_xgb = min(sweep_xgb, key=lambda d: d.result.log_loss)

    # ---- sweep log artifact --------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_log = (
        sweep_rows("v2_lgbm", sweep_v2)
        + sweep_rows("v2_lgbm_v1feat", sweep_v1f)
        + sweep_rows("xgb_v2feat", sweep_xgb)
    )
    sweep_path = args.out_dir / "value_filter_v2_sweep.csv"
    pd.DataFrame(sweep_log).to_csv(sweep_path, index=False)

    # ---- arm comparison table (pre-registered: pooled OOF log-loss) ----------
    print("\nOOF ARM COMPARISON (pooled over folds; calibrated; never accuracy):")
    print(
        f"  {'arm':>16} | {'log-loss':>9} | {'brier':>8} | {'ece':>7} | "
        f"{'calib-ece':>9} | best_iters"
    )

    def _arm_line(name: str, r: OofResult) -> None:
        print(
            f"  {name:>16} | {r.log_loss:9.5f} | {r.brier:8.5f} | {r.ece:7.4f} | "
            f"{r.calib_ece:9.4f} | {list(r.best_iters)}"
        )

    _arm_line("v1_grounding", res_ground)
    _arm_line("v2_lgbm_v1feat", best_v1f.result)
    _arm_line("v2_lgbm", best_v2.result)
    if best_xgb is not None:
        _arm_line("xgb_v2feat", best_xgb.result)

    # ---- headline selection (global argmin on pooled OOF log-loss; see
    # docstring Selection + amendment note) -------------------------------------
    selected_kind = "lgbm"
    if best_v1f.result.log_loss <= best_v2.result.log_loss:
        selected_draw, selected_fs = best_v1f, FS_V1
    else:
        selected_draw, selected_fs = best_v2, FS_V2
    if best_xgb is not None:
        xgb_wins = (
            best_xgb.result.log_loss < selected_draw.result.log_loss
            and best_xgb.result.ece <= selected_draw.result.ece + ECE_TOLERANCE
        )
        if xgb_wins:
            selected_kind, selected_draw = "xgb", best_xgb
            print(
                "\nCHALLENGER ADOPTED: XGBoost beats LightGBM on pooled OOF log-loss with "
                "ECE within tolerance. NOTE: app/models/value_filter.py has no XGBoost "
                "loader branch — a loader is required before this artifact can serve."
            )
        else:
            print(
                f"\nCHALLENGER NOT ADOPTED (pre-registered rule: ll {best_xgb.result.log_loss:.5f} "
                f"vs {selected_draw.result.log_loss:.5f}, ece {best_xgb.result.ece:.4f} vs "
                f"{selected_draw.result.ece:.4f}+{ECE_TOLERANCE}) — reported as reference line."
            )
    print(f"\nSELECTED: {selected_kind} draw {selected_draw.index} on {selected_fs.name} features")
    print(f"  params: {json.dumps(selected_draw.params, sort_keys=True)}")

    # feature-lift verdict (an honest no-lift is valuable)
    lift = best_v1f.result.log_loss - best_v2.result.log_loss
    print(
        f"\nFEATURE-LIFT CHECK: best-of-sweep V1 features {best_v1f.result.log_loss:.5f} vs "
        f"V2 features {best_v2.result.log_loss:.5f} -> delta {lift:+.5f} "
        f"({'V2 features add OOF lift' if lift > 0 else 'NO OOF lift from v2 features'})"
    )

    # ---- operating point on OOF of the selected arm ---------------------------
    oof_seasons = [f.predict_season for f in folds]
    oof_df = train_cand[train_cand["season"].isin(oof_seasons)].copy()
    oof_df["p_cal"] = selected_draw.result.p_cal[oof_df.index]
    if not args.pool_cache.exists():
        raise SystemExit(f"pool cache missing: {args.pool_cache} (run the v1 trainer once)")
    pool = pd.read_parquet(args.pool_cache)
    for col in MM_KEY:
        pool[col] = pool[col].astype(str)
    pool_oof = pool[pool["season"].isin(oof_seasons)]
    assert not set(pool_oof["season"]) & set(SPENT_SEASONS)
    null_oof = tvf.select_bets(pool_oof, "edge", 0.0)
    print(f"\nTRAIN OOF operating-point sweep (null=thr0, B={args.n_boot_train}):")
    print(
        tvf.fmt_stats(tvf.compute_stats("null thr=0", null_oof, None, args.n_boot_train, rng_train))
    )
    points: list[Any] = []
    for q in tvf.Q_GRID:
        st = tvf.compute_stats(
            f"q>={q:.3f}",
            tvf.select_bets(oof_df, "p_cal", q),
            null_oof,
            args.n_boot_train,
            rng_train,
        )
        points.append(tvf.OperatingPoint(q, st))
        print(tvf.fmt_stats(st))
    chosen = tvf.choose_operating_point(points)
    if chosen is None:
        print(
            "\nNO TRAIN-QUALIFYING OPERATING POINT (n>=300 and incCLV_max-2SE>0) -> "
            "v2 REJECTED on train; fresh slice NOT consulted."
        )
    else:
        inc = chosen.stats.inc_max
        assert inc is not None
        print(
            f"\nFROZEN OPERATING POINT q*={chosen.q:.3f} "
            f"(train ROI {chosen.stats.roi * 100:+.2f}%, n={chosen.stats.n}, "
            f"incCLV_max {inc.point:+.4f}±{2 * inc.se:.4f})"
        )

    # ---- final model: fit 1920-2223, early-stop + isotonic on 2324 ------------
    tail = TRAIN_SEASONS[-1]
    fit_df = train_cand[train_cand["season"].isin(TRAIN_SEASONS[:-1]) & train_cand["y"].notna()]
    cal_df = train_cand[(train_cand["season"] == tail) & train_cand["y"].notna()]
    x_fit, cats = prepare_matrix_v2(fit_df, selected_fs)
    x_cal_final = prepare_matrix_v2(cal_df, selected_fs, cats)[0]
    final_fd = FoldData(
        fold=tvf.Fold(tuple(TRAIN_SEASONS[:-1]), tail, tail),
        x_fit=x_fit,
        y_fit=fit_df["y"].to_numpy(dtype=int),
        x_cal=x_cal_final,
        y_cal=cal_df["y"].to_numpy(dtype=int),
        x_prd=x_cal_final,
        prd_index=cal_df.index,
        cats=cats,
    )
    if selected_kind == "lgbm":
        final_model: Any = fit_lgbm_es(selected_draw.params, final_fd, selected_fs)
    else:
        final_model = fit_xgb_es(selected_draw.params, final_fd)
    final_best_iter = best_iteration_of(final_model)
    p_raw_cal = predict_raw(final_model, final_fd.x_cal)
    cal_kind, cal_obj = tvf.fit_calibrator(p_raw_cal, final_fd.y_cal)
    _, final_cal_ece = tvf.reliability_table(
        tvf.apply_calibrator((cal_kind, cal_obj), p_raw_cal), final_fd.y_cal.astype(float)
    )
    print(
        f"\nFINAL v2 MODEL: {selected_kind}, fit {TRAIN_SEASONS[:-1]}, early-stop+{cal_kind} "
        f"on {tail} (n={len(cal_df)}, best_iteration {final_best_iter}, "
        f"calib-season ECE {final_cal_ece:.4f} — mildly optimistic, see protocol caveat)"
    )
    print("\nFEATURE IMPORTANCES (final v2 model; [v2-NEW] marks dataset-v2 features):")
    importances = print_importances(final_model, list(final_fd.x_fit.columns))

    # ---- artifacts (v2 paths ONLY — never the deployed v1 filenames) ----------
    if selected_kind == "lgbm":
        model_path = args.out_dir / "value_filter_model_v2.txt"
        final_model.booster_.save_model(
            str(model_path), num_iteration=final_best_iter or N_ESTIMATORS_CAP
        )
    else:
        model_path = args.out_dir / "value_filter_model_v2.ubj"
        final_model.save_model(str(model_path))
    assert model_path.name != "value_filter_model.txt" and "_v2" in model_path.name

    calibrator_json: dict[str, Any]
    if cal_kind == "isotonic":
        calibrator_json = {
            "kind": "isotonic",
            "x_thresholds": cal_obj.X_thresholds_.tolist(),
            "y_thresholds": cal_obj.y_thresholds_.tolist(),
        }
    else:
        calibrator_json = {
            "kind": "platt",
            "coef": float(cal_obj.coef_[0][0]),
            "intercept": float(cal_obj.intercept_[0]),
        }

    def _arm_dict(
        name: str, r: OofResult, params: dict[str, Any] | None, draw: int | None
    ) -> dict[str, Any]:
        return {
            "arm": name,
            "log_loss": r.log_loss,
            "brier": r.brier,
            "ece": r.ece,
            "calib_ece": r.calib_ece,
            "best_iters": list(r.best_iters),
            "calib_kinds": list(r.calib_kinds),
            "params": params,
            "draw": draw,
        }

    arms_json = [
        _arm_dict("v1_grounding", res_ground, None, None),
        _arm_dict("v2_lgbm_v1feat", best_v1f.result, best_v1f.params, best_v1f.index),
        _arm_dict("v2_lgbm", best_v2.result, best_v2.params, best_v2.index),
    ]
    if best_xgb is not None:
        arms_json.append(_arm_dict("xgb_v2feat", best_xgb.result, best_xgb.params, best_xgb.index))

    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/ml/train_value_filter_v2.py",
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "leagues": list(LEAGUES_18),
        "train_seasons": list(TRAIN_SEASONS),
        "spent_seasons_never_loaded": list(SPENT_SEASONS),
        "min_odds": MIN_ODDS,
        "devig": "differential_margin_weighting (ADR-0006: same method fill+close)",
        "label": "clv_max > 0 (vig-free Max-of-books close)",
        "features": list(selected_fs.all),
        "features_cat": list(selected_fs.cat),
        "v2_new_features": list(V2_NEW_FEATURES),
        "seed": SEED,
        "sweep_seed": SWEEP_SEED,
        "sweep": {
            "n_draws_lgbm": len(lgbm_draws),
            "n_draws_xgb": len(xgb_draws) if best_xgb is not None else 0,
            "es_protocol": (
                "cap 2000 rounds; early_stopping(rounds=100, min_delta=1e-4) on the "
                "CALIBRATION season (double duty with isotonic — pre-registered option A)"
            ),
            "monotone": "edge:+1, method=advanced (LGBM) / dict (XGB, max_bin=512)",
            "log": str(sweep_path),
        },
        "model": {
            "name": f"{selected_kind}_{selected_fs.name}feat_sweep_draw{selected_draw.index}",
            "kind": selected_kind,
            "params": selected_draw.params,
            "fit_seasons": list(TRAIN_SEASONS[:-1]),
            "calib_season": tail,
            "calibration": cal_kind,
            "calib_n": int(len(cal_df)),
            "best_iteration": final_best_iter,
            "calib_season_ece": final_cal_ece,
        },
        "calibrator": calibrator_json,
        "oof_metrics": {
            "log_loss": selected_draw.result.log_loss,
            "brier": selected_draw.result.brier,
            "ece": selected_draw.result.ece,
            "n_oof": int((selected_draw.result.p_cal.notna() & train_cand["y"].notna()).sum()),
        },
        "oof_arms": arms_json,
        "feature_lift_v1_to_v2_logloss": lift,
        "challenger_rule": (
            f"xgb replaces lgbm iff OOF log-loss strictly lower AND ece <= lgbm ece + "
            f"{ECE_TOLERANCE}; result: {selected_kind}"
        ),
        "operating_point": None
        if chosen is None
        else {
            "q": chosen.q,
            "criterion": "max train-OOF ROI s.t. n>=300 and incCLV_max-2SE>0",
            "train_stats": tvf._stats_dict(chosen.stats),
        },
        "importances": importances,
        "fresh_one_shot": None,
        # NEVER "ADOPT" from this script: the binding verdict is live shadow CLV
        # + the fresh 2627 season. app/models/value_filter.py refuses to load
        # any manifest whose verdict != ADOPT, so v2 cannot serve accidentally.
        "verdict": "CANDIDATE (binding verdict: live shadow CLV + fresh 2627 season)",
    }

    # ---- ONE-SHOT fresh never-consulted slice ---------------------------------
    if not args.skip_fresh and chosen is not None:
        print("\n" + "=" * 78)
        print("ONE-SHOT FRESH SLICE — never-consulted divisions EC/SC1/SC2/SC3")
        print("(maxavg era, seasons exclude SPENT 2425/2526; transportability check,")
        print(" NOT an adoption gate — binding verdict is live shadow + 2627)")
        print("=" * 78)
        fresh = fresh_cand.copy()
        fresh_seasons = sorted(set(fresh["season"]))
        # Frozen vocab: fresh league levels are unseen -> NaN category ->
        # missing-bin routing, exactly how the live scorer treats unseen levels.
        x_fresh, _ = prepare_matrix_v2(fresh, selected_fs, cats)
        fresh["p_cal"] = tvf.apply_calibrator(
            (cal_kind, cal_obj), predict_raw(final_model, x_fresh)
        )
        mask_f = fresh["p_cal"].notna() & fresh["y"].notna()
        y_f = fresh.loc[mask_f, "y"].to_numpy(dtype=float)
        p_f = fresh.loc[mask_f, "p_cal"].to_numpy(dtype=float)
        ll_f = float(log_loss(y_f, p_f, labels=[0.0, 1.0]))
        br_f = float(brier_score_loss(y_f, p_f))
        lines_f, ece_f = tvf.reliability_table(p_f, y_f)
        print(
            f"fresh scored rows: {int(mask_f.sum())} labeled | log-loss {ll_f:.5f} | "
            f"brier {br_f:.5f} | ece {ece_f:.4f}"
        )
        for line in lines_f:
            print(line)
        pool_f = build_fresh_pool(args.cache_dir, FRESH_LEAGUES, fresh_seasons)
        null_f = tvf.select_bets(pool_f, "edge", 0.0)
        rows_f: dict[str, Any] = {}
        rows_f["null"] = tvf.compute_stats("null thr=0", null_f, None, args.n_boot_fresh, rng_fresh)
        rows_f["volume"] = tvf.compute_stats(
            "edge>=0.015",
            tvf.select_bets(pool_f, "edge", tvf.VOLUME_BASELINE_THR),
            null_f,
            args.n_boot_fresh,
            rng_fresh,
        )
        rows_f["premium"] = tvf.compute_stats(
            "edge>=0.03",
            tvf.select_bets(pool_f, "edge", tvf.PREMIUM_BASELINE_THR),
            null_f,
            args.n_boot_fresh,
            rng_fresh,
        )
        meta_sel = tvf.select_bets(fresh, "p_cal", chosen.q)
        rows_f["meta"] = tvf.compute_stats(
            f"META q>={chosen.q:.3f}", meta_sel, null_f, args.n_boot_fresh, rng_fresh
        )
        for s in rows_f.values():
            print(tvf.fmt_stats(s))
        if not meta_sel.empty:
            counts = meta_sel.groupby("league").size().to_dict()
            print(f"  per-division meta-bet counts: {counts}")
        manifest["fresh_one_shot"] = {
            "leagues": list(FRESH_LEAGUES),
            "seasons": fresh_seasons,
            "n_scored_labeled": int(mask_f.sum()),
            "log_loss": ll_f,
            "brier": br_f,
            "ece": ece_f,
            "stats": {k: tvf._stats_dict(v) for k, v in rows_f.items()},
            "note": (
                "one-shot transportability check on never-consulted divisions; "
                "league levels unseen by the model (missing-bin routing); xG features "
                "all-null outside big-5. NOT an adoption gate."
            ),
        }
    elif not args.skip_fresh:
        print("\nfresh slice NOT consulted (no train-qualifying operating point).")

    manifest_path = args.out_dir / "value_filter_manifest_v2.json"
    assert manifest_path.name != "value_filter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(
        f"\nartifacts: {model_path.name}, {manifest_path.name}, {sweep_path.name} -> {args.out_dir}"
    )
    print("Decision-support only — picks are informational; this system never places bets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
