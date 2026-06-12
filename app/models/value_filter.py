"""Value-filter META-MODEL inference (meta-labeling, verdict ADOPT 2026-06-12).

Doctrine (docs/backtesting/value-findings.md, .claude/memory): the edge is
sharp-vs-soft line shopping; ML winner-prediction from team stats backtested
NEGATIVE and is forbidden as a strategy. This module is the SECONDARY
classifier of a meta-labeling pair (Lopez de Prado): the deterministic value
pipeline generates candidates; this model scores P(candidate beats the
vig-free Max-of-books close) and, when VALUE_ML_FILTER is enabled, demotes
premium candidates scoring below the frozen operating point to the volume
(shadow) tier. It never predicts match outcomes and never places bets.

Held-out evidence (one-shot, seasons 2425+2526 — docs/research/
ml-value-filter.md): META q>=0.725 n=396, ROI +12.0%, incremental CLV vs the
Max close +0.0357 ± 0.0075 (2SE); all four pre-registered adoption criteria
passed. Trained by scripts/ml/train_value_filter.py; artifacts live OUTSIDE
git (data/ml/ is gitignored) and are loaded from a configurable directory.

Dependency policy: importing this module needs core deps only (numpy/stdlib).
lightgbm and pandas are imported lazily inside `ValueFilterModel.load`/
`score`; when they are absent the loader returns None and the pipeline runs
exactly as before — the `ml` extra stays optional.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.edge.value import CONSENSUS_ANCHOR
from app.probabilities.devig import DevigMethod, devig
from app.schemas.base import Market

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "value_filter_manifest.json"
MODEL_FILENAME = "value_filter_model.txt"

# Categorical model inputs (parity with scripts/ml/train_value_filter.py
# FEATURES_CAT). The manifest carries the full ordered feature list; this set
# only marks which of them must be pandas-categorical at predict time.
CATEGORICAL_FEATURES = frozenset({"league", "market", "selection_type"})

# Disagreement-spread feature methods — parity with
# scripts/ml/build_value_dataset.py SPREAD_METHODS.
_SPREAD_METHODS = (DevigMethod.MULTIPLICATIVE, DevigMethod.SHIN, DevigMethod.POWER)

# The model was trained on football-data.co.uk league codes (18 leagues).
# Live league labels are oddsportal slugs (config) or scraped display names
# (EventDirectory). Best-effort normalization: keys are lowercased with
# non-alphanumerics stripped. UNMAPPED leagues are out of scope -> unscored
# (None), never guessed — the trained vocabulary does not transfer.
_LEAGUE_CODES: dict[str, str] = {
    # oddsportal slugs (app/config.py / register_extra_leagues)
    "englandpremierleague": "E0",
    "englandchampionship": "E1",
    "englandleagueone": "E2",
    "englandleaguetwo": "E3",
    "scotlandpremiership": "SC0",
    "germanybundesliga": "D1",
    "germany2bundesliga": "D2",
    "italyseriea": "I1",
    "italyserieb": "I2",
    "spainlaliga": "SP1",
    "spainlaliga2": "SP2",
    "franceligue1": "F1",
    "franceligue2": "F2",
    "netherlandseredivisie": "N1",
    "belgiumjupilerproleague": "B1",
    "portugalligaportugal": "P1",
    "turkeysuperlig": "T1",
    "greecesuperleague": "G1",
    # oddsportal scraped display names (league_name)
    "premierleague": "E0",
    "championship": "E1",
    "leagueone": "E2",
    "leaguetwo": "E3",
    "premiership": "SC0",
    "bundesliga": "D1",
    "2bundesliga": "D2",
    "seriea": "I1",
    "serieb": "I2",
    "laliga": "SP1",
    "laliga2": "SP2",
    "ligue1": "F1",
    "ligue2": "F2",
    "eredivisie": "N1",
    "jupilerproleague": "B1",
    "ligaportugal": "P1",
    "superlig": "T1",
    "superleague": "G1",
}


def league_code(label: str) -> str | None:
    """football-data league code for a live league label, or None (unscored)."""
    key = "".join(ch for ch in label.lower() if ch.isalnum())
    return _LEAGUE_CODES.get(key)


def _season_end(d: date) -> date:
    """Nominal European season end — June 30 convention, parity with
    build_value_dataset._season_end (Jul-Dec -> next year's June 30)."""
    return date(d.year if d.month <= 6 else d.year + 1, 6, 30)


def calibrate(calibrator: Mapping[str, Any], p_raw: np.ndarray) -> np.ndarray:
    """Apply the manifest's calibrator — clean-room replay of the trainer's
    apply_calibrator using the serialized parameters only (no sklearn).

    isotonic: piecewise-linear interpolation over (x_thresholds,
    y_thresholds); np.interp clamps to the end values, equivalent to
    IsotonicRegression(out_of_bounds="clip"). platt: sigmoid(coef*p + b).
    """
    kind = calibrator.get("kind")
    if kind == "isotonic":
        x = np.asarray(calibrator["x_thresholds"], dtype=float)
        y = np.asarray(calibrator["y_thresholds"], dtype=float)
        p = np.interp(p_raw, x, y)
    elif kind == "platt":
        z = float(calibrator["coef"]) * p_raw + float(calibrator["intercept"])
        p = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"unknown calibrator kind: {kind!r}")
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _norm_book(s: str) -> str:
    return s.strip().lower()


def _market_key(market: Market, market_detail: str | None, n_selections: int) -> str | None:
    """Trained market key, or None when out of the model's scope.

    The model saw exactly two markets: football 1x2 (3-way h2h) and
    over/under 2.5 goals. Everything else (btts, dnb, handicaps, other
    totals lines, 2-way basketball h2h) is unscored by design.
    """
    if market is Market.H2H and n_selections == 3:
        return "1x2"
    if market is Market.TOTALS and market_detail == "over_under_2_5":
        return "ou25"
    return None


def _selection_type(key: str, selection: str, fair_by_sel: Mapping[str, float]) -> str:
    """fav/draw/dog by fair-prob rank — parity with the dataset builder."""
    fair = fair_by_sel[selection]
    if key == "1x2":
        if selection == "Draw":
            return "draw"
        others = [p for s, p in fair_by_sel.items() if s not in (selection, "Draw")]
        return "fav" if not others or fair >= max(others) else "dog"
    return "fav" if fair >= max(fair_by_sel.values()) else "dog"


def live_features(
    *,
    market: Market,
    market_detail: str | None,
    selection: str,
    prices: Mapping[str, Mapping[str, float]],
    fair_by_sel: Mapping[str, float],
    anchor_book: str,
    league: str,
    kickoff_utc: datetime | None,
    now: datetime,
    min_odds: float,
) -> dict[str, Any] | None:
    """Model features for one live value candidate, or None when out of scope.

    Scope gates (each mirrors a property of the TRAINING distribution; an
    out-of-scope candidate passes through the pipeline UNSCORED — the filter
    must never veto markets the model has not seen):
      - market must map to a trained key (1x2 / ou25);
      - the anchor must be a NAMED sharp book (the trained fair side is
        devig(Pinnacle pre-match)); consensus-median anchors are unscored;
      - the league must map to a trained football-data code;
      - the candidate's best price must clear the manifest's odds floor.

    Feature semantics replay build_value_dataset.candidates_from_rows on the
    live snapshot: per-selection best = max raw price across ALL books (the
    dataset's Max columns include Pinnacle); edge = fair - 1/best (raw — the
    live pipeline's commission-netted edge is the PICK gate, not the model
    input). day_of_week/days_to_season_end use the UTC kickoff date (the
    dataset used UK-local dates; <=1h boundary drift, never a leak).
    """
    key = _market_key(market, market_detail, len(fair_by_sel))
    if key is None:
        return None
    if anchor_book == CONSENSUS_ANCHOR:
        return None
    code = league_code(league)
    if code is None:
        return None
    if selection not in fair_by_sel:
        return None

    selections = list(fair_by_sel)
    anchor_norm = _norm_book(anchor_book)
    anchor_odds: list[float] = []
    best_odds: list[float] = []
    full_market_books: set[str] | None = None
    for sel in selections:
        book_odds = prices.get(sel) or {}
        a = next((o for b, o in book_odds.items() if _norm_book(b) == anchor_norm), None)
        valid = {b: o for b, o in book_odds.items() if o > 1.0}
        if a is None or not valid:
            return None  # anchor or market side missing — cannot price features
        anchor_odds.append(a)
        best_odds.append(max(valid.values()))
        names = {_norm_book(b) for b in valid}
        full_market_books = names if full_market_books is None else full_market_books & names

    i = selections.index(selection)
    if best_odds[i] < min_odds:
        return None

    edges = [fair_by_sel[s] - 1.0 / best_odds[j] for j, s in enumerate(selections)]
    spreads = [devig(anchor_odds, method=m) for m in _SPREAD_METHODS]
    devig_spread = max(abs(a[i] - b[i]) for a in spreads for b in spreads)

    d = (kickoff_utc or now).astimezone(UTC).date()
    return {
        "edge": edges[i],
        "fair_prob": fair_by_sel[selection],
        "best_price": best_odds[i],
        "pinn_price": anchor_odds[i],
        "overround_pinn": sum(1.0 / o for o in anchor_odds) - 1.0,
        "overround_best": sum(1.0 / o for o in best_odds) - 1.0,
        "devig_spread": devig_spread,
        "book_count": len(full_market_books or set()),
        "day_of_week": d.weekday(),
        "days_to_season_end": (_season_end(d) - d).days,
        "is_argmax_edge": edges[i] >= max(edges),
        "league": code,
        "market": key,
        "selection_type": _selection_type(key, selection, fair_by_sel),
    }


@dataclass(frozen=True)
class ValueFilterModel:
    """Loaded meta-model: LightGBM booster + manifest calibrator/threshold."""

    booster: Any  # lightgbm.Booster (typed Any — lightgbm is optional)
    features: tuple[str, ...]
    calibrator: dict[str, Any]
    threshold: float  # frozen operating point q* (manifest, train-only choice)
    min_odds: float  # training odds floor — candidates below it are unscored
    manifest_created_utc: str

    @classmethod
    def load(cls, model_dir: Path) -> ValueFilterModel | None:
        """Load artifacts from `model_dir`, or None (logged) when unavailable.

        Refuses anything but a manifest whose ONE-SHOT held-out verdict is
        ADOPT with a frozen operating point: an unvalidated artifact must
        never be able to demote live picks. All failure modes are non-fatal
        — the value pipeline simply runs unfiltered, as it always has.
        """
        manifest_path = model_dir / MANIFEST_FILENAME
        model_path = model_dir / MODEL_FILENAME
        if not manifest_path.exists() or not model_path.exists():
            logger.info("value-filter artifacts not found in %s; scoring disabled", model_dir)
            return None
        try:
            manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("value-filter manifest unreadable: %s", type(exc).__name__)
            return None
        if manifest.get("verdict") != "ADOPT":
            logger.warning(
                "value-filter manifest verdict is %r (not ADOPT); scoring disabled",
                manifest.get("verdict"),
            )
            return None
        threshold = (manifest.get("operating_point") or {}).get("q")
        features = manifest.get("features")
        calibrator = manifest.get("calibrator")
        kind = (manifest.get("model") or {}).get("kind")
        if threshold is None or not features or not calibrator:
            logger.warning("value-filter manifest incomplete; scoring disabled")
            return None
        if kind != "lgbm":
            logger.warning("value-filter model kind %r unsupported; scoring disabled", kind)
            return None
        try:
            lgb = importlib.import_module("lightgbm")
        except ImportError:
            logger.warning(
                "value-filter artifacts present but lightgbm is not installed "
                "(uv sync --extra ml); scoring disabled"
            )
            return None
        try:
            booster = lgb.Booster(model_file=str(model_path))
        except Exception as exc:  # lightgbm raises bare Exception on bad files
            logger.warning("value-filter model unreadable: %s", type(exc).__name__)
            return None
        loaded = cls(
            booster=booster,
            features=tuple(features),
            calibrator=dict(calibrator),
            threshold=float(threshold),
            min_odds=float(manifest.get("min_odds", 1.6)),
            manifest_created_utc=str(manifest.get("created_utc", "")),
        )
        logger.info(
            "value-filter meta-model loaded (manifest %s, q*=%.3f, %d features)",
            loaded.manifest_created_utc,
            loaded.threshold,
            len(loaded.features),
        )
        return loaded

    def score(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        """Calibrated P(candidate beats the vig-free Max close) per row.

        Rows come from `live_features`. Categorical levels unseen at training
        become NaN inside LightGBM (the booster file carries the training
        pandas-categorical vocabulary) and follow the learned default branch.
        """
        if not rows:
            return []
        pd = importlib.import_module("pandas")  # lightgbm's hard dependency chain
        x = pd.DataFrame([{f: r[f] for f in self.features} for r in rows])
        if "is_argmax_edge" in x.columns:
            x["is_argmax_edge"] = x["is_argmax_edge"].astype("int8")
        for col in CATEGORICAL_FEATURES & set(x.columns):
            x[col] = x[col].astype("category")
        raw = np.asarray(self.booster.predict(x), dtype=float)
        return [float(p) for p in calibrate(self.calibrator, raw)]
