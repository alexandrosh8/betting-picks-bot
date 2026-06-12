"""Build the leakage-safe VALUE-CANDIDATE training dataset (one parquet).

Doctrine (docs/backtesting/value-findings.md, .claude/memory): the edge is
sharp-vs-soft line shopping. ML here means META-MODELING the value signal —
learning which candidates (best pre-match price beats devigged-Pinnacle
pre-match fair) deliver positive held-out CLV/ROI. This script only ASSEMBLES
the candidate pool; it never selects, never tunes, never touches a holdout.

One output row per (match, market in {1x2, ou25}, selection) where
    edge = p_fair_sharp[i] - 1/best_price[i] >= --min-edge (default 0.005).

Column eras (verified inventory, football-data.co.uk mmz4281 files):
  - PSH/PSD/PSA (Pinnacle pre-match) exist from 2012/13 -> seasons 1213+.
  - Best pre-match price: Max*  (2019/20+, "maxavg" era) or BbMx*
    (BetBrain consensus, 2005/06-2018/19, "betbrain" era). The two pools
    differ -> the `era` regime flag is a SIGNAL feature.
  - ou25 needs Pinnacle pre-match OU (P>2.5/P<2.5), which exists 1920+ only;
    no ou25 candidates exist in the betbrain era.
  - CLV labels: devig(PSC*) from 1213; devig(MaxC*) from 1920 only ->
    clv_max is null in the betbrain era.
  - PSH/B365H/BbMx* are football-data's collection-time snapshot (~T-1 day),
    NOT a true opener. They are signal-time-safe; `open_to_signal_drift`
    is therefore always null for this source (no opening column exists) and
    is kept only so future sources (NBA open_spread, odds_snapshots) share
    the schema.

Leakage contract: every column is classified ID / SIGNAL / LABEL in SCHEMA
below. Close-derived and outcome-derived columns are LABELs; a LABEL column
appearing among features is a build-breaking defect (assert_no_label_leak,
locked by tests/test_ml_dataset.py). ADR-0006: fill-side and close-side fair
probabilities use the SAME devig method (DIFFERENTIAL_MARGIN, the production
v4 default).

v2 EXTENSIONS (2026-06-12) — additive SIGNAL features + fresh-domain tagging:
  - Rolling pre-match form: shift(1)-equivalent per-team rolling means of
    PAST matches' goals/shots/SOT/corners (raw per-match stats are outcomes
    = LABEL class; only strictly-prior aggregates are SIGNAL). Computed from
    every match in the league-season file, not just candidate matches.
  - Favourite-longshot-bias features: odds_band buckets + fair_prob_rank
    (Buchdahl: the FLB lives in how margin distributes across odds levels).
  - Anchor-disagreement v2: best-vs-Pinnacle price ratio / prob gap, plus
    per-method devig deltas vs the canonical method (extends devig_spread).
  - Understat xG rolling features (big-5 leagues, 2014/15+): r5/r10 xG
    for/against + xG overperformance (goals - xG, mean-reversion signal).
    Nullable everywhere else — LightGBM handles NaN natively. Team-name
    resolution is the explicit UNDERSTAT_TO_FOOTBALL_DATA table below;
    unmatched names quarantine with a log line, never fuzzy-merged.
  - consulted_before tag (ID class): True iff the league is in the
    18-league universe consumed by every prior backtest/training run
    (docs/research/ml-value-filter.md section 2; scripts/value_backtest.py
    default; trainer LEAGUES_18). Fresh never-consulted divisions: EC + SC1
    (assembled into v1's parquet but excluded from ALL evaluation by the
    trainer's LEAGUES_18 filter) and SC2 + SC3 (never downloaded before).
    The football-data "new leagues" feed (BRA/ARG/...) is excluded: it
    carries closing odds only — no signal-time price, protocol impossible.

SPENT-HOLDOUT DISCIPLINE (binding): seasons 2425+2526 are SPENT (consulted
4x). Any number computed on them is CONTAMINATED-REFERENCE, never a headline.
Development happens on nested season-blocked walk-forward within <=2324;
fresh evaluation only on never-consulted domains; binding verdict = live
shadow CLV + the one-shot fresh 2627 season.

Run (downloads are cached under data/ml/cache/ -- read-only GETs):

    uv run python scripts/ml/build_value_dataset.py                  # v2
    uv run python scripts/ml/build_value_dataset.py --v1             # v1 repro
    uv run python scripts/ml/build_value_dataset.py --no-understat   # offline v2
    uv run python scripts/ml/build_value_dataset.py --leagues E0,D1 --seasons 2425

Output: data/ml/value_candidates_v2.parquet (v2 default) or
data/ml/value_candidates.parquet (--v1, identical column set + row content to
the original builder). data/ is gitignored; raw artifacts are never committed.

Decision-support only — nothing here places bets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import io
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.backtesting.clv import clv_log
from app.ingestion.football_data import LEAGUES, fetch_season_csv
from app.probabilities.devig import DevigMethod, devig

UK_TZ = ZoneInfo("Europe/London")  # football-data Date/Time are UK local

# Production v4 premium-tier default (app/config.py); ADR-0006 demands the
# same method on the fill side (fair_prob/edge) and the close side (labels).
CANONICAL_DEVIG = DevigMethod.DIFFERENTIAL_MARGIN
# Disagreement-spread feature: structurally different methods only.
SPREAD_METHODS = (DevigMethod.MULTIPLICATIVE, DevigMethod.SHIN, DevigMethod.POWER)

MIN_EDGE_DEFAULT = 0.005  # wide pool — the harness evaluates filters on top

# 1213..2526: Pinnacle pre-match (PSH) exists from 2012/13 per the verified
# column-era inventory. Earlier seasons have no sharp anchor -> not viable.
SEASONS_ALL = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2012, 2026)]

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT_V1 = REPO_ROOT / "data" / "ml" / "value_candidates.parquet"
OUT_DEFAULT_V2 = REPO_ROOT / "data" / "ml" / "value_candidates_v2.parquet"
OUT_DEFAULT = OUT_DEFAULT_V1  # back-compat alias (v1 artifact path)
CACHE_DEFAULT = REPO_ROOT / "data" / "ml" / "cache"

# Frozen v1 league universe (the LEAGUES dict at v1 build time). --v1 must
# reproduce the original artifact, so this list never grows.
V1_LEAGUES: tuple[str, ...] = (
    "E0",
    "E1",
    "E2",
    "E3",
    "EC",
    "SC0",
    "SC1",
    "D1",
    "D2",
    "I1",
    "I2",
    "SP1",
    "SP2",
    "F1",
    "F2",
    "N1",
    "B1",
    "P1",
    "T1",
    "G1",
)

# The 18-league universe consumed by EVERY prior evaluation (value_backtest
# default + trainer LEAGUES_18, docs/research/ml-value-filter.md section 2).
# Rows outside this set are never-consulted fresh domains (EC/SC1 were
# assembled into the v1 parquet but filtered out of all training/eval;
# SC2/SC3 were never downloaded at all).
CONSULTED_LEAGUES: frozenset[str] = frozenset(
    {
        "E0",
        "E1",
        "E2",
        "E3",
        "SC0",
        "D1",
        "D2",
        "I1",
        "I2",
        "SP1",
        "SP2",
        "F1",
        "F2",
        "N1",
        "B1",
        "P1",
        "T1",
        "G1",
    }
)

# ---------------------------------------------------------------------------
# Understat xG enrichment (big-5 top flights; Understat coverage 2014/15+).
# penaltyblog.scrapers.Understat verified locally 2026-06-11 (memory note).
# ---------------------------------------------------------------------------
UNDERSTAT_COMP: dict[str, str] = {
    "E0": "ENG Premier League",
    "SP1": "ESP La Liga",
    "D1": "DEU Bundesliga 1",
    "I1": "ITA Serie A",
    "F1": "FRA Ligue 1",
}
UNDERSTAT_FIRST_END_YEAR = 15  # season "1415" is the first with Understat xG

# Versioned entity-resolution table: Understat team name -> football-data
# team name. Derived 2026-06-12 by diffing the complete distinct-name sets of
# both sources over seasons 1415-2526 (cached files); identity matches are
# omitted (the join falls back to the raw name). Unmatched names QUARANTINE
# with a log line — no fuzzy auto-merge in the hot path (ingestion rule).
UNDERSTAT_TO_FOOTBALL_DATA: dict[str, str] = {
    # E0
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    # SP1
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "La Coruna",
    "Espanyol": "Espanol",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    "Sporting Gijon": "Sp Gijon",
    # D1
    "Arminia Bielefeld": "Bielefeld",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "FC Heidenheim": "Heidenheim",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Greuther Fuerth": "Greuther Furth",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha Berlin": "Hertha",
    "Mainz 05": "Mainz",
    "Nuernberg": "Nurnberg",
    "RasenBallsport Leipzig": "RB Leipzig",
    "St. Pauli": "St Pauli",
    "VfB Stuttgart": "Stuttgart",
    # I1
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",
    "SPAL 2013": "Spal",
    # F1
    "Clermont Foot": "Clermont",
    "GFC Ajaccio": "Ajaccio GFCO",
    "Paris Saint Germain": "Paris SG",
    "SC Bastia": "Bastia",
    "Saint-Etienne": "St Etienne",
}

# Per-book home-price columns used as a book-count proxy in the maxavg era
# (the betbrain era carries an explicit Bb1X2 book count).
BOOK_HOME_COLS = (
    "B365H",
    "BWH",
    "IWH",
    "PSH",
    "WHH",
    "VCH",
    "LBH",
    "SBH",
    "GBH",
    "SJH",
    "BSH",
    "SOH",
    "1XBH",
    "BFH",
    "BFEH",
    "CLH",
)


def _f(x: object) -> float | None:
    """Odds parser — parity with scripts/value_backtest.py::_f (rejects <=1.0)."""
    try:
        v = float(str(x))
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


# --------------------------------------------------------------------------
# Market specs — column maps verified against the data inventory.
# --------------------------------------------------------------------------
def _won_1x2(r: dict[str, str], i: int) -> bool | None:
    ftr = r.get("FTR")
    if ftr not in ("H", "D", "A"):
        return None
    return ftr == ("H", "D", "A")[i]


def _won_ou25(r: dict[str, str], i: int) -> bool | None:
    try:
        goals = int(r["FTHG"]) + int(r["FTAG"])
    except (KeyError, TypeError, ValueError):
        return None
    return (goals >= 3) if i == 0 else (goals <= 2)


@dataclass(frozen=True)
class MarketSpec:
    pinn_cols: tuple[str, ...]  # SIGNAL: Pinnacle pre-match
    best_by_era: tuple[tuple[str, tuple[str, ...]], ...]  # SIGNAL, first match wins
    pinn_close_cols: tuple[str, ...]  # LABEL: Pinnacle close
    max_close_cols: tuple[str, ...]  # LABEL: Max-of-books close (maxavg era only)
    selections: tuple[str, ...]
    won_fn: Callable[[dict[str, str], int], bool | None]


MARKETS: dict[str, MarketSpec] = {
    "1x2": MarketSpec(
        pinn_cols=("PSH", "PSD", "PSA"),
        best_by_era=(
            ("maxavg", ("MaxH", "MaxD", "MaxA")),
            ("betbrain", ("BbMxH", "BbMxD", "BbMxA")),
        ),
        pinn_close_cols=("PSCH", "PSCD", "PSCA"),
        max_close_cols=("MaxCH", "MaxCD", "MaxCA"),
        selections=("H", "D", "A"),
        won_fn=_won_1x2,
    ),
    "ou25": MarketSpec(
        pinn_cols=("P>2.5", "P<2.5"),
        best_by_era=(("maxavg", ("Max>2.5", "Max<2.5")),),  # no betbrain Pinnacle OU
        pinn_close_cols=("PC>2.5", "PC<2.5"),
        max_close_cols=("MaxC>2.5", "MaxC<2.5"),
        selections=("over", "under"),
        won_fn=_won_ou25,
    ),
}

# --------------------------------------------------------------------------
# Leakage contract: ID / SIGNAL / LABEL classification for every column.
# SIGNAL = available at pick time (pre-match snapshot). LABEL = close- or
# outcome-derived; references only, NEVER features.
# --------------------------------------------------------------------------
# Rolling-form features (v2): shift(1)-equivalent per-team rolling means of
# strictly-prior matches in the same league-season. Goals get r5+r10; the
# stats columns (eras vary by division: 1213+ in E0..SC0, 1718+ elsewhere)
# get r5. Nullable: first match of a season and stat-less files yield None.
_FORM_FEATS: tuple[str, ...] = (
    "gf_r5",
    "ga_r5",
    "gf_r10",
    "ga_r10",
    "shots_for_r5",
    "shots_against_r5",
    "sot_for_r5",
    "sot_against_r5",
    "corners_for_r5",
    "corners_against_r5",
)
# Understat xG rolling features (v2): big-5 leagues 1415+ only, else null.
_XG_FEATS: tuple[str, ...] = (
    "xg_for_r5",
    "xg_against_r5",
    "xg_for_r10",
    "xg_against_r10",
    "xg_overperf_r10",  # mean(goals_for - xg_for) last 10 — mean-reversion signal
)

SCHEMA: dict[str, str] = {
    # identifiers (join keys, split keys — not model features, not labels)
    "league": "ID",
    "season": "ID",
    "match_date": "ID",  # UK-local match date (football-data join key)
    "kickoff_utc": "ID",  # Date+Time (UK local) -> UTC; null pre-2019/20 (no Time col)
    "home_team": "ID",
    "away_team": "ID",
    "market": "ID",
    "selection": "ID",
    "consulted_before": "ID",  # v2 split key: league in CONSULTED_LEAGUES (18)
    # SIGNAL features — pre-match only
    "era": "SIGNAL",  # best-price pool regime: maxavg (1920+) vs betbrain (1213-1819)
    "fair_prob": "SIGNAL",  # devig(Pinnacle pre-match)[i], CANONICAL_DEVIG
    "edge": "SIGNAL",  # fair_prob - 1/best_price
    "best_price": "SIGNAL",  # odds level: best pre-match price for the selection
    "pinn_price": "SIGNAL",  # Pinnacle pre-match price for the selection
    "overround_pinn": "SIGNAL",  # sum(1/PS*) - 1
    "overround_best": "SIGNAL",  # sum(1/best*) - 1 (can be < 0: composite line)
    "devig_spread": "SIGNAL",  # max pairwise |fair_i| diff across SPREAD_METHODS
    "open_to_signal_drift": "SIGNAL",  # always null here: football-data has no opener
    "book_count": "SIGNAL",  # Bb1X2 (betbrain) | count of per-book 1x2 cols (maxavg)
    "selection_type": "SIGNAL",  # fav/draw/dog by fair-prob rank within the market
    "day_of_week": "SIGNAL",  # 0=Mon .. 6=Sun
    "days_to_season_end": "SIGNAL",  # vs nominal June 30 (fixed, no schedule leak)
    "is_argmax_edge": "SIGNAL",  # highest-edge selection within (match, market)
    # SIGNAL v2 — favourite-longshot-bias structure (odds level, pre-match)
    "odds_band": "SIGNAL",  # best_price bucket: (1,1.5] (1.5,2] (2,3] (3,5] (5,10] (10,inf)
    "fair_prob_rank": "SIGNAL",  # 1 = highest fair_prob within (match, market)
    # SIGNAL v2 — anchor disagreement (Pinnacle vs best, devig-method spread)
    "price_ratio_best_pinn": "SIGNAL",  # best_price / pinn_price
    "prob_gap_pinn_best": "SIGNAL",  # 1/pinn_price - 1/best_price (vig-in prob gap)
    "fair_mult_delta": "SIGNAL",  # devig(multiplicative)[i] - fair_prob
    "fair_shin_delta": "SIGNAL",  # devig(shin)[i] - fair_prob
    "fair_power_delta": "SIGNAL",  # devig(power)[i] - fair_prob
    # SIGNAL v2 — rolling pre-match form (nullable; strictly-prior matches only)
    **{f"home_{f}": "SIGNAL" for f in _FORM_FEATS},
    **{f"away_{f}": "SIGNAL" for f in _FORM_FEATS},
    # SIGNAL v2 — Understat xG rolling (nullable; big-5 1415+ only)
    **{f"home_{f}": "SIGNAL" for f in _XG_FEATS},
    **{f"away_{f}": "SIGNAL" for f in _XG_FEATS},
    # LABELS — close-derived / outcome-derived; never features
    "won": "LABEL",
    "bet_odds": "LABEL",  # settlement-side copy of best_price (profit accounting)
    "pinn_close_fair": "LABEL",  # devig(PSC*)[i], same method (ADR-0006)
    "max_close_fair": "LABEL",  # devig(MaxC*)[i]; null in betbrain era
    "clv_pinn": "LABEL",  # ln(bet_odds * pinn_close_fair)
    "clv_max": "LABEL",  # ln(bet_odds * max_close_fair) — stricter reference
    "profit_units": "LABEL",  # flat 1u: odds-1 if won else -1
}
FEATURE_COLUMNS: tuple[str, ...] = tuple(c for c, k in SCHEMA.items() if k == "SIGNAL")
LABEL_COLUMNS: tuple[str, ...] = tuple(c for c, k in SCHEMA.items() if k == "LABEL")
ID_COLUMNS: tuple[str, ...] = tuple(c for c, k in SCHEMA.items() if k == "ID")

# Exact v1 column set + order — the --v1 output and the original artifact.
SCHEMA_V1: tuple[str, ...] = (
    "league",
    "season",
    "match_date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "market",
    "selection",
    "era",
    "fair_prob",
    "edge",
    "best_price",
    "pinn_price",
    "overround_pinn",
    "overround_best",
    "devig_spread",
    "open_to_signal_drift",
    "book_count",
    "selection_type",
    "day_of_week",
    "days_to_season_end",
    "is_argmax_edge",
    "won",
    "bet_odds",
    "pinn_close_fair",
    "max_close_fair",
    "clv_pinn",
    "clv_max",
    "profit_units",
)


@dataclass(frozen=True)
class Candidate:
    league: str
    season: str
    match_date: date
    kickoff_utc: datetime | None
    home_team: str
    away_team: str
    market: str
    selection: str
    era: str
    fair_prob: float
    edge: float
    best_price: float
    pinn_price: float
    overround_pinn: float
    overround_best: float
    devig_spread: float
    open_to_signal_drift: float | None
    book_count: int
    selection_type: str
    day_of_week: int
    days_to_season_end: int
    is_argmax_edge: bool
    # v2 — fresh-domain split key
    consulted_before: bool
    # v2 — FLB structure
    odds_band: str
    fair_prob_rank: int
    # v2 — anchor disagreement
    price_ratio_best_pinn: float
    prob_gap_pinn_best: float
    fair_mult_delta: float
    fair_shin_delta: float
    fair_power_delta: float
    # v2 — rolling pre-match form (strictly-prior matches; nullable)
    home_gf_r5: float | None
    home_ga_r5: float | None
    home_gf_r10: float | None
    home_ga_r10: float | None
    home_shots_for_r5: float | None
    home_shots_against_r5: float | None
    home_sot_for_r5: float | None
    home_sot_against_r5: float | None
    home_corners_for_r5: float | None
    home_corners_against_r5: float | None
    away_gf_r5: float | None
    away_ga_r5: float | None
    away_gf_r10: float | None
    away_ga_r10: float | None
    away_shots_for_r5: float | None
    away_shots_against_r5: float | None
    away_sot_for_r5: float | None
    away_sot_against_r5: float | None
    away_corners_for_r5: float | None
    away_corners_against_r5: float | None
    # v2 — Understat xG rolling (big-5 1415+; nullable elsewhere)
    home_xg_for_r5: float | None
    home_xg_against_r5: float | None
    home_xg_for_r10: float | None
    home_xg_against_r10: float | None
    home_xg_overperf_r10: float | None
    away_xg_for_r5: float | None
    away_xg_against_r5: float | None
    away_xg_for_r10: float | None
    away_xg_against_r10: float | None
    away_xg_overperf_r10: float | None
    won: bool
    bet_odds: float
    pinn_close_fair: float | None
    max_close_fair: float | None
    clv_pinn: float | None
    clv_max: float | None
    profit_units: float


def assert_no_label_leak() -> None:
    """Build-breaking leakage gate: LABELs must never enter the feature set."""
    feats, labels, ids = set(FEATURE_COLUMNS), set(LABEL_COLUMNS), set(ID_COLUMNS)
    assert not feats & labels, f"label columns leaked into features: {feats & labels}"
    assert not feats & ids and not labels & ids, "ID columns must stay out of both sets"
    close_or_outcome = {
        "won",
        "bet_odds",
        "pinn_close_fair",
        "max_close_fair",
        "clv_pinn",
        "clv_max",
        "profit_units",
    }
    assert close_or_outcome <= labels, "every close/outcome-derived column must be a LABEL"
    # no raw close column (PSC*/MaxC*/PC*) may appear in the schema at all
    raw_close = {c for s in MARKETS.values() for c in (*s.pinn_close_cols, *s.max_close_cols)}
    assert not raw_close & set(SCHEMA), "raw close columns must not be dataset columns"
    fields = {f.name for f in dataclasses.fields(Candidate)}
    assert fields == set(SCHEMA), f"SCHEMA out of sync with Candidate: {fields ^ set(SCHEMA)}"
    # v2 invariants ----------------------------------------------------------
    # v1 columns survive verbatim (column-set reproducibility under --v1)
    assert set(SCHEMA_V1) <= set(SCHEMA), "SCHEMA_V1 must be a subset of SCHEMA"
    assert {c: SCHEMA[c] for c in SCHEMA_V1} == {
        c: k for c, k in SCHEMA.items() if c in set(SCHEMA_V1)
    }, "v1 column classification must not change"
    # raw per-match stats are OUTCOMES: only rolling PRE-MATCH aggregates may
    # appear, and every rolling/xG column must be a strictly-prior aggregate
    raw_stats = {"FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"}
    assert not raw_stats & set(SCHEMA), "raw match-stat columns are LABEL-class, never columns"
    rolling = {f"{side}_{f}" for side in ("home", "away") for f in (*_FORM_FEATS, *_XG_FEATS)}
    assert rolling <= feats, "every rolling form/xG column must be classified SIGNAL"
    assert "consulted_before" in ids, "consulted_before is a split key (ID), not a feature"


# --------------------------------------------------------------------------
# Row -> candidates
# --------------------------------------------------------------------------
def _parse_match_date(raw: str | None) -> date | None:
    if not raw:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_kickoff_utc(d: date, time_raw: str | None) -> datetime | None:
    """Date + Time are UK local (inventory note); convert to UTC. Null when
    the Time column is absent (pre-2019/20)."""
    if not time_raw or not time_raw.strip():
        return None
    try:
        t = datetime.strptime(time_raw.strip(), "%H:%M").time()
    except ValueError:
        return None
    return datetime.combine(d, t, tzinfo=UK_TZ).astimezone(UTC)


def _season_end(season: str) -> date:
    """Nominal season end: June 30 of the end year. Fixed by convention —
    using the actual last fixture date would leak future schedule info."""
    return date(2000 + int(season[2:4]), 6, 30)


def _book_count(r: dict[str, str], era: str) -> int:
    if era == "betbrain":
        raw = (r.get("Bb1X2") or "").strip()
        try:
            return int(float(raw))
        except ValueError:
            pass  # fall through to the column-count proxy
    return sum(1 for c in BOOK_HOME_COLS if _f(r.get(c)) is not None)


def _selection_type(market: str, fair: Sequence[float], i: int) -> str:
    if market == "1x2" and i == 1:
        return "draw"
    if market == "1x2":
        other = 2 if i == 0 else 0
        return "fav" if fair[i] >= fair[other] else "dog"
    return "fav" if fair[i] >= max(fair) else "dog"


def _fairs_by_method(odds: Sequence[float]) -> dict[DevigMethod, Sequence[float]]:
    """Fair-prob vectors for each structurally different devig method."""
    return {m: devig(odds, method=m) for m in SPREAD_METHODS}


def _devig_spreads(fairs: Mapping[DevigMethod, Sequence[float]], n: int) -> list[float]:
    """Per-selection max pairwise fair-prob disagreement across SPREAD_METHODS."""
    vecs = list(fairs.values())
    return [max(abs(a[i] - b[i]) for a in vecs for b in vecs) for i in range(n)]


# --------------------------------------------------------------------------
# v2 features: FLB structure, rolling pre-match form, Understat xG
# --------------------------------------------------------------------------
_ODDS_BANDS: tuple[tuple[float, str], ...] = (
    (1.5, "(1.0,1.5]"),
    (2.0, "(1.5,2.0]"),
    (3.0, "(2.0,3.0]"),
    (5.0, "(3.0,5.0]"),
    (10.0, "(5.0,10.0]"),
)


def _odds_band(price: float) -> str:
    """FLB odds-level bucket of the takeable (best) price."""
    for upper, label in _ODDS_BANDS:
        if price <= upper:
            return label
    return "(10.0,inf)"


def _fair_prob_rank(fair: Sequence[float], i: int) -> int:
    """1 = highest fair prob within the market; ties break by index (stable)."""
    return 1 + sum(1 for j, p in enumerate(fair) if p > fair[i] or (p == fair[i] and j < i))


def _mean_last(values: Sequence[float], k: int) -> float | None:
    """Mean of the last k values — shift(1).rolling(k, min_periods=1).mean()
    equivalence holds because callers append the current match only AFTER
    reading features (strictly-prior window)."""
    if not values:
        return None
    window = values[-k:]
    return sum(window) / len(window)


def _stat(r: dict[str, str], col: str) -> float | None:
    """Non-negative count parser for stats columns (HS/HC/FTHG/...)."""
    raw = (r.get(col) or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


_EMPTY_FORM: dict[str, float | None] = dict.fromkeys(_FORM_FEATS)
_EMPTY_XG: dict[str, float | None] = dict.fromkeys(_XG_FEATS)

# (stat key, home column, away column) — home's "for" is the home column.
_FORM_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("goals", "FTHG", "FTAG"),
    ("shots", "HS", "AS"),
    ("sot", "HST", "AST"),
    ("corners", "HC", "AC"),
)


def _form_feats(state: Mapping[str, list[float]]) -> dict[str, float | None]:
    return {
        "gf_r5": _mean_last(state["goals_for"], 5),
        "ga_r5": _mean_last(state["goals_against"], 5),
        "gf_r10": _mean_last(state["goals_for"], 10),
        "ga_r10": _mean_last(state["goals_against"], 10),
        "shots_for_r5": _mean_last(state["shots_for"], 5),
        "shots_against_r5": _mean_last(state["shots_against"], 5),
        "sot_for_r5": _mean_last(state["sot_for"], 5),
        "sot_against_r5": _mean_last(state["sot_against"], 5),
        "corners_for_r5": _mean_last(state["corners_for"], 5),
        "corners_against_r5": _mean_last(state["corners_against"], 5),
    }


FormLookup = dict[tuple[str, str, date], tuple[dict[str, float | None], dict[str, float | None]]]


def _build_form_lookup(rows: list[dict[str, str]]) -> FormLookup:
    """(home, away, date) -> (home-side, away-side) rolling pre-match form.

    Built from EVERY parseable match in the league-season file in
    chronological order (stable sort keeps file order within a date).
    Features are read BEFORE the current match is appended — the explicit
    shift(1) guarantee (betting-feature-engineering skill, rule 1).
    """
    parsed: list[tuple[date, dict[str, str]]] = []
    for r in rows:
        if not r.get("HomeTeam") or not r.get("Date"):
            continue
        d = _parse_match_date(r.get("Date"))
        if d is not None:
            parsed.append((d, r))
    parsed.sort(key=lambda t: t[0])  # stable mergesort-like ordering in CPython
    state: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lookup: FormLookup = {}
    for d, r in parsed:
        home = r["HomeTeam"].strip()
        away = (r.get("AwayTeam") or "").strip()
        lookup[(home, away, d)] = (_form_feats(state[home]), _form_feats(state[away]))
        for stat, home_col, away_col in _FORM_SOURCES:
            hv, av = _stat(r, home_col), _stat(r, away_col)
            if hv is not None:
                state[home][f"{stat}_for"].append(hv)
                state[away][f"{stat}_against"].append(hv)
            if av is not None:
                state[home][f"{stat}_against"].append(av)
                state[away][f"{stat}_for"].append(av)
    return lookup


# xG lookup: (mapped home, mapped away) -> per-side feature dicts. Pairings
# are unique within a league-season, so no date join is needed (avoids
# cross-source timezone drift). Rolling order = Understat's own chronology.
XgLookup = dict[tuple[str, str], tuple[dict[str, float | None], dict[str, float | None]]]


def _understat_cache_path(cache_dir: Path, league: str, season: str) -> Path:
    return cache_dir / "understat" / f"{league}_{season}.csv"


def _fetch_understat_fixtures(
    league: str, season: str, cache_dir: Path
) -> list[dict[str, str]] | None:
    """Cached read-only GET of Understat results+xG; failures quarantine.

    Lazy penaltyblog import: penaltyblog.scrapers' package __init__ pulls in
    its FBRef module (tls_requests dependency) — we never call FBRef, and the
    import stays out of --v1 / --no-understat runs entirely.
    """
    path = _understat_cache_path(cache_dir, league, season)
    if not path.exists():
        try:
            from penaltyblog.scrapers.understat import Understat

            season_str = f"20{season[:2]}-20{season[2:]}"
            df = Understat(UNDERSTAT_COMP[league], season_str).get_fixtures().reset_index()
            cols = [
                "datetime",
                "team_home",
                "team_away",
                "goals_home",
                "goals_away",
                "xg_home",
                "xg_away",
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            df[cols].to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001 — never log URLs/exception text
            print(f"  quarantine understat {league} {season}: {type(exc).__name__}")
            return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _xg_feats(
    xg_for: Sequence[float], xg_against: Sequence[float], overperf: Sequence[float]
) -> dict[str, float | None]:
    return {
        "xg_for_r5": _mean_last(xg_for, 5),
        "xg_against_r5": _mean_last(xg_against, 5),
        "xg_for_r10": _mean_last(xg_for, 10),
        "xg_against_r10": _mean_last(xg_against, 10),
        "xg_overperf_r10": _mean_last(overperf, 10),
    }


def build_xg_lookup(fixtures: list[dict[str, str]]) -> XgLookup:
    """Strictly-prior rolling xG per team, keyed by mapped (home, away).

    State keys are raw Understat names (stable per team); only the JOIN key
    is passed through UNDERSTAT_TO_FOOTBALL_DATA (identity fallback). The
    current fixture is appended AFTER features are read — shift(1) again.
    """
    xg_for: dict[str, list[float]] = defaultdict(list)
    xg_against: dict[str, list[float]] = defaultdict(list)
    overperf: dict[str, list[float]] = defaultdict(list)
    lookup: XgLookup = {}
    ordered = sorted(fixtures, key=lambda f: (f.get("datetime") or "", f.get("team_home") or ""))
    for fx in ordered:
        h_us, a_us = (fx.get("team_home") or "").strip(), (fx.get("team_away") or "").strip()
        try:
            xgh, xga = float(fx["xg_home"]), float(fx["xg_away"])
            gh, ga = float(fx["goals_home"]), float(fx["goals_away"])
        except (KeyError, TypeError, ValueError):
            continue  # unresolved fixture -> no state update, no key
        if not h_us or not a_us:
            continue
        h_fd = UNDERSTAT_TO_FOOTBALL_DATA.get(h_us, h_us)
        a_fd = UNDERSTAT_TO_FOOTBALL_DATA.get(a_us, a_us)
        lookup[(h_fd, a_fd)] = (
            _xg_feats(xg_for[h_us], xg_against[h_us], overperf[h_us]),
            _xg_feats(xg_for[a_us], xg_against[a_us], overperf[a_us]),
        )
        xg_for[h_us].append(xgh)
        xg_against[h_us].append(xga)
        overperf[h_us].append(gh - xgh)
        xg_for[a_us].append(xga)
        xg_against[a_us].append(xgh)
        overperf[a_us].append(ga - xga)
    return lookup


def _quarantine_unmatched_names(
    league: str, season: str, rows: list[dict[str, str]], xg_lookup: XgLookup
) -> None:
    """Log mapped Understat names absent from the football-data file.

    Those fixtures simply never join (xG features stay null) — explicit
    quarantine, no fuzzy auto-merge in the hot path."""
    fd_names = {r["HomeTeam"].strip() for r in rows if r.get("HomeTeam")}
    fd_names |= {(r.get("AwayTeam") or "").strip() for r in rows if r.get("AwayTeam")}
    mapped = {name for key in xg_lookup for name in key}
    unmatched = sorted(mapped - fd_names)
    if unmatched:
        print(
            f"  quarantine understat names {league} {season}: {unmatched} "
            "(no fuzzy merge; xG features stay null for these joins)"
        )


def candidates_from_rows(
    league: str,
    season: str,
    rows: list[dict[str, str]],
    min_edge: float = MIN_EDGE_DEFAULT,
    xg_lookup: XgLookup | None = None,
) -> list[Candidate]:
    """All candidate selections (edge >= min_edge) for one league-season.

    Candidate rule parity with scripts/value_backtest.py::bets_for, except the
    pool keeps EVERY qualifying selection (the harness applies one-bet-per-
    match via is_argmax_edge and any threshold/odds filters downstream).
    v2 features are purely additive — they never gate row emission, so --v1
    row content is unchanged.
    """
    form_lookup = _build_form_lookup(rows)
    consulted = league in CONSULTED_LEAGUES
    out: list[Candidate] = []
    for r in rows:
        if not r.get("HomeTeam") or not r.get("Date"):
            continue
        d = _parse_match_date(r.get("Date"))
        if d is None:
            continue
        kickoff = _parse_kickoff_utc(d, r.get("Time"))
        home = r["HomeTeam"].strip()
        away = (r.get("AwayTeam") or "").strip()
        form_h, form_a = form_lookup.get((home, away, d), (_EMPTY_FORM, _EMPTY_FORM))
        xg_h, xg_a = (xg_lookup or {}).get((home, away), (_EMPTY_XG, _EMPTY_XG))
        for market, spec in MARKETS.items():
            ps_opt = [_f(r.get(c)) for c in spec.pinn_cols]
            if None in ps_opt:
                continue
            ps: list[float] = [v for v in ps_opt if v is not None]
            era = ""
            best: list[float] = []
            for era_name, cols in spec.best_by_era:
                bx = [_f(r.get(c)) for c in cols]
                if None not in bx:
                    era, best = era_name, [v for v in bx if v is not None]
                    break
            if not best:
                continue
            fair = devig(ps, method=CANONICAL_DEVIG)
            edges = [fair[i] - 1.0 / best[i] for i in range(len(ps))]
            argmax_i = max(range(len(edges)), key=lambda i: edges[i])
            psc = [_f(r.get(c)) for c in spec.pinn_close_cols]
            close_p = (
                devig([v for v in psc if v is not None], method=CANONICAL_DEVIG)
                if None not in psc
                else None
            )
            mxc = [_f(r.get(c)) for c in spec.max_close_cols] if era == "maxavg" else [None]
            close_m = (
                devig([v for v in mxc if v is not None], method=CANONICAL_DEVIG)
                if None not in mxc
                else None
            )
            overround_pinn = sum(1.0 / o for o in ps) - 1.0
            overround_best = sum(1.0 / o for o in best) - 1.0
            fairs = _fairs_by_method(ps)
            spreads = _devig_spreads(fairs, len(ps))
            fair_mult = fairs[DevigMethod.MULTIPLICATIVE]
            fair_shin = fairs[DevigMethod.SHIN]
            fair_power = fairs[DevigMethod.POWER]
            for i, edge in enumerate(edges):
                if edge < min_edge:
                    continue
                won = spec.won_fn(r, i)
                if won is None:
                    continue  # unresolvable result -> skipped (match counts printed)
                out.append(
                    Candidate(
                        league=league,
                        season=season,
                        match_date=d,
                        kickoff_utc=kickoff,
                        home_team=home,
                        away_team=away,
                        market=market,
                        selection=spec.selections[i],
                        era=era,
                        fair_prob=fair[i],
                        edge=edge,
                        best_price=best[i],
                        pinn_price=ps[i],
                        overround_pinn=overround_pinn,
                        overround_best=overround_best,
                        devig_spread=spreads[i],
                        open_to_signal_drift=None,
                        book_count=_book_count(r, era),
                        selection_type=_selection_type(market, fair, i),
                        day_of_week=d.weekday(),
                        days_to_season_end=(_season_end(season) - d).days,
                        is_argmax_edge=(i == argmax_i),
                        consulted_before=consulted,
                        odds_band=_odds_band(best[i]),
                        fair_prob_rank=_fair_prob_rank(fair, i),
                        price_ratio_best_pinn=best[i] / ps[i],
                        prob_gap_pinn_best=1.0 / ps[i] - 1.0 / best[i],
                        fair_mult_delta=fair_mult[i] - fair[i],
                        fair_shin_delta=fair_shin[i] - fair[i],
                        fair_power_delta=fair_power[i] - fair[i],
                        home_gf_r5=form_h["gf_r5"],
                        home_ga_r5=form_h["ga_r5"],
                        home_gf_r10=form_h["gf_r10"],
                        home_ga_r10=form_h["ga_r10"],
                        home_shots_for_r5=form_h["shots_for_r5"],
                        home_shots_against_r5=form_h["shots_against_r5"],
                        home_sot_for_r5=form_h["sot_for_r5"],
                        home_sot_against_r5=form_h["sot_against_r5"],
                        home_corners_for_r5=form_h["corners_for_r5"],
                        home_corners_against_r5=form_h["corners_against_r5"],
                        away_gf_r5=form_a["gf_r5"],
                        away_ga_r5=form_a["ga_r5"],
                        away_gf_r10=form_a["gf_r10"],
                        away_ga_r10=form_a["ga_r10"],
                        away_shots_for_r5=form_a["shots_for_r5"],
                        away_shots_against_r5=form_a["shots_against_r5"],
                        away_sot_for_r5=form_a["sot_for_r5"],
                        away_sot_against_r5=form_a["sot_against_r5"],
                        away_corners_for_r5=form_a["corners_for_r5"],
                        away_corners_against_r5=form_a["corners_against_r5"],
                        home_xg_for_r5=xg_h["xg_for_r5"],
                        home_xg_against_r5=xg_h["xg_against_r5"],
                        home_xg_for_r10=xg_h["xg_for_r10"],
                        home_xg_against_r10=xg_h["xg_against_r10"],
                        home_xg_overperf_r10=xg_h["xg_overperf_r10"],
                        away_xg_for_r5=xg_a["xg_for_r5"],
                        away_xg_against_r5=xg_a["xg_against_r5"],
                        away_xg_for_r10=xg_a["xg_for_r10"],
                        away_xg_against_r10=xg_a["xg_against_r10"],
                        away_xg_overperf_r10=xg_a["xg_overperf_r10"],
                        won=won,
                        bet_odds=best[i],
                        pinn_close_fair=close_p[i] if close_p else None,
                        max_close_fair=close_m[i] if close_m else None,
                        clv_pinn=clv_log(best[i], close_p[i]) if close_p else None,
                        clv_max=clv_log(best[i], close_m[i]) if close_m else None,
                        profit_units=(best[i] - 1.0) if won else -1.0,
                    )
                )
    return out


# --------------------------------------------------------------------------
# IO: cached, paced, read-only GETs (reuses the tenacity-wrapped fetcher)
# --------------------------------------------------------------------------
async def _load_csv(
    client: httpx.AsyncClient, cache_dir: Path, league: str, season: str
) -> list[dict[str, str]] | None:
    """Returns parsed rows, or None if the file is unavailable (quarantined)."""
    cache = cache_dir / f"{season}_{league}.csv"
    if cache.exists():
        text = cache.read_text(encoding="utf-8", errors="replace")
    else:
        text = ""
        for attempt in range(4):
            try:
                text = await fetch_season_csv(client, league, season)
                break
            except httpx.HTTPStatusError as exc:  # never log URLs/exception text
                status = exc.response.status_code
                if status == 404:
                    print(f"  quarantine {league} {season}: HTTP 404 (no such file)")
                    return None
                print(f"  retry {league} {season}: HTTP {status} (attempt {attempt + 1})")
                await asyncio.sleep(1.5)
            except httpx.HTTPError as exc:
                print(f"  retry {league} {season}: {type(exc).__name__} (attempt {attempt + 1})")
                await asyncio.sleep(1.5)
        if not text:
            print(f"  quarantine {league} {season}: fetch failed after 4 attempts")
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        await asyncio.sleep(0.3)  # pacing — respect football-data.co.uk
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))


def _to_dataframe(cands: list[Candidate], schema_version: int = 2) -> pd.DataFrame:
    df = pd.DataFrame([dataclasses.asdict(c) for c in cands], columns=list(SCHEMA))
    df = df.sort_values(
        ["league", "season", "match_date", "home_team", "away_team", "market", "selection"],
        kind="mergesort",  # stable -> deterministic output ordering
    ).reset_index(drop=True)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    nullable_v2 = tuple(
        f"{side}_{f}" for side in ("home", "away") for f in (*_FORM_FEATS, *_XG_FEATS)
    )
    cols = (
        "pinn_close_fair",
        "max_close_fair",
        "clv_pinn",
        "clv_max",
        "open_to_signal_drift",
        *nullable_v2,
    )
    for col in cols:
        df[col] = df[col].astype("float64")
    df["won"] = df["won"].astype(bool)
    df["is_argmax_edge"] = df["is_argmax_edge"].astype(bool)
    df["consulted_before"] = df["consulted_before"].astype(bool)
    df["book_count"] = df["book_count"].astype("int32")
    df["fair_prob_rank"] = df["fair_prob_rank"].astype("int32")
    if schema_version == 1:
        df = df[list(SCHEMA_V1)]
    return df


def _understat_eligible(league: str, season: str) -> bool:
    """Understat covers the big-5 top flights from 2014/15 onward."""
    return league in UNDERSTAT_COMP and int(season[:2]) >= UNDERSTAT_FIRST_END_YEAR - 1


async def build(
    leagues: list[str],
    seasons: list[str],
    cache_dir: Path,
    min_edge: float,
    schema_version: int = 2,
    use_understat: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str, int, int]]]:
    counts: list[tuple[str, str, int, int]] = []  # (league, season, matches, candidates)
    all_cands: list[Candidate] = []
    async with httpx.AsyncClient() as client:
        for league in leagues:
            for season in seasons:
                rows = await _load_csv(client, cache_dir, league, season)
                if rows is None:
                    counts.append((league, season, 0, 0))
                    continue
                xg_lookup: XgLookup | None = None
                if schema_version >= 2 and use_understat and _understat_eligible(league, season):
                    # blocking requests call -> worker thread (batch script,
                    # but keep the loop honest anyway)
                    fixtures = await asyncio.to_thread(
                        _fetch_understat_fixtures, league, season, cache_dir
                    )
                    if fixtures:
                        xg_lookup = build_xg_lookup(fixtures)
                        _quarantine_unmatched_names(league, season, rows, xg_lookup)
                n_matches = sum(1 for r in rows if r.get("HomeTeam") and r.get("Date"))
                cands = candidates_from_rows(league, season, rows, min_edge, xg_lookup)
                counts.append((league, season, n_matches, len(cands)))
                all_cands.extend(cands)
    return _to_dataframe(all_cands, schema_version), counts


def _quality_report(df: pd.DataFrame, counts: list[tuple[str, str, int, int]]) -> None:
    print("\nrow counts per league/season (matches -> candidate rows):")
    by_league: dict[str, tuple[int, int]] = {}
    for league, season, n_m, n_c in counts:
        m, c = by_league.get(league, (0, 0))
        by_league[league] = (m + n_m, c + n_c)
        print(f"  {league:>4} {season}: {n_m:5d} matches -> {n_c:5d} candidates")
    print("\nper-league totals:")
    for league, (m, c) in sorted(by_league.items()):
        print(f"  {league:>4}: {m:6d} matches -> {c:6d} candidates")
    print(f"\nTOTAL: {sum(m for m, _ in by_league.values())} matches -> {len(df)} candidate rows")
    if df.empty:
        print("DATA-QUALITY ALERT: zero candidates assembled")
        return
    print("\nper-market / per-era candidate counts:")
    print(df.groupby(["market", "era"], observed=True).size().to_string())
    print("\nlabel availability (data-quality gates):")
    n = len(df)
    for col, expect in (
        ("clv_pinn", "near-100% (PSC* fill is 95-100% from 1213)"),
        ("clv_max", "maxavg-era rows only (MaxC* starts 2019/20)"),
    ):
        null_rate = float(df[col].isna().mean())
        print(
            f"  {col}: {n - int(df[col].isna().sum())}/{n} present "
            f"(null rate {null_rate:.2%}) — expected {expect}"
        )
    betbrain_with_max = df[(df["era"] == "betbrain") & df["clv_max"].notna()]
    if not betbrain_with_max.empty:
        print(f"DATA-QUALITY ALERT: {len(betbrain_with_max)} betbrain rows carry clv_max")
    maxavg = df[df["era"] == "maxavg"]
    if not maxavg.empty and float(maxavg["clv_pinn"].isna().mean()) > 0.10:
        print("DATA-QUALITY ALERT: >10% missing Pinnacle close in the maxavg era")
    if "consulted_before" in df.columns:
        _v2_quality_report(df)
    print("\nschema classification:")
    print(f"  ID:     {', '.join(ID_COLUMNS)}")
    print(f"  SIGNAL: {', '.join(FEATURE_COLUMNS)}")
    print(f"  LABEL:  {', '.join(LABEL_COLUMNS)}")
    print("\nNOTE: open_to_signal_drift is all-null for football-data (pre-match")
    print("columns are a single collection-time snapshot, ~T-1 day; no true opener).")
    # TODO(elo enrichment): point-in-time ClubElo via penaltyblog — join
    #   get_elo_by_team validity windows (from <= match_date <= to) so each
    #   rating uses only strictly-prior matches. Blocked on the versioned
    #   (sport, canonical_name) -> per-source alias mapping table (ClubElo
    #   short names differ from football-data names, e.g. 'Bayern' vs
    #   'Bayern Munich'); unmatched names quarantine with a log line — no
    #   fuzzy auto-merge. Never join "latest rating as of today" onto
    #   historical matches (future-rating leak).


def _v2_quality_report(df: pd.DataFrame) -> None:
    """v2 gates: fresh-slice sizing + null-rate checks for the new features."""
    fresh = df[~df["consulted_before"]]
    print("\nfresh never-consulted slice (consulted_before=False):")
    if fresh.empty:
        print("  (none in this build)")
    else:
        print(fresh.groupby(["league", "era"], observed=True).size().to_string())
        print(f"  TOTAL fresh candidate rows: {len(fresh)}")
        fresh_maxavg = int((fresh["era"] == "maxavg").sum())
        print(f"  of which maxavg era (clv_max-labelable): {fresh_maxavg}")
        if len(fresh) < 1500:
            print(
                "\n*** DATA-QUALITY ALERT — FRESH SLICE TOO SMALL: "
                f"{len(fresh)} < 1500 candidate rows. A one-shot fresh-domain "
                "test on this slice alone is underpowered; the validator must "
                "combine it with the other fresh domains (AH market) or rely "
                "on live shadow CLV + the 2627 season. ***"
            )
    print("\nv2 feature null rates (nullable by construction — LightGBM-native NaN):")
    form_cols = [f"{s}_{f}" for s in ("home", "away") for f in _FORM_FEATS]
    xg_cols = [f"{s}_{f}" for s in ("home", "away") for f in _XG_FEATS]
    n = len(df)
    if n:
        form_null = float(df[form_cols].isna().any(axis=1).mean())
        goals_null = float(df[["home_gf_r5", "away_gf_r5"]].isna().any(axis=1).mean())
        print(f"  any form col null: {form_null:.2%} (stats-column eras differ by division)")
        print(f"  goals form null:   {goals_null:.2%} (expected ~ first-matches share)")
        eligible = df[
            df["league"].isin(UNDERSTAT_COMP)
            & (df["season"].str[:2].astype(int) >= UNDERSTAT_FIRST_END_YEAR - 1)
        ]
        if not eligible.empty:
            xg_null = float(eligible[xg_cols].isna().any(axis=1).mean())
            print(
                f"  xG null within big-5 1415+ rows: {xg_null:.2%} "
                "(first matches + quarantined names)"
            )
        outside = df.drop(eligible.index) if not eligible.empty else df
        if not outside.empty:
            leaked_xg = int(outside[xg_cols].notna().any(axis=1).sum())
            if leaked_xg:
                print(f"DATA-QUALITY ALERT: {leaked_xg} non-big-5/pre-1415 rows carry xG")
            else:
                print("  xG outside big-5 1415+: all null (expected missingness pattern)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leagues", default=None, help="default: all LEAGUES (v2) / V1_LEAGUES (--v1)")
    p.add_argument("--seasons", default=",".join(SEASONS_ALL))
    p.add_argument("--min-edge", type=float, default=MIN_EDGE_DEFAULT)
    p.add_argument("--out", type=Path, default=None, help="default: versioned parquet path")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DEFAULT)
    p.add_argument(
        "--v1",
        action="store_true",
        help="reproduce the v1 dataset exactly (v1 columns, frozen v1 league set, no Understat)",
    )
    p.add_argument(
        "--no-understat",
        action="store_true",
        help="skip Understat xG enrichment (offline mode); xG features stay null",
    )
    args = p.parse_args(argv)
    schema_version = 1 if args.v1 else 2
    leagues_default = ",".join(V1_LEAGUES) if args.v1 else ",".join(LEAGUES)
    leagues = [x.strip() for x in (args.leagues or leagues_default).split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    out: Path = args.out or (OUT_DEFAULT_V1 if args.v1 else OUT_DEFAULT_V2)

    assert_no_label_leak()  # build-breaking gate, runs before any work

    print(
        f"building value-candidate pool v{schema_version}: "
        f"{len(leagues)} leagues x {len(seasons)} seasons, "
        f"min_edge {args.min_edge}, devig {CANONICAL_DEVIG.value}"
    )
    df, counts = asyncio.run(
        build(
            leagues,
            seasons,
            args.cache_dir,
            args.min_edge,
            schema_version=schema_version,
            use_understat=not (args.v1 or args.no_understat),
        )
    )
    _quality_report(df, counts)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"\nwrote {len(df)} rows ({len(df.columns)} columns, schema v{schema_version}) -> {out}")
    print("Decision-support only — this dataset informs manual picks; nothing places bets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
