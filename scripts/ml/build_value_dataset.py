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

Run (downloads are cached under data/ml/cache/ -- read-only GETs):

    uv run python scripts/ml/build_value_dataset.py
    uv run python scripts/ml/build_value_dataset.py --leagues E0,D1 --seasons 2425

Output: data/ml/value_candidates.parquet (data/ is gitignored; raw artifacts
are never committed).

Decision-support only — nothing here places bets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import io
import sys
from collections.abc import Callable, Sequence
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
OUT_DEFAULT = REPO_ROOT / "data" / "ml" / "value_candidates.parquet"
CACHE_DEFAULT = REPO_ROOT / "data" / "ml" / "cache"

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


def _devig_spreads(odds: Sequence[float]) -> list[float]:
    """Per-selection max pairwise fair-prob disagreement across SPREAD_METHODS."""
    fairs = [devig(odds, method=m) for m in SPREAD_METHODS]
    return [max(abs(a[i] - b[i]) for a in fairs for b in fairs) for i in range(len(odds))]


def candidates_from_rows(
    league: str,
    season: str,
    rows: list[dict[str, str]],
    min_edge: float = MIN_EDGE_DEFAULT,
) -> list[Candidate]:
    """All candidate selections (edge >= min_edge) for one league-season.

    Candidate rule parity with scripts/value_backtest.py::bets_for, except the
    pool keeps EVERY qualifying selection (the harness applies one-bet-per-
    match via is_argmax_edge and any threshold/odds filters downstream).
    """
    out: list[Candidate] = []
    for r in rows:
        if not r.get("HomeTeam") or not r.get("Date"):
            continue
        d = _parse_match_date(r.get("Date"))
        if d is None:
            continue
        kickoff = _parse_kickoff_utc(d, r.get("Time"))
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
            spreads = _devig_spreads(ps)
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
                        home_team=r["HomeTeam"].strip(),
                        away_team=(r.get("AwayTeam") or "").strip(),
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


def _to_dataframe(cands: list[Candidate]) -> pd.DataFrame:
    df = pd.DataFrame([dataclasses.asdict(c) for c in cands], columns=list(SCHEMA))
    df = df.sort_values(
        ["league", "season", "match_date", "home_team", "away_team", "market", "selection"],
        kind="mergesort",  # stable -> deterministic output ordering
    ).reset_index(drop=True)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    cols = ("pinn_close_fair", "max_close_fair", "clv_pinn", "clv_max", "open_to_signal_drift")
    for col in cols:
        df[col] = df[col].astype("float64")
    df["won"] = df["won"].astype(bool)
    df["is_argmax_edge"] = df["is_argmax_edge"].astype(bool)
    df["book_count"] = df["book_count"].astype("int32")
    return df


async def build(
    leagues: list[str], seasons: list[str], cache_dir: Path, min_edge: float
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
                n_matches = sum(1 for r in rows if r.get("HomeTeam") and r.get("Date"))
                cands = candidates_from_rows(league, season, rows, min_edge)
                counts.append((league, season, n_matches, len(cands)))
                all_cands.extend(cands)
    return _to_dataframe(all_cands), counts


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leagues", default=",".join(LEAGUES))
    p.add_argument("--seasons", default=",".join(SEASONS_ALL))
    p.add_argument("--min-edge", type=float, default=MIN_EDGE_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DEFAULT)
    args = p.parse_args(argv)
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]

    assert_no_label_leak()  # build-breaking gate, runs before any work

    print(
        f"building value-candidate pool: {len(leagues)} leagues x {len(seasons)} seasons, "
        f"min_edge {args.min_edge}, devig {CANONICAL_DEVIG.value}"
    )
    df, counts = asyncio.run(build(leagues, seasons, args.cache_dir, args.min_edge))
    _quality_report(df, counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print("Decision-support only — this dataset informs manual picks; nothing places bets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
