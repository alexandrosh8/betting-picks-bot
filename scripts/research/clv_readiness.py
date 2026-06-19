"""Per-sport CLV readiness probe — held-out incremental CLV vs the Pinnacle close.

GENERALIZES `scripts/research/tennis_clv_readiness.py` to any sport via --sport.
Tennis was the template (verdict there: INSUFFICIENT-DATA — a player-name format
gap meant 0/56 scraped fixtures matched a Pinnacle close). This script reports
the SAME funnel + CLV picture for SOCCER and BASKETBALL, the two sports that mint
picks today, using the matcher/alias table tuned for them (commits #43/#44).

DECISION QUESTION
  Does our SIGNAL-TIME value price beat the de-vigged Pinnacle CLOSING line by
  more than 2 SE above zero, measured OFFLINE on real scraped data? This is a
  LIVE-DATA CROSS-CHECK on DB-scraped odds — NOT the authoritative held-out ROI
  backtest (soccer's +22.39% comes from scripts/value_backtest.py on the spent
  football-data holdout; nothing here re-tunes on that).

WHAT DIFFERS PER SPORT (vs the tennis template)
  - soccer    : our scraped market is THREE-way `1x2`; Pinnacle close is `h2h`
                (also three-way, carries Draw). ordered=True (home/away mean).
  - basketball: our scraped market is two-way `home_away`; close is `h2h`.
                ordered=True (home/away mean).
  - selections are TEAM NAMES (+ 'Draw'), not Home/Away tokens. We re-key our
    sides to the matched fixture's home/away via the SAME alias-canonical compare
    the matcher uses, so "Bayern" vs "Bayern Munich" still lines up. The close is
    NEVER a feature — it is only the grade target.

DOCTRINE HONORED (same as tennis)
  - SIGNAL-TIME price, never the close: our value price is line-shopped from the
    EARLIEST snapshot per book. No closing odds enter the feature/edge.
  - The grade target IS the Pinnacle de-vigged close; CLV uses the SAME devig
    method on both fill and close. clv_log = ln(fill x p_close).
  - Bootstrap CI is CLUSTERED BY MATCH (one bet per fixture).
  - Real-data gate: too few matched fixtures -> INSUFFICIENT-DATA with exact n.
  - ROI: outcomes are unsettled in this offline DB window (all events
    'scheduled'); ROI is reported N/A here and stays the job of the settled
    football-data backtest. Nothing is manufactured.

READ-ONLY. Writes nothing, flips no config, never places a bet.

    uv run python scripts/research/clv_readiness.py --sport soccer
    uv run python scripts/research/clv_readiness.py --sport basketball --devig power
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import aliased

from app.backtesting.clv import clv_log
from app.probabilities.devig import DevigMethod, devig

# A Pinnacle archive capture of the same fixture may sit a calendar day off
# (timezone / capture lag). Same default the settlement matcher uses.
MAX_DAY_DRIFT = 1
GRADUATION_SE = 2.0  # CLV lower bound must clear 2 SE above 0
MIN_N = 150  # below this, no honest verdict — INSUFFICIENT-DATA
DRAW_SELECTION = "draw"  # normalized literal for the soccer draw side


@dataclass(frozen=True)
class SportConfig:
    """Per-sport market wiring. `ordered` is True for soccer/basketball (home vs
    away is meaningful); `n_outcomes` is the priced-side count (3 for 1X2)."""

    our_sport_key: str
    pinnacle_sport_key: str
    our_market: str
    pinnacle_market: str
    n_outcomes: int
    has_draw: bool


SPORTS: dict[str, SportConfig] = {
    "soccer": SportConfig(
        our_sport_key="soccer",
        pinnacle_sport_key="pinnacle_soccer",
        our_market="1x2",
        pinnacle_market="h2h",
        n_outcomes=3,
        has_draw=True,
    ),
    "basketball": SportConfig(
        our_sport_key="basketball",
        pinnacle_sport_key="pinnacle_basketball",
        our_market="home_away",
        pinnacle_market="h2h",
        n_outcomes=2,
        has_draw=False,
    ),
    # tennis is kept for reference; the dedicated tennis_clv_readiness.py is the
    # authority there (unordered player pair, different selection re-key).
    "tennis": SportConfig(
        our_sport_key="tennis",
        pinnacle_sport_key="pinnacle_tennis",
        our_market="match_winner",
        pinnacle_market="h2h",
        n_outcomes=2,
        has_draw=False,
    ),
}


# --------------------------------------------------------------------------- #
# Data records (frozen; pure once loaded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SidePrice:
    """One side of one fixture: our line-shopped signal price + the matched
    Pinnacle de-vigged closing probability for that same side."""

    fixture_id: int
    selection: str  # canonical side label: 'home' | 'away' | 'draw'
    our_odds: float  # best soft-book price at SIGNAL time (line-shop)
    our_book: str
    anchor_fair_prob: float  # signal-time consensus no-vig prob (edge source)
    close_fair_prob: float  # Pinnacle de-vigged CLOSING prob (grade target)


@dataclass(frozen=True)
class Pick:
    fixture_id: int
    selection: str
    our_odds: float
    edge: float  # anchor_fair_prob - 1/our_odds at SIGNAL time
    clv: float  # ln(our_odds x close_fair_prob): + means beat the close


# --------------------------------------------------------------------------- #
# Pure math helpers
# --------------------------------------------------------------------------- #
def _tokens(name: str, normalize) -> set[str]:  # noqa: ANN001
    return set(normalize(name).split())


def fair_probs(
    odds_by_selection: dict[str, float], method: DevigMethod, n_expected: int
) -> dict[str, float] | None:
    """De-vig a market into no-vig probs keyed by selection label.

    Returns None unless EXACTLY ``n_expected`` mutually-exclusive, full-coverage
    selections are priced (devig is only sound on a complete market: 2 sides for
    h2h, 3 for 1X2). Any side priced <= 1.0 is rejected."""
    if len(odds_by_selection) != n_expected:
        return None
    sels = list(odds_by_selection)
    odds = [odds_by_selection[s] for s in sels]
    if any(o <= 1.0 for o in odds):
        return None
    probs = devig(odds, method=method)
    return dict(zip(sels, probs, strict=True))


def bootstrap_clv_ci(
    picks: list[Pick],
    *,
    n_boot: int = 5000,
    seed: int = 20260619,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean CLV, resampling CLUSTERS (fixtures).
    One pick per fixture here, so cluster == pick, but the clustered resampler
    stays correct if the market set is ever widened to multiple bets per game."""
    by_fix: dict[int, list[Pick]] = {}
    for p in picks:
        by_fix.setdefault(p.fixture_id, []).append(p)
    clusters = list(by_fix.values())
    rng = random.Random(seed)
    means: list[float] = []
    k = len(clusters)
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(k)] for _ in range(k)]
        flat = [p.clv for cl in sample for p in cl]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def mean_se(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var) / math.sqrt(len(xs))


# --------------------------------------------------------------------------- #
# Side re-keying: map raw selection literals -> canonical 'home'/'away'/'draw'
# --------------------------------------------------------------------------- #
def _relabel_sides(
    raw_by_sel: dict[str, float],
    home: str,
    away: str,
    cfg: SportConfig,
    aliases,  # noqa: ANN001
    normalize,  # noqa: ANN001
) -> dict[str, float] | None:
    """Re-key {raw_selection -> value} to {'home'|'away'|'draw' -> value}.

    Sides are matched to the fixture's home/away team by the SAME alias-canonical
    compare the matcher uses (so naming gaps like 'Bayern' vs 'Bayern Munich' map
    correctly). 'Draw' maps by normalized literal. Returns None if the labels are
    not a clean, complete, unambiguous set — never guesses a side."""
    canon_home = aliases.canonical(home)
    canon_away = aliases.canonical(away)
    if not canon_home or not canon_away or canon_home == canon_away:
        return None
    out: dict[str, float] = {}
    for sel, val in raw_by_sel.items():
        if cfg.has_draw and normalize(sel) == DRAW_SELECTION:
            label = "draw"
        elif aliases.canonical(sel) == canon_home:
            label = "home"
        elif aliases.canonical(sel) == canon_away:
            label = "away"
        else:
            return None  # an unrecognized side -> cannot trust the re-key
        if label in out:
            return None  # two raw sides collapsed to one label -> ambiguous
        out[label] = val
    expected = {"home", "away"} | ({"draw"} if cfg.has_draw else set())
    return out if set(out) == expected else None


# --------------------------------------------------------------------------- #
# DB layer (read-only)
# --------------------------------------------------------------------------- #
async def _load_signal_prices(session, fixture_ids: list[int], cfg: SportConfig):  # noqa: ANN001
    """Per fixture, the EARLIEST snapshot per (book, selection) = signal time."""
    from app.storage.models import Event, OddsSnapshot, Sport

    rows = (
        await session.execute(
            select(
                OddsSnapshot.event_id,
                OddsSnapshot.bookmaker,
                OddsSnapshot.selection,
                OddsSnapshot.decimal_odds,
                OddsSnapshot.captured_at,
            )
            .join(Event, Event.id == OddsSnapshot.event_id)
            .join(Sport, Sport.id == Event.sport_id)
            .where(
                Sport.key == cfg.our_sport_key,
                OddsSnapshot.market == cfg.our_market,
                OddsSnapshot.event_id.in_(fixture_ids),
            )
            .order_by(OddsSnapshot.event_id, OddsSnapshot.captured_at)
        )
    ).all()
    # earliest per (event, book, selection): rows ascend by captured_at.
    earliest: dict[tuple[int, str, str], float] = {}
    for ev, book, sel, odds, _ts in rows:
        key = (ev, book, sel)
        if key not in earliest:
            earliest[key] = float(odds)
    return earliest


async def _load_pinnacle_close(
    session,  # noqa: ANN001
    pin_event_id: int,
    cutoff: datetime,
    cfg: SportConfig,
) -> dict[str, float]:
    """Last Pinnacle odds per selection at-or-before the cutoff (the close).

    Change-only persist: a price row may be old yet still the book's close if it
    never moved. We take the latest-captured row per selection capped at the
    fixture kickoff (no in-play prices)."""
    from app.storage.models import OddsSnapshot

    rows = (
        await session.execute(
            select(
                OddsSnapshot.selection,
                OddsSnapshot.decimal_odds,
                OddsSnapshot.captured_at,
            )
            .where(
                OddsSnapshot.event_id == pin_event_id,
                OddsSnapshot.market == cfg.pinnacle_market,
                OddsSnapshot.captured_at <= cutoff,
            )
            .order_by(OddsSnapshot.selection, OddsSnapshot.captured_at.desc())
        )
    ).all()
    close: dict[str, float] = {}
    for sel, odds, _ts in rows:
        if sel not in close:  # rows desc by captured_at -> first seen is latest
            close[sel] = float(odds)
    return close


async def gather_side_prices(  # noqa: PLR0912, PLR0915
    cfg: SportConfig, devig_method: DevigMethod
) -> tuple[list[SidePrice], dict[str, int]]:
    """Match our fixtures to the Pinnacle close and build per-side prices.

    Returns (side_prices, diagnostics). Diagnostics count every funnel stage so a
    thin sample is explained, not hidden."""
    from app.config import get_settings
    from app.database import create_engine, create_session_factory
    from app.resolution import EventCandidate, default_aliases, match_event
    from app.resolution.matching import normalize_name
    from app.storage.models import Event, Sport, Team

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    aliases = default_aliases()
    window = timedelta(days=MAX_DAY_DRIFT + 1)

    diag = {
        "our_fixtures": 0,
        "matched_fixtures": 0,
        "no_pinnacle_candidate": 0,
        "unmatched_alias_suspect": 0,
        "unmatched_coverage_only": 0,
        "matched_but_no_close": 0,
        "matched_but_no_signal": 0,
        "graded_fixtures": 0,
    }
    side_prices: list[SidePrice] = []

    try:
        async with session_factory() as session:
            home_t, away_t = aliased(Team), aliased(Team)
            our_rows = (
                await session.execute(
                    select(Event.id, home_t.name, away_t.name, Event.starts_at)
                    .select_from(Event)
                    .join(Sport, Event.sport_id == Sport.id)
                    .join(home_t, Event.home_team_id == home_t.id)
                    .join(away_t, Event.away_team_id == away_t.id)
                    .where(Sport.key == cfg.our_sport_key, Event.starts_at.is_not(None))
                )
            ).all()
            diag["our_fixtures"] = len(our_rows)
            if not our_rows:
                return side_prices, diag

            kickoffs = [r[3] for r in our_rows]
            ah, aw = aliased(Team), aliased(Team)
            arc_rows = (
                await session.execute(
                    select(Event.id, ah.name, aw.name, Event.starts_at)
                    .join(Sport, Event.sport_id == Sport.id)
                    .join(ah, Event.home_team_id == ah.id)
                    .join(aw, Event.away_team_id == aw.id)
                    .where(
                        Sport.key == cfg.pinnacle_sport_key,
                        Event.starts_at.is_not(None),
                        Event.starts_at >= min(kickoffs) - window,
                        Event.starts_at <= max(kickoffs) + window,
                    )
                )
            ).all()
            archive = [
                EventCandidate(ref=str(eid), home=h, away=a, kickoff=ko)
                for (eid, h, a, ko) in arc_rows
            ]

            our_ids = [r[0] for r in our_rows]
            signal = await _load_signal_prices(session, our_ids, cfg)

            for fix_id, home, away, kickoff in our_rows:
                in_window = [
                    c
                    for c in archive
                    if abs((c.kickoff.date() - kickoff.date()).days) <= MAX_DAY_DRIFT
                ]
                cand = match_event(
                    home,
                    away,
                    kickoff,
                    in_window,
                    aliases=aliases,
                    max_day_drift=MAX_DAY_DRIFT,
                    ordered=True,  # soccer/basketball: home vs away is meaningful
                )
                if cand is None:
                    if not in_window:
                        diag["no_pinnacle_candidate"] += 1
                    else:
                        pick_toks = _tokens(home, normalize_name) | _tokens(away, normalize_name)
                        suspect = any(
                            (_tokens(c.home, normalize_name) | _tokens(c.away, normalize_name))
                            & pick_toks
                            for c in in_window
                        )
                        key = "unmatched_alias_suspect" if suspect else "unmatched_coverage_only"
                        diag[key] += 1
                    continue

                diag["matched_fixtures"] += 1
                pin_id = int(cand.ref)
                cutoff = min(cand.kickoff, kickoff)
                pin_close_raw = await _load_pinnacle_close(session, pin_id, cutoff, cfg)
                # Pinnacle home/away orientation == OUR fixture's (ordered match),
                # so re-key the close against OUR home/away labels.
                pin_close = _relabel_sides(pin_close_raw, home, away, cfg, aliases, normalize_name)
                if pin_close is None:
                    diag["matched_but_no_close"] += 1
                    continue
                close_probs = fair_probs(pin_close, devig_method, cfg.n_outcomes)
                if close_probs is None:
                    diag["matched_but_no_close"] += 1
                    continue

                # our signal-time market: best soft-book price per RAW selection +
                # median consensus per RAW selection, then re-key both to sides.
                our_best_raw: dict[str, tuple[float, str]] = {}
                consensus_raw: dict[str, list[float]] = {}
                for (ev, book, sel), odds in signal.items():
                    if ev != fix_id:
                        continue
                    consensus_raw.setdefault(sel, []).append(odds)
                    if sel not in our_best_raw or odds > our_best_raw[sel][0]:
                        our_best_raw[sel] = (odds, book)

                best_odds = _relabel_sides(
                    {s: v[0] for s, v in our_best_raw.items()},
                    home,
                    away,
                    cfg,
                    aliases,
                    normalize_name,
                )
                if best_odds is None:
                    diag["matched_but_no_signal"] += 1
                    continue
                # recover the best-price book string per side label
                book_by_side: dict[str, str] = {}
                for s, (_o, bk) in our_best_raw.items():
                    if cfg.has_draw and normalize_name(s) == DRAW_SELECTION:
                        book_by_side["draw"] = bk
                    elif aliases.canonical(s) == aliases.canonical(home):
                        book_by_side["home"] = bk
                    elif aliases.canonical(s) == aliases.canonical(away):
                        book_by_side["away"] = bk

                median_raw = {s: sorted(v)[len(v) // 2] for s, v in consensus_raw.items()}
                median_sides = _relabel_sides(median_raw, home, away, cfg, aliases, normalize_name)
                if median_sides is None:
                    diag["matched_but_no_signal"] += 1
                    continue
                anchor = fair_probs(median_sides, devig_method, cfg.n_outcomes)
                if anchor is None:
                    diag["matched_but_no_signal"] += 1
                    continue

                staged: list[SidePrice] = []
                for side, odds in best_odds.items():
                    cp = close_probs.get(side)
                    if cp is None or not (0.0 < cp < 1.0):
                        continue
                    staged.append(
                        SidePrice(
                            fixture_id=fix_id,
                            selection=side,
                            our_odds=odds,
                            our_book=book_by_side.get(side, "?"),
                            anchor_fair_prob=anchor[side],
                            close_fair_prob=cp,
                        )
                    )
                if len(staged) == cfg.n_outcomes:  # every side re-keyed cleanly
                    side_prices.extend(staged)
                    diag["graded_fixtures"] += 1
                else:
                    diag["matched_but_no_close"] += 1
    finally:
        await engine.dispose()

    return side_prices, diag


# --------------------------------------------------------------------------- #
# Strategy: one value pick per fixture (highest signal-time edge >= min_edge)
# --------------------------------------------------------------------------- #
def pick_value_bets(sides: list[SidePrice], min_edge: float) -> list[Pick]:
    by_fix: dict[int, list[SidePrice]] = {}
    for s in sides:
        by_fix.setdefault(s.fixture_id, []).append(s)
    picks: list[Pick] = []
    for fix_id, fixture_sides in by_fix.items():
        best: Pick | None = None
        for s in fixture_sides:
            edge = s.anchor_fair_prob - (1.0 / s.our_odds)
            if edge < min_edge:
                continue
            clv = clv_log(s.our_odds, s.close_fair_prob)
            cand = Pick(fix_id, s.selection, s.our_odds, edge, clv)
            if best is None or cand.edge > best.edge:
                best = cand
        if best is not None:
            picks.append(best)
    return picks


# --------------------------------------------------------------------------- #
# Reporting + verdict
# --------------------------------------------------------------------------- #
def report(  # noqa: PLR0915
    sport: str,
    picks: list[Pick],
    diag: dict[str, int],
    devig_method: DevigMethod,
    min_edge: float,
) -> str:
    out: list[str] = []
    w = out.append
    w("=" * 72)
    w(f"{sport.upper()} CLV READINESS — incremental CLV vs the Pinnacle de-vigged close")
    w("=" * 72)
    w(
        f"devig={devig_method.value}  min_edge={min_edge:.3f}  "
        f"graduation_bar=CLV_lo > 0 at {GRADUATION_SE:.0f} SE  min_n={MIN_N}"
    )
    w("LIVE-DATA cross-check on DB-scraped odds (NOT the football-data ROI backtest).")
    w("")
    w(f"--- match funnel (our scraped {sport} -> Pinnacle close) ---")
    w(f"  our fixtures (with kickoff)        : {diag['our_fixtures']}")
    w(f"  no Pinnacle candidate in window    : {diag['no_pinnacle_candidate']}  (COVERAGE gap)")
    w(f"  unmatched, alias suspect           : {diag['unmatched_alias_suspect']}  (ALIAS gap)")
    w(f"  unmatched, same-day noise only     : {diag['unmatched_coverage_only']}  (COVERAGE gap)")
    w(f"  matched by strict matcher          : {diag['matched_fixtures']}")
    w(f"  matched but no usable close        : {diag['matched_but_no_close']}")
    w(f"  matched but no signal-time price   : {diag['matched_but_no_signal']}")
    w(f"  GRADED fixtures (close + signal)   : {diag['graded_fixtures']}")
    if diag["our_fixtures"]:
        rate = diag["matched_fixtures"] / diag["our_fixtures"] * 100.0
        w(f"  MATCH RATE (matched / our fixtures): {rate:.1f}%")
    w("")

    n = len(picks)
    if n == 0:
        w("PICKS: 0 — nothing to grade.")
        w("")
        w("VERDICT: INSUFFICIENT-DATA (n=0 graded picks).")
        return "\n".join(out)

    clvs = [p.clv for p in picks]
    m_clv, se_clv = mean_se(clvs)
    beat = sum(1 for c in clvs if c > 0.0) / n
    lo, hi = bootstrap_clv_ci(picks)
    analytic_lo = m_clv - GRADUATION_SE * se_clv

    w(f"--- graded picks (one value side per fixture, edge >= {min_edge:.3f}) ---")
    w(f"  n picks (== n clusters)            : {n}")
    w(f"  mean log-CLV                       : {m_clv:+.5f}")
    w(f"  analytic SE (iid)                  : {se_clv:.5f}")
    w(f"  mean - {GRADUATION_SE:.0f}*SE (analytic lo)         : {analytic_lo:+.5f}")
    w(f"  bootstrap 95% CI (match-clustered) : [{lo:+.5f}, {hi:+.5f}]")
    w(f"  beat-close rate                    : {beat * 100:.1f}%")
    w("  ROI                                : N/A (events unsettled in this offline window)")
    w("")

    # per-side breakdown (home/away/draw)
    w("--- per-side CLV ---")
    for side in ("home", "away", "draw"):
        bucket = [p.clv for p in picks if p.selection == side]
        if bucket:
            bm, _ = mean_se(bucket)
            bb = sum(1 for c in bucket if c > 0) / len(bucket)
            w(f"  side={side:5s}  n={len(bucket):>4}  meanCLV={bm:+.5f}  beat={bb * 100:.0f}%")
    w("")

    # per-odds-band breakdown (settlement-free CLV)
    w("--- per-odds-band CLV ---")
    bands = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99.0)]
    for lo_b, hi_b in bands:
        bucket = [p.clv for p in picks if lo_b <= p.our_odds < hi_b]
        if bucket:
            bm, _ = mean_se(bucket)
            bb = sum(1 for c in bucket if c > 0) / len(bucket)
            w(
                f"  odds [{lo_b:.1f},{hi_b:.1f})  n={len(bucket):>4}  "
                f"meanCLV={bm:+.5f}  beat={bb * 100:.0f}%"
            )
    w("")

    # VERDICT — computed from held-out numbers, never hardcoded.
    if n < MIN_N:
        verdict = (
            f"INSUFFICIENT-DATA (n={n} < min_n={MIN_N}). Too few matched fixtures "
            f"for a credible 2-SE separation; do NOT graduate on this alone."
        )
    elif lo > 0.0 and analytic_lo > 0.0:
        verdict = (
            f"READY. Mean log-CLV {m_clv:+.5f} clears 0 by >{GRADUATION_SE:.0f} SE "
            f"(analytic lo {analytic_lo:+.5f} > 0; bootstrap lo {lo:+.5f} > 0). "
            f"Our signal-time price beats the Pinnacle close on live-scraped data."
        )
    else:
        verdict = (
            f"NOT-READY. CLV does not clear 0 by {GRADUATION_SE:.0f} SE "
            f"(analytic lo {analytic_lo:+.5f}, bootstrap lo {lo:+.5f}). "
            f"No demonstrated edge over the close on this window."
        )
    w("=" * 72)
    w(f"VERDICT: {verdict}")
    w("=" * 72)
    return "\n".join(out)


async def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-sport CLV readiness vs the Pinnacle close (read-only)."
    )
    ap.add_argument("--sport", required=True, choices=sorted(SPORTS))
    ap.add_argument("--devig", default="power", choices=[m.value for m in DevigMethod])
    ap.add_argument("--min-edge", type=float, default=0.0)
    args = ap.parse_args()

    cfg = SPORTS[args.sport]
    method = DevigMethod(args.devig)
    sides, diag = await gather_side_prices(cfg, method)
    picks = pick_value_bets(sides, args.min_edge)
    print(report(args.sport, picks, diag, method, args.min_edge))


if __name__ == "__main__":
    asyncio.run(_main())
