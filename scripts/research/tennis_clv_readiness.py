"""Tennis CLV readiness probe — held-out incremental CLV vs the Pinnacle close.

DECISION QUESTION
  Is TENNIS ready to graduate from visibility-only to LIVE picks? It is iff our
  SIGNAL-TIME value price beats the de-vigged Pinnacle CLOSING line by more than
  2 standard errors above zero, measured OFFLINE on real scraped data.

WHY THIS SCRIPT EXISTS (and tennis_backtest.py does not answer it)
  `scripts/sports/tennis_backtest.py` grades against tennis-data.co.uk, which has
  NO closing columns — so CLV-vs-close is structurally undefined there and tennis
  is permanently "visibility-only". The data NOW exists in the warehouse:
    - our scraped soft-book tennis odds  : sport key 'tennis' (17 books, h2h)
    - the sharp Pinnacle ARCADIA close   : sport key 'pinnacle_tennis' (h2h)
  This script attaches the Pinnacle close to each of our fixtures via the STRICT
  cross-source matcher (`app.resolution.match_event`, ordered=False for tennis —
  two players, no home/away) and computes incremental log-CLV per matched pick.

DOCTRINE HONORED
  - SIGNAL-TIME price, never the close: our value price is line-shopped from the
    EARLIEST snapshot per book (far from kickoff). No closing odds in the feature.
  - The grade target IS the Pinnacle de-vigged close; CLV uses the SAME devig
    method on both fill and close (odds-math rule). clv_log = ln(fill x p_close).
  - Bootstrap CI is CLUSTERED BY MATCH (one bet per fixture; the H2H sides are
    perfectly anti-correlated, so one pick per match — never both).
  - Real-data gate: if too few fixtures match, the verdict is INSUFFICIENT-DATA
    with the exact n. No result is manufactured.

READ-ONLY. Writes nothing, flips no config, never places a bet.

    uv run python scripts/research/tennis_clv_readiness.py
    uv run python scripts/research/tennis_clv_readiness.py --min-edge 0.0 --devig power
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

# Tennis fixtures play once per day; a Pinnacle archive capture of the same match
# may sit a calendar day off. Same default the settlement matcher uses.
MAX_DAY_DRIFT = 1
OUR_SPORT_KEY = "tennis"
PINNACLE_SPORT_KEY = "pinnacle_tennis"
OUR_MARKET = "match_winner"  # our scraped h2h moneyline
PINNACLE_MARKET = "h2h"  # Pinnacle archive h2h moneyline
GRADUATION_SE = 2.0  # CLV lower bound must clear 2 SE above 0
MIN_N = 150  # below this, no honest verdict — INSUFFICIENT-DATA


# --------------------------------------------------------------------------- #
# Data records (frozen; pure once loaded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SidePrice:
    """One side of one fixture: our line-shopped signal price + the matched
    Pinnacle de-vigged closing probability for that same side."""

    fixture_id: int
    selection: str
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


def two_way_fair_probs(
    odds_by_selection: dict[str, float], method: DevigMethod
) -> dict[str, float] | None:
    """De-vig a two-way (h2h) market into no-vig probs keyed by selection.

    Returns None unless exactly two selections are priced (tennis h2h is two
    mutually-exclusive, full-coverage outcomes — devig-sound)."""
    if len(odds_by_selection) != 2:
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
    stays correct if the market set is ever widened."""
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
# DB layer (read-only)
# --------------------------------------------------------------------------- #
async def _load_signal_prices(session, fixture_ids: list[int]):  # noqa: ANN001
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
                Sport.key == OUR_SPORT_KEY,
                OddsSnapshot.market == OUR_MARKET,
                OddsSnapshot.event_id.in_(fixture_ids),
            )
            .order_by(OddsSnapshot.event_id, OddsSnapshot.captured_at)
        )
    ).all()
    # earliest per (event, book, selection)
    earliest: dict[tuple[int, str, str], float] = {}
    for ev, book, sel, odds, _ts in rows:
        key = (ev, book, sel)
        if key not in earliest:  # rows are ascending by captured_at
            earliest[key] = float(odds)
    return earliest


async def _load_pinnacle_close(
    session,  # noqa: ANN001
    pin_event_id: int,
    cutoff: datetime,
) -> dict[str, float]:
    """Last Pinnacle h2h odds per selection at-or-before the cutoff (the close).

    Uses the change-only persist model: a price row may be old yet still the
    book's close if it never moved. We take the latest-captured row per
    selection capped at the fixture kickoff (no in-play prices)."""
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
                OddsSnapshot.market == PINNACLE_MARKET,
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
    devig_method: DevigMethod,
) -> tuple[list[SidePrice], dict[str, int]]:
    """Match our tennis fixtures to the Pinnacle close and build per-side prices.

    Returns (side_prices, diagnostics). Diagnostics counts every funnel stage so
    a thin sample is explained, not hidden."""
    from app.config import get_settings
    from app.database import create_engine, create_session_factory
    from app.resolution import EventCandidate, default_aliases, match_event
    from app.resolution.matching import normalize_name
    from app.resolution.tennis_names import canonical_tennis_name
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
                    .where(Sport.key == OUR_SPORT_KEY, Event.starts_at.is_not(None))
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
                        Sport.key == PINNACLE_SPORT_KEY,
                        Event.starts_at.is_not(None),
                        Event.starts_at >= min(kickoffs) - window,
                        Event.starts_at <= max(kickoffs) + window,
                    )
                )
            ).all()
            archive = [
                EventCandidate(
                    ref=str(eid),
                    home=canonical_tennis_name(h),
                    away=canonical_tennis_name(a),
                    kickoff=ko,
                )
                for (eid, h, a, ko) in arc_rows
            ]

            # signal-time prices for all our fixtures, one query
            our_ids = [r[0] for r in our_rows]
            signal = await _load_signal_prices(session, our_ids)

            for fix_id, home, away, kickoff in our_rows:
                in_window = [
                    c
                    for c in archive
                    if abs((c.kickoff.date() - kickoff.date()).days) <= MAX_DAY_DRIFT
                ]
                cand = match_event(
                    canonical_tennis_name(home),
                    canonical_tennis_name(away),
                    kickoff,
                    in_window,
                    aliases=aliases,
                    max_day_drift=MAX_DAY_DRIFT,
                    ordered=False,
                )
                if cand is None:
                    if not in_window:
                        diag["no_pinnacle_candidate"] += 1
                    else:
                        # ALIAS suspect iff some in-window Pinnacle event shares a
                        # normalized token with our fixture (else same-day noise).
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
                pin_close = await _load_pinnacle_close(session, pin_id, cutoff)
                close_probs = two_way_fair_probs(pin_close, devig_method)
                if close_probs is None:
                    diag["matched_but_no_close"] += 1
                    continue

                # our signal-time market for THIS fixture: best soft-book price
                # per selection + consensus fair anchor across books.
                our_best: dict[str, tuple[float, str]] = {}
                consensus_input: dict[str, list[float]] = {}
                for (ev, book, sel), odds in signal.items():
                    if ev != fix_id:
                        continue
                    consensus_input.setdefault(sel, []).append(odds)
                    if sel not in our_best or odds > our_best[sel][0]:
                        our_best[sel] = (odds, book)
                if len(our_best) != 2:
                    diag["matched_but_no_signal"] += 1
                    continue

                # consensus anchor = devig of the MEDIAN price per selection
                # (outlier-resistant), our own signal-time fair value (no close).
                median_odds = {sel: sorted(v)[len(v) // 2] for sel, v in consensus_input.items()}
                anchor = two_way_fair_probs(median_odds, devig_method)
                if anchor is None:
                    diag["matched_but_no_signal"] += 1
                    continue

                # selection vocab differs (our 'Fery A.' vs Pinnacle full name).
                # match our selections to the Pinnacle close by NORMALIZED name.
                pin_norm = {canonical_tennis_name(s): p for s, p in close_probs.items()}
                staged: list[SidePrice] = []
                for sel, (odds, book) in our_best.items():
                    cp = pin_norm.get(canonical_tennis_name(sel))
                    if cp is None or not (0.0 < cp < 1.0):
                        continue
                    staged.append(
                        SidePrice(
                            fixture_id=fix_id,
                            selection=sel,
                            our_odds=odds,
                            our_book=book,
                            anchor_fair_prob=anchor[sel],
                            close_fair_prob=cp,
                        )
                    )
                if len(staged) == 2:  # both sides re-keyed cleanly
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
    picks: list[Pick], diag: dict[str, int], devig_method: DevigMethod, min_edge: float
) -> str:
    out: list[str] = []
    w = out.append
    w("=" * 72)
    w("TENNIS CLV READINESS — incremental CLV vs the Pinnacle de-vigged close")
    w("=" * 72)
    w(
        f"devig={devig_method.value}  min_edge={min_edge:.3f}  "
        f"graduation_bar=CLV_lo > 0 at {GRADUATION_SE:.0f} SE  min_n={MIN_N}"
    )
    w("")
    w("--- match funnel (our scraped tennis -> Pinnacle close) ---")
    w(f"  our fixtures (with kickoff)        : {diag['our_fixtures']}")
    w(f"  no Pinnacle candidate in window    : {diag['no_pinnacle_candidate']}  (COVERAGE gap)")
    w(f"  unmatched, alias suspect           : {diag['unmatched_alias_suspect']}  (ALIAS gap)")
    w(f"  unmatched, same-day noise only     : {diag['unmatched_coverage_only']}  (COVERAGE gap)")
    w(f"  matched by strict matcher          : {diag['matched_fixtures']}")
    w(f"  matched but no usable close        : {diag['matched_but_no_close']}")
    w(f"  matched but no signal-time price   : {diag['matched_but_no_signal']}")
    w(f"  GRADED fixtures (close + signal)   : {diag['graded_fixtures']}")
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
    w("  (ROI not graded: outcomes unsettled in this offline window)")
    w("")

    # per-odds-band breakdown (feasible; settlement-free CLV)
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
            f"INSUFFICIENT-DATA (n={n} < min_n={MIN_N}). The matched-fixture count "
            f"is too small for a credible 2-SE separation; do NOT graduate on this."
        )
    elif lo > 0.0 and analytic_lo > 0.0:
        verdict = (
            f"READY. Mean log-CLV {m_clv:+.5f} clears 0 by >{GRADUATION_SE:.0f} SE "
            f"(analytic lo {analytic_lo:+.5f} > 0; bootstrap lo "
            f"{lo:+.5f} > 0). Our signal-time price beats the Pinnacle close."
        )
    else:
        verdict = (
            f"NOT-READY. CLV does not clear 0 by {GRADUATION_SE:.0f} SE "
            f"(analytic lo {analytic_lo:+.5f}, bootstrap lo {lo:+.5f}). "
            f"No demonstrated edge over the close — keep tennis visibility-only."
        )
    w("=" * 72)
    w(f"VERDICT: {verdict}")
    w("=" * 72)
    return "\n".join(out)


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Tennis CLV readiness vs Pinnacle close (read-only).")
    ap.add_argument("--devig", default="power", choices=[m.value for m in DevigMethod])
    ap.add_argument("--min-edge", type=float, default=0.0)
    args = ap.parse_args()
    method = DevigMethod(args.devig)

    sides, diag = await gather_side_prices(method)
    picks = pick_value_bets(sides, args.min_edge)
    print(report(picks, diag, method, args.min_edge))


if __name__ == "__main__":
    asyncio.run(_main())
