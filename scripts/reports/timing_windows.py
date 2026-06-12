"""Timing-window analyzer over odds_snapshots — when is detection worth most?

For every finished event with a *settled close* (kickoff known and passed,
snapshot coverage reaching to within CLOSE_MAX_GAP of kickoff), buckets the
event's pre-kickoff price observations by hours-to-kickoff and measures:

  (i)  price DRIFT toward the close per bucket — ln(close/then) per
       (market, selection, book) observation: negative mean = prices
       shorten toward kickoff (early prices were better);
  (ii) hypothetical EDGE-IF-DETECTED-THEN vs realized CLV — replays the
       live value scan (same anchoring + devig + commission netting as
       app/pipeline.py) on the reconstructed market state at each bucket's
       latest observation time, then scores every hypothetical pick against
       the devigged close: clv_log = ln(eff_fill_then x close_fair).

Honesty gate (binding): until at least --min-events events qualify (>= 2
distinct pre-kickoff snapshot times AND a settled close), the report prints
'INSUFFICIENT DATA: n=...' and per-bucket observation counts ONLY — no point
estimates. odds_snapshots started filling 2026-06-11; rerun weekly:

    uv run python scripts/reports/timing_windows.py

Read-only analysis of our own warehouse. Nothing here places bets.
Pure computation lives in the module-level functions (unit-tested on
synthetic fixtures in tests/test_timing_windows.py); the DB read and
Settings access happen only in main() — the script's composition root.
"""

import argparse
import asyncio
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.backtesting.clv import clv_log as _clv_log  # noqa: E402
from app.edge.value import anchor_fair_probs, find_value_bets  # noqa: E402
from app.probabilities.devig import DevigMethod  # noqa: E402

# Hours-to-kickoff buckets: lo <= h < hi. Mirrors the mandate's windows.
BUCKETS: tuple[tuple[float, float, str], ...] = (
    (48.0, math.inf, ">48h"),
    (24.0, 48.0, "24-48h"),
    (5.0, 24.0, "5-24h"),
    (1.0, 5.0, "1-5h"),
    (0.0, 1.0, "<1h"),
)

# Same scrape-coverage rule as the snapshot close in app/clv_trueup.py: the
# event's LAST pre-kickoff capture must reach to within this gap of kickoff,
# else the event fell out of the scrape and its "close" is not a close.
CLOSE_MAX_GAP = timedelta(hours=4)

# Below this many qualified events the report refuses point estimates.
DEFAULT_MIN_EVENTS = 30


@dataclass(frozen=True)
class SnapshotRow:
    """One odds_snapshots observation, floats only (Decimal stays at the DB)."""

    bookmaker: str
    market: str  # stored line-qualified market key — the devig group
    selection: str
    decimal_odds: float
    captured_at: datetime  # UTC-aware


@dataclass(frozen=True)
class EventSnapshots:
    event_label: str
    kickoff: datetime  # UTC-aware
    rows: tuple[SnapshotRow, ...]


@dataclass
class BucketAgg:
    drift_logs: list[float] = field(default_factory=list)
    # (edge_if_detected_then, realized_clv_log) per hypothetical pick
    picks: list[tuple[float, float]] = field(default_factory=list)


def bucket_label(hours_to_kickoff: float) -> str | None:
    """Bucket for a pre-kickoff observation; None for post-kickoff times."""
    for lo, hi, label in BUCKETS:
        if lo <= hours_to_kickoff < hi:
            return label
    return None


def _hours_to_kickoff(at: datetime, kickoff: datetime) -> float:
    return (kickoff - at).total_seconds() / 3600.0


def state_at(rows: Iterable[SnapshotRow], at: datetime) -> dict[str, dict[str, dict[str, float]]]:
    """Market state at `at`: market -> selection -> book -> odds, taking the
    LAST observation at-or-before `at` per key. Change-only persistence means
    an old row is still the live price until a newer one exists — per-row age
    must never gate validity (same rule as closing_odds_from_snapshots)."""
    latest: dict[tuple[str, str, str], SnapshotRow] = {}
    for row in rows:
        if row.captured_at > at or row.decimal_odds <= 1.0:
            continue
        key = (row.market, row.selection, row.bookmaker)
        kept = latest.get(key)
        if kept is None or row.captured_at > kept.captured_at:
            latest[key] = row
    state: dict[str, dict[str, dict[str, float]]] = {}
    for (market, selection, book), row in latest.items():
        state.setdefault(market, {}).setdefault(selection, {})[book] = row.decimal_odds
    return state


def event_qualifies(event: EventSnapshots, max_gap: timedelta = CLOSE_MAX_GAP) -> bool:
    """A settled close exists AND there is line history to bucket: >= 2
    distinct pre-kickoff capture times, the last within `max_gap` of kickoff."""
    pre = sorted({r.captured_at for r in event.rows if r.captured_at <= event.kickoff})
    return len(pre) >= 2 and event.kickoff - pre[-1] <= max_gap


def drift_by_bucket(event: EventSnapshots) -> dict[str, list[float]]:
    """ln(close/then) per pre-kickoff observation, bucketed by hours-to-
    kickoff. The close row itself is excluded — it would contribute an exact
    0 to its own bucket and bias late-bucket drift toward zero."""
    close = state_at(event.rows, event.kickoff)
    close_time: dict[tuple[str, str, str], datetime] = {}
    for row in event.rows:
        if row.captured_at > event.kickoff or row.decimal_odds <= 1.0:
            continue
        key = (row.market, row.selection, row.bookmaker)
        kept = close_time.get(key)
        if kept is None or row.captured_at > kept:
            close_time[key] = row.captured_at
    out: dict[str, list[float]] = {}
    for row in event.rows:
        if row.captured_at > event.kickoff or row.decimal_odds <= 1.0:
            continue
        key = (row.market, row.selection, row.bookmaker)
        if row.captured_at == close_time.get(key):
            continue  # the close observation itself
        close_odds = close.get(row.market, {}).get(row.selection, {}).get(row.bookmaker)
        if close_odds is None:
            continue
        label = bucket_label(_hours_to_kickoff(row.captured_at, event.kickoff))
        if label is None:
            continue
        out.setdefault(label, []).append(math.log(close_odds / row.decimal_odds))
    return out


def edge_vs_clv_by_bucket(
    event: EventSnapshots,
    *,
    min_edge: float,
    min_odds: float,
    devig_method: DevigMethod,
) -> dict[str, list[tuple[float, float]]]:
    """(edge_if_detected_then, realized_clv_log) per hypothetical pick, per
    bucket. Detection replays the LIVE scan (find_value_bets: anchored fair,
    commission-netted best price, same devig) on the market state at the
    bucket's latest observation time — the moment detection could actually
    have run; realized CLV scores the then-fill against the devigged close.
    SAME devig method on both sides, EFFECTIVE odds on both sides (the
    netting convention of app/clv_trueup.py)."""
    close_fair_by_key: dict[tuple[str, str], float] = {}
    for market, prices in state_at(event.rows, event.kickoff).items():
        anchored = anchor_fair_probs(prices, devig_method=devig_method)
        if anchored is None:
            continue
        for sel, fair in anchored[1].items():
            close_fair_by_key[(market, sel)] = fair
    if not close_fair_by_key:
        return {}

    eval_at: dict[str, datetime] = {}
    for row in event.rows:
        if row.captured_at > event.kickoff:
            continue
        label = bucket_label(_hours_to_kickoff(row.captured_at, event.kickoff))
        if label is None:
            continue
        kept = eval_at.get(label)
        if kept is None or row.captured_at > kept:
            eval_at[label] = row.captured_at

    out: dict[str, list[tuple[float, float]]] = {}
    for label, at in eval_at.items():
        for market, prices in state_at(event.rows, at).items():
            for v in find_value_bets(
                prices, min_edge=min_edge, min_odds=min_odds, devig_method=devig_method
            ):
                close_fair = close_fair_by_key.get((market, v.selection))
                if close_fair is None or not 0.0 < close_fair < 1.0:
                    continue
                clv = _clv_log(v.best_odds_effective, close_fair)
                out.setdefault(label, []).append((v.edge, clv))
    return out


def analyze_events(
    events: Sequence[EventSnapshots],
    *,
    min_edge: float,
    min_odds: float,
    devig_method: DevigMethod,
    max_gap: timedelta = CLOSE_MAX_GAP,
) -> tuple[int, dict[str, BucketAgg]]:
    """(n_qualified_events, per-bucket aggregates) over qualifying events."""
    aggs: dict[str, BucketAgg] = {label: BucketAgg() for _, _, label in BUCKETS}
    qualified = 0
    for event in events:
        if not event_qualifies(event, max_gap):
            continue
        qualified += 1
        for label, drifts in drift_by_bucket(event).items():
            aggs[label].drift_logs.extend(drifts)
        picks = edge_vs_clv_by_bucket(
            event, min_edge=min_edge, min_odds=min_odds, devig_method=devig_method
        )
        for label, pairs in picks.items():
            aggs[label].picks.extend(pairs)
    return qualified, aggs


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(value: float | None, width: int, precision: int) -> str:
    return f"{value:>{width}.{precision}f}" if value is not None else f"{'—':>{width}}"


def render_report(
    qualified: int,
    aggs: Mapping[str, BucketAgg],
    *,
    min_events: int,
    min_edge: float,
    devig_method: DevigMethod,
) -> str:
    """The weekly report text. Below `min_events` qualified events the
    estimates are WITHHELD — counts only, behind the INSUFFICIENT banner
    (point estimates without depth would be noise wearing a number's face)."""
    sufficient = qualified >= min_events
    lines = [
        "TIMING-WINDOW ANALYZER — odds_snapshots price history",
        f"qualified events (>=2 snapshot times and a close): {qualified}",
        f"hypothetical scan: min_edge={min_edge}, devig={devig_method}",
    ]
    if not sufficient:
        lines.insert(
            1,
            f"INSUFFICIENT DATA: n={qualified} events with >=2 snapshot times "
            f"and a close (need >={min_events}) — counts shown, estimates "
            "withheld; rerun weekly as snapshot history accumulates.",
        )
    header = (
        f"{'bucket':>8} | {'obs':>6} | {'drift ln(close/then)':>21} | "
        f"{'picks':>6} | {'edge then':>9} | {'clv_log':>8} | {'beat%':>5}"
    )
    lines += [header, "-" * len(header)]
    for _, _, label in BUCKETS:
        agg = aggs.get(label) or BucketAgg()
        n_obs, n_picks = len(agg.drift_logs), len(agg.picks)
        if sufficient:
            beat = 100.0 * sum(1 for _, c in agg.picks if c > 0) / n_picks if agg.picks else None
            lines.append(
                f"{label:>8} | {n_obs:>6} | {_fmt(_mean(agg.drift_logs), 21, 5)} | "
                f"{n_picks:>6} | {_fmt(_mean([e for e, _ in agg.picks]), 9, 4)} | "
                f"{_fmt(_mean([c for _, c in agg.picks]), 8, 4)} | {_fmt(beat, 5, 1)}"
            )
        else:
            lines.append(
                f"{label:>8} | {n_obs:>6} | {'withheld':>21} | "
                f"{n_picks:>6} | {'withheld':>9} | {'—':>8} | {'—':>5}"
            )
    lines.append(
        "drift: mean ln(close/then) per observation — negative = prices "
        "shortened toward kickoff (earlier detection got better prices). "
        "Evidence only; nothing here is a profit promise."
    )
    return "\n".join(lines)


async def _load_events(days: int) -> list[EventSnapshots]:
    """DB read (composition root): finished events of the last `days` days
    with any snapshot history, as plain-float EventSnapshots."""
    from sqlalchemy import select
    from sqlalchemy.orm import aliased

    from app.config import get_settings
    from app.database import create_engine, create_session_factory
    from app.storage.models import Event, OddsSnapshot, Team

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    now = datetime.now(tz=UTC)
    out: list[EventSnapshots] = []
    try:
        async with session_factory() as session:
            home, away = aliased(Team), aliased(Team)
            events = (
                await session.execute(
                    select(Event.id, Event.starts_at, home.name, away.name)
                    .join(home, Event.home_team_id == home.id)
                    .join(away, Event.away_team_id == away.id)
                    .where(
                        Event.starts_at.is_not(None),
                        Event.starts_at <= now,
                        Event.starts_at >= now - timedelta(days=days),
                    )
                )
            ).all()
            for event_id, kickoff, home_name, away_name in events:
                rows = (
                    await session.execute(
                        select(
                            OddsSnapshot.bookmaker,
                            OddsSnapshot.market,
                            OddsSnapshot.selection,
                            OddsSnapshot.decimal_odds,
                            OddsSnapshot.captured_at,
                        ).where(OddsSnapshot.event_id == event_id)
                    )
                ).all()
                if not rows:
                    continue
                out.append(
                    EventSnapshots(
                        event_label=f"{home_name} vs {away_name}",
                        kickoff=kickoff,
                        rows=tuple(
                            SnapshotRow(
                                bookmaker=bookmaker,
                                market=market,
                                selection=selection,
                                decimal_odds=float(odds),
                                captured_at=captured_at,
                            )
                            for bookmaker, market, selection, odds, captured_at in rows
                        ),
                    )
                )
    finally:
        await engine.dispose()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Timing-window analyzer over odds_snapshots")
    parser.add_argument("--days", type=int, default=60, help="lookback window (days)")
    parser.add_argument(
        "--min-events",
        type=int,
        default=DEFAULT_MIN_EVENTS,
        help="qualified-event floor below which estimates are withheld",
    )
    parser.add_argument("--min-edge", type=float, default=None, help="default: VALUE_MIN_EDGE")
    parser.add_argument("--min-odds", type=float, default=None, help="default: VALUE_MIN_ODDS")
    args = parser.parse_args()

    from app.config import get_settings

    settings = get_settings()
    min_edge = args.min_edge if args.min_edge is not None else settings.value_min_edge
    min_odds = args.min_odds if args.min_odds is not None else settings.value_min_odds
    devig_method = DevigMethod(settings.value_devig)

    events = asyncio.run(_load_events(args.days))
    qualified, aggs = analyze_events(
        events, min_edge=min_edge, min_odds=min_odds, devig_method=devig_method
    )
    print(
        render_report(
            qualified,
            aggs,
            min_events=args.min_events,
            min_edge=min_edge,
            devig_method=devig_method,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
