"""Anchor-calibration report over settled picks — is the devigged fair anchor
honest? "When the anchor says X%, do those bets land ~X%?"

DOCTRINE NOTE — a DIAGNOSTIC, not the live validator. CLV stays the evaluation
currency for the line-shopping edge; this only sanity-checks the devigged sharp
anchor (Pick.model_probability under the validated value strategy — NOT
Pick.fair_probability, which is the soft price we took; see _load_rows). It is
computed over SETTLED PICKS, so it is pick-CONDITIONAL (selection-biased toward
selections where we found value) — read it as "are the anchor probs on the bets
I actually make calibrated?", never as a profit signal.
Non-binary settlements (push / void / quarter-line half_won / half_lost) have
no win/lose label and are excluded (their count is reported).

Honesty gate (binding, same floor as the live-evidence CLV strata): below
--min-n binary observations a (sub)report prints INSUFFICIENT and per-bin
counts only — no point estimates. Picks only accrue once a slate settles; rerun
as results land:

    uv run python scripts/reports/anchor_calibration.py
    uv run python scripts/reports/anchor_calibration.py --by-market --min-n 50

Read-only analysis of our own warehouse. Nothing here places bets. The scoring
math is the pure, unit-tested app/backtesting/calibration.py; the DB read and
Settings access happen only in main() — this script's composition root.
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.backtesting.calibration import (  # noqa: E402
    MIN_CALIBRATION_N,
    CalibrationObservation,
    CalibrationReport,
    calibration_report,
)

# Only binary settlements carry a win/lose label the anchor can be graded on.
_WON = "won"
_LOST = "lost"


async def _load_rows() -> tuple[list[tuple[str, str, float, str]], int]:
    """Return (market, tier, anchor_prob, outcome) for every settled pick, plus
    the count of rows dropped for a non-binary outcome (push/void/half_*).

    `anchor_prob` is **Pick.model_probability** — under the validated VALUE
    strategy (the live default) that column holds the devigged SHARP fair prob
    (v.sharp_fair_prob, app/pipeline.py:826): the anchor this diagnostic grades.
    It is NOT Pick.fair_probability, which on the value path is the soft price we
    took (1/effective_odds, v.implied_prob) — grading wins against that would
    just re-derive the +EV margin, not test anchor calibration. Caveat: under
    the non-default model strategy (PICK_STRATEGY=model, screens-only)
    model_probability is the model's own prediction; the whole app already reads
    model_probability as the value-path fair prob (the dashboard Fair% column),
    so this diagnostic adopts the same value-strategy semantics.

    tier ("premium"/"volume") lets the caller split calibration by selection
    bias — premium picks gate on edge>=VALUE_MIN_EDGE and reserve exposure;
    volume is the lower-edge shadow tier — so their anchor-prob distributions
    are not interchangeable and an aggregate conflates two populations."""
    from sqlalchemy import select

    from app.config import get_settings
    from app.database import create_engine, create_session_factory
    from app.storage.models import Pick, ResultTracking

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(Pick.market, Pick.tier, Pick.model_probability, ResultTracking.outcome).join(
                    Pick, ResultTracking.pick_id == Pick.id
                )
            )
            raw = result.all()
    finally:
        await engine.dispose()

    rows: list[tuple[str, str, float, str]] = []
    dropped = 0
    for market, tier, anchor_prob, outcome in raw:
        if outcome not in (_WON, _LOST):
            dropped += 1
            continue
        rows.append((str(market), str(tier), float(anchor_prob), str(outcome)))
    return rows, dropped


def _to_obs(rows: list[tuple[str, str, float, str]]) -> list[CalibrationObservation]:
    return [
        CalibrationObservation(fair_prob=fp, won=(oc == _WON)) for _market, _tier, fp, oc in rows
    ]


def _print_report(title: str, rep: CalibrationReport) -> None:
    print(f"\n== {title} ==")
    if rep.insufficient:
        print(f"  INSUFFICIENT DATA: n={rep.n} (< min-n) — no estimates shown.")
        return
    assert rep.log_loss is not None and rep.brier is not None  # not insufficient
    assert rep.ignorance is not None and rep.ece is not None and rep.mce is not None
    assert rep.base_rate is not None and rep.mean_pred is not None
    print(
        f"  n={rep.n}  log_loss={rep.log_loss:.4f}  ignorance={rep.ignorance:.4f} bits  "
        f"brier={rep.brier:.4f}  ECE={rep.ece:.4f}  MCE={rep.mce:.4f}"
    )
    print(f"  base_rate(observed win)={rep.base_rate:.3f}  mean_pred(anchor)={rep.mean_pred:.3f}")
    print("  reliability (predicted -> observed, n):")
    for b in rep.bins:
        if b.n == 0:
            continue
        assert b.mean_pred is not None and b.observed is not None  # n>0 -> set
        flag = "  <-- gap" if abs(b.mean_pred - b.observed) >= 0.10 else ""
        print(
            f"    [{b.lo:.1f},{b.hi:.1f})  pred={b.mean_pred:.3f}  "
            f"obs={b.observed:.3f}  n={b.n}{flag}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anchor-calibration report over settled picks (diagnostic, not a profit signal)"
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=MIN_CALIBRATION_N,
        help=(
            "binary-observation floor below which estimates are withheld "
            f"(default {MIN_CALIBRATION_N})"
        ),
    )
    parser.add_argument("--bins", type=int, default=10, help="reliability bins over [0,1]")
    parser.add_argument(
        "--by-market", action="store_true", help="also break the report down per market key"
    )
    parser.add_argument(
        "--by-tier",
        action="store_true",
        help="also break the report down per tier (premium vs volume/shadow) — "
        "they have different selection biases, so an aggregate conflates them",
    )
    args = parser.parse_args()

    rows, dropped = asyncio.run(_load_rows())

    print("Anchor calibration — devigged fair-prob vs realized outcome (DIAGNOSTIC).")
    print("Pick-conditional (only settled picks); not a profit signal; CLV remains the validator.")
    print(f"binary settled picks: {len(rows)}  (excluded non-binary push/void/half_*: {dropped})")

    overall = calibration_report(_to_obs(rows), n_bins=args.bins, min_n=args.min_n)
    _print_report("OVERALL", overall)

    if args.by_tier:
        by_tier: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
        for row in rows:
            by_tier[row[1]].append(row)  # row[1] = tier
        for tier in sorted(by_tier):
            rep = calibration_report(_to_obs(by_tier[tier]), n_bins=args.bins, min_n=args.min_n)
            _print_report(f"tier={tier}", rep)

    if args.by_market:
        by_market: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
        for row in rows:
            by_market[row[0]].append(row)  # row[0] = market
        for market in sorted(by_market):
            rep = calibration_report(_to_obs(by_market[market]), n_bins=args.bins, min_n=args.min_n)
            _print_report(f"market={market}", rep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
