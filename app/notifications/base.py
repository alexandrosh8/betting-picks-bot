"""Alert model and sink protocol.

Every alert ends with the manual-betting reminder — alerts inform a human
decision; nothing here (or anywhere) places bets.
"""

import hashlib
from dataclasses import dataclass
from typing import Protocol

from app.edge.value import ceil_odds, min_acceptable_odds
from app.schemas.picks import MANUAL_BETTING_REMINDER, PickOut


@dataclass(frozen=True)
class Alert:
    pick_id: str
    title: str
    body: str
    dedupe_key: str


class AlertSink(Protocol):
    """Delivery channel. Implementations NEVER raise — they return success."""

    name: str

    async def send(self, alert: Alert) -> bool: ...


def build_pick_alert(pick: PickOut, value_min_edge: float | None = None) -> Alert:
    """Render a pick into an alert with a stable idempotency key.

    The key deliberately EXCLUDES pick_id (a fresh uuid per cycle): the same
    market state must not re-alert every poll; a price change produces a new
    key and a fresh alert.

    `value_min_edge` (the VALUE pipeline's premium threshold, passed from
    PipelineDeps) adds the execution line "Still +EV down to X.XX": the
    minimum displayed odds at which the pick retains >= that edge. VALUE-
    strategy semantics only: for value picks `model_probability` holds the
    devigged sharp fair probability (app/pipeline.py maps
    v.sharp_fair_prob there) — the model strategy must pass None, its edge
    (p_model - p_fair) does not shrink with the price the same way.
    """
    raw_key = f"{pick.event_id}|{pick.bookmaker}|{pick.market}|{pick.selection}|{pick.decimal_odds}"
    dedupe_key = hashlib.sha256(raw_key.encode()).hexdigest()[:32]
    title = f"+EV pick: {pick.event} — {pick.selection} @ {pick.decimal_odds:.2f}"
    still_ev_line: list[str] = []
    if value_min_edge is not None:
        floor = min_acceptable_odds(pick.model_probability, value_min_edge, book=pick.bookmaker)
        if floor is not None:
            still_ev_line.append(
                f"Still +EV down to: {ceil_odds(floor):.2f} at {pick.bookmaker} "
                f"(below that the edge drops under {value_min_edge:.1%} — skip)"
            )
    body = "\n".join(
        [
            title,
            f"Sport/League: {pick.sport} / {pick.league}",
            f"Market: {pick.market} | Bookmaker: {pick.bookmaker}",
            f"Model probability: {pick.model_probability:.3f}",
            f"Fair (vig-free) probability: {pick.fair_probability:.3f}",
            f"Edge: {pick.edge:+.3f} | EV: {pick.ev:+.3f}",
            f"Confidence: {pick.confidence:.2f}",
            (
                f"Recommended stake: {pick.recommended_stake_fraction:.2%} of bankroll"
                f" (~{pick.recommended_stake_amount}) — informational only"
            ),
            *still_ev_line,
            f"Odds age: {pick.odds_age_seconds:.0f}s"
            + (f" | Liquidity: {pick.liquidity}" if pick.liquidity is not None else ""),
            f"Why: {pick.reason_summary}",
            f"Generated: {pick.created_at.isoformat()}",
            pick.risk_warning,
            MANUAL_BETTING_REMINDER,
        ]
    )
    return Alert(pick_id=pick.pick_id, title=title, body=body, dedupe_key=dedupe_key)
