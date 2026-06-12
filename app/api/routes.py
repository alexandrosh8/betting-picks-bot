"""API routes: latest picks, manual result tracking, health.

POST /picks/{id}/result is the MANUAL result-tracking entrypoint — the user
records what THEY did (bet placed or not, stake, outcome). Nothing here can
place a bet.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.backtesting.live_evidence import live_evidence_report
from app.schemas.events import EventResultIn, ResultIn
from app.settlement.engine import settle_event_picks
from app.storage.models import Event, ManualBetLog, Pick, ResultTracking
from app.storage.repositories import (
    latest_picks_with_events,
    live_evidence_rows,
    performance_report,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Self-contained dashboard page (no build step, no CDN — works offline and
# identically on the Ubuntu VPS). Data is fetched from /picks client-side.
_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> str:
    return _DASHBOARD_HTML


@router.get("/health")
async def health() -> dict[str, Any]:
    from app.config import get_settings
    from app.maintenance.upstream_watch import LAST_CHECK
    from app.pipeline import LAST_POLL

    return {
        "status": "ok",
        "mode": "picks-only",
        "upstream": LAST_CHECK,
        "polls": LAST_POLL,
        # The dashboard derives its "verified within" window from the actual
        # poll cadence (max(45min, 3 * interval)) instead of hardcoding it.
        "poll_interval_seconds": get_settings().poll_interval_seconds,
    }


@router.get("/picks")
async def latest_picks(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    tier: Annotated[str | None, Query(pattern="^(premium|volume)$")] = None,
) -> list[dict[str, Any]]:
    """Latest picks, newest first. `tier` scopes the window server-side —
    the volume shadow tier runs ~6x premium volume, so an unscoped
    latest-200 window would fill with volume rows and hide open premium
    picks entirely (the dashboard fetches each tier separately).
    None = both tiers (legacy feed).

    `min_acceptable_odds` per row is the execution helper: the minimum
    displayed odds at which the pick still retains the premium edge floor
    ("still +EV down to X.XX" on the dashboard)."""
    from app.config import get_settings

    return await latest_picks_with_events(
        session, limit, tier=tier, min_edge=get_settings().value_min_edge
    )


@lru_cache(maxsize=1)
def _ml_operating_point() -> float | None:
    """The configured value-filter manifest's frozen q* (None = no artifact).

    Cached for the process lifetime: artifacts only change at deploy, and a
    per-request disk read would be blocking IO in the event loop. Reports
    accept ANY manifest verdict — stratifying shadow scores is annotation,
    never enforcement (demotion keeps ValueFilterModel.load's ADOPT gate).
    """
    from app.config import get_settings
    from app.models.value_filter import manifest_operating_point

    settings = get_settings()
    return manifest_operating_point(
        Path(settings.value_ml_model_dir), settings.value_ml_manifest_filename
    )


@router.get("/performance")
async def performance(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """ROI + stake-weighted log-CLV over settled picks (phase 4 report).

    Headline fields are PREMIUM-tier scoped ("tier_scope": "premium"); the
    volume shadow tier's aggregates ride under "volume" so its many small
    edges can never distort the alerted strategy's numbers.

    "live_evidence" stratifies the settled picks by ML score bucket
    (q* from the configured manifest), tier, and — once the column lands —
    anchor type: the accumulating live instrument for the VALUE_ML_FILTER
    flip decision. Every stratum carries its n; strata under min_n are
    flagged insufficient and the dashboard shows the state, not estimates.
    """
    report = await performance_report(session)
    rows = await live_evidence_rows(session)
    report["live_evidence"] = live_evidence_report(rows, ml_threshold=_ml_operating_point())
    return report


@router.post("/events/{event_id}/result")
async def settle_event(
    event_id: int,
    payload: EventResultIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, int]:
    """Settle ALL open picks of an event from a user-entered final score.

    Manual settlement path (dashboard settle button) — records outcomes
    only; nothing here can place a bet.
    """
    event = await session.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    settled, skipped = await settle_event_picks(
        session, event_id, payload.home_score, payload.away_score, datetime.now(tz=UTC)
    )
    await session.commit()
    return {"settled": settled, "skipped": skipped}


@router.post("/picks/{pick_id}/result", status_code=201)
async def record_result(
    pick_id: int,
    payload: ResultIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    pick = await session.get(Pick, pick_id)
    if pick is None:
        raise HTTPException(status_code=404, detail="pick not found")

    pnl: Decimal | None = None
    roi: Decimal | None = None
    if payload.bet_placed and payload.actual_stake is not None:
        odds = payload.actual_odds or float(pick.decimal_odds)
        if payload.outcome == "won":
            pnl = payload.actual_stake * Decimal(str(odds - 1.0))
        elif payload.outcome == "lost":
            pnl = -payload.actual_stake
        else:  # void / push: stake returned
            pnl = Decimal("0.00")
        if payload.actual_stake > 0:
            roi = pnl / payload.actual_stake

    await session.execute(
        insert(ManualBetLog).values(
            pick_id=pick_id,
            bet_placed=payload.bet_placed,
            actual_stake=payload.actual_stake,
            actual_odds=payload.actual_odds,
            bookmaker_used=payload.bookmaker_used,
            notes=payload.notes,
        )
    )
    await session.execute(
        insert(ResultTracking).values(
            pick_id=pick_id,
            outcome=str(payload.outcome),
            pnl=pnl,
            roi=roi,
            settled_at=payload.settled_at,
        )
    )
    await session.execute(update(Pick).where(Pick.id == pick_id).values(status="settled"))
    await session.commit()
    return {"status": "recorded", "outcome": str(payload.outcome)}
