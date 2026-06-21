"""Read-only, VIEW-ONLY oddsmath.com dropping odds — per-book, INFORMATIONAL.

Mirrors oddsmath's own dropping-odds page: pick a BOOK (provider) + a time
window, see that named bookmaker's per-outcome OPENING (`first`) -> CURRENT
(`last`) move + drop%. Attributed real prices (not a blended average); times are
UTC and the drop% recomputes exactly from open->current (verified 2026-06-21).

VIEW-ONLY: never enters devig / edge / staking / CLV / persistence. Read-only
GET; fails soft to [] (the view never crashes); logs the exception TYPE only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

_ENDPOINT = "https://www.oddsmath.com/api/v1/dropping-odds.json/"
_HEADERS = {
    "user-agent": "Mozilla/5.0 (betting-ai read-only odds viewer)",
    "accept": "application/json",
}
#: oddsmath provider_id -> book name (its provider selector, 2026-06-21).
PROVIDERS: dict[int, str] = {
    8: "SBOBET",
    11: "Titanbet",
    13: "Dafabet",
    16: "Betway",
    20: "Bwin",
    32: "1XBET",
    38: "Marathonbet",
    41: "BetWinner",
    56: "Tipico",
    64: "Campobet",
    73: "NetBet",
    74: "Megapari",
    80: "Suprabets",
    81: "Winner",
    82: "Mozzart",
    83: "Bettogoal",
    84: "FEZbet",
    87: "BetInAsia",
}
DEFAULT_PROVIDER = 32  # 1XBET
#: oddsmath "Drop in the" windows (minutes) -> label, for the UI.
INTERVALS: dict[int, str] = {60: "1h", 360: "6h", 1440: "24h"}


@dataclass(frozen=True, slots=True)
class OutcomeMove:
    """One outcome's opening->current move for the selected book."""

    label: str
    open: float | None
    current: float | None
    drop_pct: float | None


@dataclass(frozen=True, slots=True)
class MatchDrop:
    """One fixture's full per-outcome moves for the selected book."""

    sport: str
    kickoff_utc: datetime | None
    league: str
    match: str
    market: str
    book: str
    outcomes: tuple[OutcomeMove, ...]
    max_drop: float  # most-negative outcome drop% (for sorting); 0.0 if none


def _to_float(raw: object) -> float | None:
    try:
        return float(raw) if raw is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_time(raw: object) -> datetime | None:
    # oddsmath returns UTC "YYYY-MM-DD HH:MM:SS" (generatedAt == UTC, verified).
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_oddsmath(payload: dict, book: str) -> list[MatchDrop]:
    """Parse one provider's dropping-odds JSON into per-fixture, per-outcome moves.
    Pure + defensive: a malformed event is skipped, never raised."""
    schema = payload.get("schema") or []
    market = "1X2" if schema == ["1", "X", "2"] else "/".join(str(s) for s in schema)
    data = payload.get("data")
    rows: list[MatchDrop] = []
    if not isinstance(data, dict):
        return rows
    for ev in data.values():
        if not isinstance(ev, dict):
            continue
        first = ev.get("first") or {}
        last = ev.get("last") or {}
        dp = ev.get("dropping%") or {}
        outcomes: list[OutcomeMove] = []
        drops: list[float] = []
        for key in schema:
            drop = _to_float(dp.get(key))
            outcomes.append(
                OutcomeMove(
                    label=str(key),
                    open=_to_float(first.get(key)),
                    current=_to_float(last.get(key)),
                    drop_pct=drop,
                )
            )
            if drop is not None:
                drops.append(drop)
        if not outcomes:
            continue
        rows.append(
            MatchDrop(
                sport="soccer",
                kickoff_utc=_parse_time(ev.get("time")),
                league=str(ev.get("league") or ev.get("league_label") or ""),
                match=f"{ev.get('hometeam', '')} — {ev.get('awayteam', '')}".strip(" —"),
                market=market,
                book=book,
                outcomes=tuple(outcomes),
                max_drop=min(drops) if drops else 0.0,
            )
        )
    return rows


async def fetch_oddsmath_book(
    client: httpx.AsyncClient,
    *,
    provider_id: int = DEFAULT_PROVIDER,
    interval: int = 1440,
    cat_id: int = 0,  # 0 = 1X2
    limit: int = 40,
    top: int = 40,
    timeout: float = 15.0,
) -> list[MatchDrop]:
    """Fetch ONE book's dropping odds (biggest drop first). [] on any failure."""
    book = PROVIDERS.get(provider_id, f"book {provider_id}")
    try:
        resp = await client.get(
            _ENDPOINT,
            params={
                "sport_type": "soccer",
                "cat_id": cat_id,
                "provider_id": provider_id,
                "interval": interval,
                "language": "en",
                "limit": limit,
            },
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as exc:  # network / HTTP / timeout — view-only, swallow
        logger.warning(
            "oddsmath book fetch failed (provider %s): %s", provider_id, type(exc).__name__
        )
        return []
    rows = parse_oddsmath(resp.json(), book)
    rows.sort(key=lambda r: r.max_drop)  # most-negative (biggest drop) first
    return rows[:top]
