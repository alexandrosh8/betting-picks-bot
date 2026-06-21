"""Read-only, VIEW-ONLY fetch of oddsmath.com "dropping odds" — INFORMATIONAL.

Replaces the betmonitor consensus feed: oddsmath exposes ATTRIBUTED per-book
drops via a clean JSON API (verified 2026-06-21) — each row is one named book's
OPENING (`first`) -> CURRENT (`last`) odds per outcome, so a drop is traceable
to a real bookmaker (e.g. "1XBET 5.00 -> 3.57"), not a blended average.

Still VIEW-ONLY: it is rendered in a dashboard tab and MUST NEVER enter
devig / edge / staking / CLV / persistence. The fetch is a read-only GET data
query (no bets, no login, no credentials, no anti-bot bypass). ANY failure
yields [] so the view can never crash the app; only the exception TYPE is logged.

The endpoint REQUIRES a provider_id (a specific book) + cat_id, so we fetch a
curated set of major books concurrently and merge their drops, each tagged with
its book name.
"""

from __future__ import annotations

import asyncio
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
#: oddsmath provider_id -> book name (from its provider selector, 2026-06-21).
_PROVIDERS: dict[int, str] = {
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
#: Curated major books fetched + merged for the widget.
_DEFAULT_PROVIDERS: tuple[int, ...] = (32, 38, 20, 16, 8, 56, 87, 13)


@dataclass(frozen=True, slots=True)
class DropRow:
    """One attributed dropping-odds entry (one named book's biggest drop on a
    fixture), for view-only display. Codes (1/X/2/Over/Under) are kept as-is —
    the match column already carries the team names."""

    sport: str
    kickoff_utc: datetime | None
    league: str
    match: str
    market: str
    book: str
    selection: str
    open_odds: float | None
    current_odds: float | None
    drop_pct: float | None


def _to_float(raw: object) -> float | None:
    try:
        return float(raw) if raw is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_time(raw: object) -> datetime | None:
    # oddsmath returns "YYYY-MM-DD HH:MM:SS" (treated as UTC; the frontend
    # converts to local for display, same as every other tab).
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_oddsmath_drops(payload: dict, book: str) -> list[DropRow]:
    """Parse ONE provider's dropping-odds JSON into rows — the single most-dropped
    outcome per fixture (most-negative ``dropping%``). Pure + defensive: a
    malformed/rose-only row is skipped, never raised."""
    schema = payload.get("schema") or []
    market = "1X2" if schema == ["1", "X", "2"] else "/".join(str(s) for s in schema)
    data = payload.get("data")
    rows: list[DropRow] = []
    if not isinstance(data, dict):
        return rows
    for ev in data.values():
        if not isinstance(ev, dict):
            continue
        dp = ev.get("dropping%") or {}
        moves = [(k, v) for k, v in dp.items() if isinstance(v, int | float)]
        if not moves:
            continue
        sel, pct = min(moves, key=lambda kv: kv[1])
        if pct >= 0:
            continue  # nothing actually dropped for this book
        first = ev.get("first") or {}
        last = ev.get("last") or {}
        rows.append(
            DropRow(
                sport="soccer",
                kickoff_utc=_parse_time(ev.get("time")),
                league=str(ev.get("league") or ev.get("league_label") or ""),
                match=f"{ev.get('hometeam', '')} — {ev.get('awayteam', '')}".strip(" —"),
                market=market,
                book=book,
                selection=str(sel),
                open_odds=_to_float(first.get(sel)),
                current_odds=_to_float(last.get(sel)),
                drop_pct=float(pct),
            )
        )
    return rows


async def _fetch_one(
    client: httpx.AsyncClient,
    provider_id: int,
    *,
    interval: int,
    cat_id: int,
    limit: int,
    timeout: float,
) -> list[DropRow]:
    book = _PROVIDERS.get(provider_id, f"book {provider_id}")
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
            "oddsmath drops fetch failed (provider %s): %s", provider_id, type(exc).__name__
        )
        return []
    return parse_oddsmath_drops(resp.json(), book)


async def fetch_oddsmath_drops(
    client: httpx.AsyncClient,
    *,
    providers: tuple[int, ...] = _DEFAULT_PROVIDERS,
    interval: int = 1440,  # 60=1h, 360=6h, 1440=24h
    cat_id: int = 0,
    limit: int = 20,
    top: int = 30,
    timeout: float = 15.0,
) -> list[DropRow]:
    """Fetch the curated books concurrently, merge, and return the biggest drops
    (most-negative first). Returns [] on total failure (the view never crashes)."""
    results = await asyncio.gather(
        *(
            _fetch_one(client, pid, interval=interval, cat_id=cat_id, limit=limit, timeout=timeout)
            for pid in providers
        )
    )
    rows = [r for book_rows in results for r in book_rows]
    rows.sort(key=lambda r: r.drop_pct if r.drop_pct is not None else 0.0)
    return rows[:top]
