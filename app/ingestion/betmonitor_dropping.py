"""Read-only, VIEW-ONLY fetch of betmonitor.com "dropping odds" — INFORMATIONAL.

NOT a pick / edge / CLV source. betmonitor's dropping-odds price is a CONSENSUS
AVERAGE across all books (live-verified 2026-06-21: it equals the mean of the
book list and matches no single book — not the sharp, not the best), it is
~5 minutes stale (300s client poll, no realtime feed) and capped at the top-20
droppers. It is rendered in a view-only dashboard tab and MUST NEVER enter
devig / edge / staking / CLV / persistence — feeding a consensus average in
would be exactly the banned "consensus-as-fill" fake-CLV.

The fetch is a read-only POST data query (betmonitor hydrates the table via
get_changes.php): no bets, no login, no writes, no stored credentials, no
anti-bot bypass (the site has none). ANY failure returns [] so the view can
never crash the app; only the exception TYPE is logged (never the URL).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_ENDPOINT = "https://www.betmonitor.com/content/get_changes.php"
_HEADERS = {
    "x-requested-with": "XMLHttpRequest",
    "user-agent": "Mozilla/5.0 (betting-ai read-only odds viewer)",
}
#: betmonitor data-bet code -> human selection label.
_BET_LABELS = {"q1": "1", "qx": "X", "q2": "2", "qo": "Over", "qu": "Under"}


@dataclass(frozen=True, slots=True)
class DroppingSelection:
    """One leg of a dropping-odds row. ``dropped`` flags the leg betmonitor
    highlighted as the mover. ``decimal_odds`` is the CONSENSUS average, not a
    book price — display only."""

    label: str
    decimal_odds: float | None
    dropped: bool


@dataclass(frozen=True, slots=True)
class DroppingRow:
    """One betmonitor dropping-odds entry, for view-only display."""

    event_id: str
    kickoff_utc: datetime | None
    kickoff_label: str
    league: str
    match: str
    market: str
    drop_pct: float | None
    selections: tuple[DroppingSelection, ...]


def _to_float(raw: str | None) -> float | None:
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _attr(tag: Tag, name: str) -> str | None:
    """Single-valued attribute as a str (bs4 returns lists for multi-valued)."""
    value = tag.get(name)
    return value if isinstance(value, str) else None


def _has_class(tag: Tag, cls: str) -> bool:
    classes = tag.get("class")
    return isinstance(classes, list) and cls in classes


def parse_dropping_odds(html: str) -> list[DroppingRow]:
    """Parse betmonitor's get_changes.php HTML fragment into view rows. Pure and
    defensive: a malformed row (missing price/selection) is skipped, never
    raised — a view-only feed degrades to fewer rows, never an error."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[DroppingRow] = []
    for cont in soup.select("div.odds-changes-cont"):
        event_id = (_attr(cont, "id") or "").split("_", 1)[0]
        ev = cont.select_one("div.evtime")
        ts = _to_float(_attr(ev, "data-value")) if ev else None
        kickoff = datetime.fromtimestamp(ts, tz=UTC) if ts is not None else None
        league_a = cont.select_one("div.league a")
        teams_a = cont.select_one("div.teams a")
        bettype = cont.select_one("div.bettype-string")
        value = cont.select_one("div.value")
        selections: list[DroppingSelection] = []
        for link in cont.select("div.odds > a.odd-link"):
            label = _BET_LABELS.get((_attr(link, "data-bet") or "").strip())
            if label is None:
                continue  # empty/unknown slot (e.g. a 2-way's blank 3rd cell)
            dec = link.select_one("span.odd-decimal")
            odds = _to_float(dec.get_text(strip=True)) if dec else None
            if odds is None:
                continue
            inner = link.select_one("div")
            dropped = inner is not None and _has_class(inner, "highlight")
            selections.append(DroppingSelection(label=label, decimal_odds=odds, dropped=dropped))
        if not selections:
            continue
        rows.append(
            DroppingRow(
                event_id=event_id,
                kickoff_utc=kickoff,
                kickoff_label=ev.get_text(" ", strip=True) if ev else "",
                league=league_a.get_text(strip=True) if league_a else "",
                match=teams_a.get_text(strip=True) if teams_a else "",
                market=bettype.get_text(strip=True) if bettype else "",
                drop_pct=_to_float(_attr(value, "data-value")) if value else None,
                selections=tuple(selections),
            )
        )
    return rows


async def fetch_dropping_odds(
    client: httpx.AsyncClient,
    *,
    market: str = "1",  # 1 = European, 2 = Asian
    time_window: str = "2",  # 0=2h, 1=4h, 2=24h
    bettype: str = "all",
    sport: str = "all",
    limit: int = 20,
    timeout: float = 15.0,
) -> list[DroppingRow]:
    """Read-only fetch + parse of betmonitor's top dropping-odds. Returns [] on
    ANY failure (the view must never crash). Logs the exception TYPE only."""
    try:
        resp = await client.post(
            _ENDPOINT,
            data={
                "market": market,
                "time": time_window,
                "bettype": bettype,
                "sport": sport,
                "limit": str(limit),
            },
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as exc:  # network / HTTP / timeout — view-only, swallow
        logger.warning("betmonitor dropping-odds fetch failed: %s", type(exc).__name__)
        return []
    return parse_dropping_odds(resp.text)
