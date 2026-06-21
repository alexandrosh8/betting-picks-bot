"""Parser for betmonitor.com dropping-odds fragments (VIEW-ONLY feature).

The parsed data is a CONSENSUS AVERAGE line (not a book), informational only —
it never enters devig/edge/CLV/persistence. These tests pin the parse contract
against a real (script-stripped) 2-row sample fragment.
"""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.ingestion.betmonitor_dropping import (
    DroppingRow,
    DroppingSelection,
    fetch_dropping_odds,
    parse_dropping_odds,
)

_SAMPLE = (Path(__file__).parent / "fixtures" / "betmonitor_dropping_sample.html").read_text(
    encoding="utf-8"
)


def test_parse_three_way_row() -> None:
    rows = parse_dropping_odds(_SAMPLE)
    assert len(rows) == 2
    r = rows[0]
    assert isinstance(r, DroppingRow)
    assert r.event_id == "109599057"
    assert r.league == "Football · Peru · Peru Copa de la Liga"
    assert r.match == "CD Estudiantil CNI — Sporting Cristal"
    assert r.market == "3-Way"
    assert r.drop_pct == -24.0
    assert r.kickoff_utc == datetime.fromtimestamp(1782073800, tz=UTC)
    assert [(s.label, s.decimal_odds, s.dropped) for s in r.selections] == [
        ("1", 2.09, True),
        ("X", 3.18, False),
        ("2", 3.17, False),
    ]


def test_parse_two_way_row_skips_empty_slot() -> None:
    r = parse_dropping_odds(_SAMPLE)[1]
    assert r.event_id == "109600476"
    assert r.market == "Match Winner"
    assert r.drop_pct == -16.1
    # the empty third 'q ' slot carries no price -> dropped from the parse
    assert [(s.label, s.decimal_odds, s.dropped) for s in r.selections] == [
        ("1", 1.54, False),
        ("2", 2.32, True),
    ]


def test_parse_empty_or_garbage_returns_empty() -> None:
    assert parse_dropping_odds("") == []
    assert parse_dropping_odds("<div>nothing here</div>") == []


async def test_fetch_parses_transport_response() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_SAMPLE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await fetch_dropping_odds(client)
    assert [r.event_id for r in rows] == ["109599057", "109600476"]


async def test_fetch_is_graceful_on_error() -> None:
    # A view-only feed must NEVER crash the dashboard — a transport/HTTP failure
    # returns [] (the tab shows "unavailable"), never raises.
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert await fetch_dropping_odds(client) == []


def test_selection_is_frozen() -> None:
    s = DroppingSelection(label="1", decimal_odds=2.0, dropped=True)
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass / attribute error
        s.label = "X"  # type: ignore[misc]
