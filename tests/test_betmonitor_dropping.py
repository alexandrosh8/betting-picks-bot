"""Parser for betmonitor.com dropping-odds fragments (VIEW-ONLY feature).

The parsed data is a CONSENSUS AVERAGE line (not a book), informational only —
it never enters devig/edge/CLV/persistence. These tests pin the parse contract
against a real 2-row sample fragment (with its inline chart history).
"""

from datetime import UTC
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
    assert r.event_id == "109600130"
    assert r.sport == "Football"  # split from the "Sport · Country · League" string
    assert r.league == "Football · Iceland · Iceland Urvalsdeild Women"
    assert r.match == "Thor KA Akureyri (W) — Throttur Reykjavik (W)"
    assert r.market == "3-Way"
    assert r.drop_pct == -22.7
    assert r.open_odds == 2.32  # first chart point = the dropped leg's OPENING price
    assert r.kickoff_utc is not None and r.kickoff_utc.tzinfo == UTC
    assert [(s.label, s.decimal_odds, s.dropped) for s in r.selections] == [
        ("1", 4.82, False),
        ("X", 4.31, False),
        ("2", 1.52, True),
    ]


def test_parse_two_way_row_skips_empty_slot() -> None:
    r = parse_dropping_odds(_SAMPLE)[1]
    assert r.event_id == "109600476"
    assert r.sport == "Tennis"
    assert r.market == "Match Winner"
    assert r.drop_pct == -16.1
    assert r.open_odds == 3.7
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
    assert [r.event_id for r in rows] == ["109600130", "109600476"]


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
