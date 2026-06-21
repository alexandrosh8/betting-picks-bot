"""Parser for oddsmath.com per-book dropping-odds JSON (VIEW-ONLY feature).

Attributed per-outcome moves (open->current + drop%), informational only — never
enters devig/edge/CLV/persistence. Pinned against a real 2-event sample.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.ingestion.oddsmath_dropping import (
    MatchDrop,
    OutcomeMove,
    fetch_oddsmath_book,
    parse_oddsmath,
)

_SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures" / "oddsmath_dropping_sample.json").read_text(
        encoding="utf-8"
    )
)


def test_parse_per_outcome_moves() -> None:
    rows = parse_oddsmath(_SAMPLE, "SBOBET")
    assert len(rows) == 2
    r = rows[0]
    assert isinstance(r, MatchDrop)
    assert r.book == "SBOBET"
    assert r.sport == "soccer"
    assert r.league == "FIFA - World Cup 2026"
    assert r.match == "Netherlands — Tunisia"
    assert r.market == "1X2"
    assert r.kickoff_utc == datetime(2026, 6, 25, 23, 5, tzinfo=UTC)
    assert [(o.label, o.open, o.current, o.drop_pct) for o in r.outcomes] == [
        ("1", 1.32, 1.14, -13.64),
        ("X", 5.1, 8.0, 56.86),
        ("2", 7.5, 17.0, 126.67),
    ]
    assert r.max_drop == -13.64  # most-negative outcome, used for sorting


def test_parse_second_fixture_max_drop() -> None:
    g = parse_oddsmath(_SAMPLE, "SBOBET")[1]
    assert g.match == "Germany — Ecuador"
    two = next(o for o in g.outcomes if o.label == "2")
    assert (two.open, two.current, two.drop_pct) == (5.0, 3.57, -28.6)
    assert g.max_drop == -28.6


def test_parse_empty_and_non_dict() -> None:
    assert parse_oddsmath({}, "X") == []
    assert parse_oddsmath({"schema": ["1", "X", "2"], "data": {}}, "X") == []


async def test_fetch_book_sorts_by_biggest_drop() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SAMPLE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await fetch_oddsmath_book(client, provider_id=32)
    assert [r.book for r in rows] == ["1XBET", "1XBET"]  # provider 32
    assert rows[0].max_drop == -28.6  # Germany (biggest) sorts first
    assert rows[1].max_drop == -13.64


async def test_fetch_is_graceful_on_error() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert await fetch_oddsmath_book(client, provider_id=32) == []


def test_outcome_move_is_frozen() -> None:
    o = OutcomeMove("1", 2.0, 1.8, -10.0)
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass
        o.label = "X"  # type: ignore[misc]
