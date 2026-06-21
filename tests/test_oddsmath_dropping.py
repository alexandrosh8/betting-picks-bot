"""Parser for oddsmath.com dropping-odds JSON (VIEW-ONLY feature).

Attributed per-book drops (open->current), informational only — never enters
devig/edge/CLV/persistence. Pinned against a real 2-event sample.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.ingestion.oddsmath_dropping import (
    DropRow,
    fetch_oddsmath_drops,
    parse_oddsmath_drops,
)

_SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures" / "oddsmath_dropping_sample.json").read_text(
        encoding="utf-8"
    )
)


def test_parse_picks_most_dropped_outcome_per_fixture() -> None:
    rows = parse_oddsmath_drops(_SAMPLE, "SBOBET")
    assert len(rows) == 2
    r = rows[0]
    assert isinstance(r, DropRow)
    assert r.book == "SBOBET"
    assert r.sport == "soccer"
    assert r.league == "FIFA - World Cup 2026"
    assert r.match == "Netherlands — Tunisia"
    assert r.market == "1X2"
    assert r.selection == "1"  # most-negative dropping%
    assert r.open_odds == 1.32
    assert r.current_odds == 1.14
    assert r.drop_pct == -13.64
    assert r.kickoff_utc == datetime(2026, 6, 25, 23, 5, tzinfo=UTC)
    # second fixture: the away side (2) dropped hardest
    r2 = rows[1]
    assert (r2.selection, r2.open_odds, r2.current_odds, r2.drop_pct) == ("2", 5.0, 3.57, -28.6)


def test_parse_skips_non_dict_empty_and_rose_only() -> None:
    assert parse_oddsmath_drops({}, "X") == []
    assert parse_oddsmath_drops({"schema": ["1", "X", "2"], "data": {}}, "X") == []
    rose = {
        "schema": ["1", "X", "2"],
        "data": {
            "e": {
                "hometeam": "A",
                "awayteam": "B",
                "first": {"1": 2.0},
                "last": {"1": 2.2},
                "dropping%": {"1": 10.0},  # rose, not a drop
            }
        },
    }
    assert parse_oddsmath_drops(rose, "X") == []


async def test_fetch_merges_books_and_sorts_by_biggest_drop() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SAMPLE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await fetch_oddsmath_drops(client, providers=(8, 32), top=10)
    # 2 books x 2 events = 4 rows, most-negative drop first
    assert len(rows) == 4
    assert rows[0].drop_pct == -28.6
    assert rows[0].selection == "2"
    assert rows[-1].drop_pct == -13.64
    assert {r.book for r in rows} == {"SBOBET", "1XBET"}


async def test_fetch_is_graceful_on_error() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert await fetch_oddsmath_drops(client, providers=(8,)) == []


def test_droprow_is_frozen() -> None:
    r = DropRow("soccer", None, "L", "A — B", "1X2", "SBOBET", "1", 2.0, 1.8, -10.0)
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass
        r.book = "X"  # type: ignore[misc]
