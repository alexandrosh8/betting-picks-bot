"""Read-only loader for jokecamp/FootballData (MIT) — `all.csv`, 22 countries.

A NAMED **Pinnacle** 1X2 close (betexplorer last-displayed, close-GRADE) for many
leagues football-data.co.uk does not cover, frozen 2004-2016. GET-only: a static
MIT historical CSV on GitHub raw — NEVER an order venue, NEVER places bets.

The file is HEADERLESS with 18 positional columns; "None" marks a missing value
and only ~86k of 157k rows carry Pinnacle 1X2 (cols 11-13). There is NO soft
best-price column, so this validates the SHARP ANCHOR's calibration on an
independent sample — it is NOT a sharp-vs-soft ROI source on its own.

Positional columns: 0 id · 1 country · 2 league · 3 home · 4 away · 5 round ·
6 url · 7 awa · 8 home_score · 9 away_score · 10 date(YYYY-MM-DD) ·
11 pinnacle_home · 12 pinnacle_draw · 13 pinnacle_away · 14-17 Asian-handicap.

The loader is INJECTED so tests run offline; `_default_loader` streams the raw CSV.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Awaitable, Callable, Sequence
from datetime import date

from app.schemas.base import InternalModel

_NONE = "None"
_RAW_URL = (
    "https://raw.githubusercontent.com/jokecamp/FootballData/master/"
    "Football-results%20(22%20countries)/all.csv"
)


class JokecampMatch(InternalModel):
    """One match with a close-grade Pinnacle 1X2 line + final score. Frozen,
    extra=forbid. `match_date` is a date (the source carries no kickoff time)."""

    country: str
    league: str
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    pinnacle_home: float
    pinnacle_draw: float
    pinnacle_away: float


def _num(v: str) -> float | None:
    if v == _NONE or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_jokecamp_rows(rows: Sequence[Sequence[str]]) -> list[JokecampMatch]:
    """Map headerless positional rows to JokecampMatch, skipping rows without a
    Pinnacle 1X2 line, a score, or a parseable date. Pure: no IO, no network."""
    out: list[JokecampMatch] = []
    for r in rows:
        if len(r) < 14:
            continue
        ph, pd_, pa = _num(r[11]), _num(r[12]), _num(r[13])
        hs, as_ = _num(r[8]), _num(r[9])
        if ph is None or pd_ is None or pa is None or hs is None or as_ is None:
            continue
        try:
            match_date = date.fromisoformat(r[10])
        except ValueError:
            continue
        out.append(
            JokecampMatch(
                country=r[1],
                league=r[2],
                match_date=match_date,
                home_team=r[3],
                away_team=r[4],
                home_score=int(hs),
                away_score=int(as_),
                pinnacle_home=ph,
                pinnacle_draw=pd_,
                pinnacle_away=pa,
            )
        )
    return out


JokecampLoader = Callable[[], Awaitable[Sequence[Sequence[str]]]]


async def load_jokecamp_matches(*, loader: JokecampLoader | None = None) -> list[JokecampMatch]:
    """Read-only: fetch + parse the jokecamp all.csv. GET-only. The loader is
    injected so tests run offline; the default streams the raw CSV."""
    load = loader or _default_loader
    rows = await load()
    return parse_jokecamp_rows(rows)


async def _default_loader() -> Sequence[Sequence[str]]:  # pragma: no cover - network
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(_RAW_URL)
        resp.raise_for_status()
        text = resp.text
    return list(csv.reader(io.StringIO(text)))
