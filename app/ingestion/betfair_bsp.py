"""Read-only loader for free Betfair historical STREAM data — the SHARP CLOSE.

WHAT THIS IS. Betfair publishes historical exchange data at
``historicdata.betfair.com`` in the proprietary **stream** format: one
newline-delimited-JSON file per market (``.bz2``-compressed), where each line is
a Betfair Exchange Stream **market-change message** (``op == "mcm"``). The Basic
(free) tier covers settled markets but is ACCOUNT-GATED — the data is therefore
OPERATOR-PLACED on disk (same pattern as ``beatthebookie_series.py``); a live
fetch from this sandbox returns HTTP 401.

This loader is GET/READ-only over those static files. It NEVER authenticates, it
NEVER places a bet, and it stores no credentials. ``load_betfair_dir`` reads a
directory of operator-placed market files; ``parse_market_stream`` turns one
market's message sequence into a :class:`BetfairMarketClose`.

HONEST SCOPE — BSP IS A *CLOSE*, NOT A PRE-MATCH BET PRICE. A Betfair Starting
Price (BSP) / last-pre-in-play price is the SHARP CLOSING reference and the
settled result. It is NOT a price you could have bet pre-match in this dataset,
so it is joined (by date + team-name match, via ``app/resolution/matching``) to a
PRE-MATCH source (football-data Max soft) that the backtest already loads. The
backtest then measures CLV of the pre-match bet vs this real sharp close — the
sharp-anchor complement to the consensus-anchored BeatTheBookie breadth loader.

DOCUMENTED FORMAT (Betfair Exchange Stream API ``MarketChange`` message; the
public schema is mirrored by the betfair-datascientists historic-data docs).
Fields used — no invented fields, READ-ONLY (this loader parses files; it has no
client and never connects):

* message:        ``{"op":"mcm", "pt": <publish-time epoch MILLIS, UTC>,
                     "mc": [ <market change> ]}``
* market change:  ``{"id": "<marketId>", "marketDefinition": {...}, "rc": [...]}``
* marketDefinition: ``marketType`` ("MATCH_ODDS"), ``eventTypeId``
                    ("1"=Soccer, "7522"=Basketball), ``eventName``,
                    ``competition`` {``name``}, ``marketTime`` (ISO-8601 ``Z``
                    UTC scheduled start = kickoff), ``status``
                    ("OPEN"/"SUSPENDED"/"CLOSED"), ``inPlay`` (bool),
                    ``bspMarket`` / ``bspReconciled`` (bool),
                    ``runners`` [ runner definition ]
* runner def:     ``id`` (selectionId), ``name``, ``sortPriority``, ``status``
                  ("ACTIVE"/"WINNER"/"LOSER"/"REMOVED"), ``bsp`` (reconciled SP)
* runner change:  ``id`` (selectionId), ``ltp`` (last traded price), ``batb``
                  (best available to back ladder: ``[[level, price, size], ...]``),
                  ``bdatb`` (best display available to back) — same shape.

CLOSE DERIVATION. Walk the messages in order tracking each runner's best-back
(``batb``/``bdatb`` level 0, falling back to ``ltp``). When ``marketDefinition``
first flips ``inPlay: true`` we SNAPSHOT the prices accumulated from all PRIOR
messages — that is the last pre-in-play price. The final settled
``marketDefinition`` carries WINNER/LOSER and (for BSP markets) ``bsp``. The
per-runner CLOSE is the reconciled ``bsp`` when present, else the pre-in-play
snapshot, else the last seen price.
"""

from __future__ import annotations

import bz2
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.resolution.matching import AliasTable

logger = logging.getLogger(__name__)

# Betfair event-type ids (stable across the platform).
SOCCER_EVENT_TYPE_ID = "1"
BASKETBALL_EVENT_TYPE_ID = "7522"

# "The Draw" has a FIXED selection id across every soccer MATCH_ODDS market, so
# the draw runner is identifiable even when a Basic-tier file omits runner names.
DRAW_SELECTION_ID = 58805


@dataclass(frozen=True, slots=True)
class BetfairRunner:
    """One market runner with its settled status and CLOSING price (Decimal).

    ``close_price`` is the reconciled ``bsp`` if present, otherwise the last
    pre-in-play best-back price. ``won`` is True for WINNER, False for LOSER,
    None for any non-settled / removed status."""

    selection_id: int
    name: str | None
    sort_priority: int | None
    status: str  # ACTIVE | WINNER | LOSER | REMOVED | ""
    close_price: Decimal | None
    bsp: Decimal | None
    won: bool | None


@dataclass(frozen=True, slots=True)
class BetfairMarketClose:
    """One settled market: definition + per-runner sharp close and result."""

    market_id: str
    event_type_id: str | None
    event_name: str | None
    competition: str | None
    market_type: str | None
    kickoff_utc: datetime | None  # marketTime, tz-aware UTC (never naive)
    in_play_utc: datetime | None  # pt at the in-play turn, tz-aware UTC
    settled: bool
    bsp_reconciled: bool
    runners: tuple[BetfairRunner, ...]


def _to_decimal(value: object) -> Decimal | None:
    """Odds/price -> Decimal, or None for missing/<=1.0/garbage. Goes through
    ``str`` so float artefacts never leak into the boundary value."""
    if value is None:
        return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return dec if dec > 1 else None


def _parse_market_time(raw: object) -> datetime | None:
    """ISO-8601 ``marketTime`` -> tz-aware UTC datetime. Trailing ``Z`` is
    normalised to ``+00:00``; a naive value is assumed UTC (never left naive)."""
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _epoch_ms_to_utc(pt: object) -> datetime | None:
    if not isinstance(pt, (int, float)):
        return None
    return datetime.fromtimestamp(pt / 1000.0, tz=UTC)


def _best_back(rc: dict) -> Decimal | None:
    """Best available-to-back price from a runner change: ``bdatb`` (display)
    then ``batb``, level-0 ``[level, price, size]`` -> price; else ``ltp``."""
    for key in ("bdatb", "batb"):
        ladder = rc.get(key)
        if isinstance(ladder, list) and ladder:
            level0 = ladder[0]
            if isinstance(level0, (list, tuple)) and len(level0) >= 2:
                price = _to_decimal(level0[1])
                if price is not None:
                    return price
    return _to_decimal(rc.get("ltp"))


def _won(status: str) -> bool | None:
    if status == "WINNER":
        return True
    if status == "LOSER":
        return False
    return None


def parse_market_stream(lines: Iterable[str]) -> BetfairMarketClose | None:
    """Parse one market's ``mcm`` message sequence into a BetfairMarketClose.

    Returns None if no usable ``marketDefinition`` with runners is seen. Prices
    accumulate across messages; the pre-in-play snapshot is taken the first time
    ``inPlay`` flips true (using the marketDefinition BEFORE that message's own
    ``rc`` is applied, so it is genuinely the last pre-in-play price)."""
    market_id: str | None = None
    latest_def: dict | None = None
    running: dict[int, Decimal] = {}
    pre_inplay: dict[int, Decimal] | None = None
    in_play_utc: datetime | None = None

    for line in lines:
        if not line or not line.strip():
            continue
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            logger.warning("skip betfair stream line: not JSON")
            continue
        if not isinstance(msg, dict) or msg.get("op") != "mcm":
            continue
        pt = msg.get("pt")
        for mc in msg.get("mc", []) or []:
            if not isinstance(mc, dict):
                continue
            if mc.get("id"):
                market_id = str(mc["id"])
            mdef = mc.get("marketDefinition")
            if isinstance(mdef, dict):
                latest_def = mdef
                # in-play turn: snapshot prices from PRIOR messages once only.
                if mdef.get("inPlay") is True and pre_inplay is None:
                    pre_inplay = dict(running)
                    in_play_utc = _epoch_ms_to_utc(pt)
            for rc in mc.get("rc", []) or []:
                if not isinstance(rc, dict):
                    continue
                sid = rc.get("id")
                if not isinstance(sid, int):
                    continue
                price = _best_back(rc)
                if price is not None:
                    running[sid] = price

    if latest_def is None:
        return None
    raw_runners = latest_def.get("runners")
    if not isinstance(raw_runners, list) or not raw_runners:
        return None

    snapshot = pre_inplay if pre_inplay is not None else running
    runners: list[BetfairRunner] = []
    for rd in raw_runners:
        if not isinstance(rd, dict):
            continue
        sid = rd.get("id")
        if not isinstance(sid, int):
            continue
        status = str(rd.get("status") or "")
        bsp = _to_decimal(rd.get("bsp"))
        close = bsp if bsp is not None else snapshot.get(sid)
        sort_priority = rd.get("sortPriority")
        runners.append(
            BetfairRunner(
                selection_id=sid,
                name=(rd.get("name") if isinstance(rd.get("name"), str) else None),
                sort_priority=(sort_priority if isinstance(sort_priority, int) else None),
                status=status,
                close_price=close,
                bsp=bsp,
                won=_won(status),
            )
        )
    if not runners:
        return None
    runners.sort(
        key=lambda r: (r.sort_priority if r.sort_priority is not None else 99, r.selection_id)
    )

    competition = None
    comp = latest_def.get("competition")
    if isinstance(comp, dict) and isinstance(comp.get("name"), str):
        competition = comp["name"]

    def _str_field(key: str) -> str | None:
        value = latest_def.get(key)
        return value if isinstance(value, str) else None

    return BetfairMarketClose(
        market_id=market_id or "",
        event_type_id=_str_field("eventTypeId"),
        event_name=_str_field("eventName"),
        competition=competition,
        market_type=_str_field("marketType"),
        kickoff_utc=_parse_market_time(latest_def.get("marketTime")),
        in_play_utc=in_play_utc,
        settled=str(latest_def.get("status") or "") == "CLOSED",
        bsp_reconciled=bool(latest_def.get("bspReconciled")),
        runners=tuple(runners),
    )


@dataclass(frozen=True, slots=True)
class HdaClose:
    """Home/Draw/Away closing prices + result mapped to a (home, away) pick.

    ``draw_close`` is None for 2-way (basketball moneyline) markets. ``result``
    is H/D/A (D only possible for 3-way) from the WINNER runner."""

    home_close: Decimal | None
    draw_close: Decimal | None
    away_close: Decimal | None
    result: str | None  # "H" | "D" | "A" | None


def home_draw_away_close(
    market: BetfairMarketClose,
    home: str,
    away: str,
    *,
    aliases: AliasTable,
) -> HdaClose | None:
    """Map a market's runners to the (home, away) pick orientation.

    Home/away runners are matched by canonical (alias-resolved) name; the draw
    runner is the fixed ``DRAW_SELECTION_ID`` (or any runner canonicalising to
    "draw"). Returns None if either side cannot be uniquely identified — a wrong
    orientation would attach a wrong close (fake CLV, the cardinal sin)."""
    target_home = aliases.canonical(home)
    target_away = aliases.canonical(away)
    if not target_home or not target_away or target_home == target_away:
        return None

    home_runner: BetfairRunner | None = None
    away_runner: BetfairRunner | None = None
    draw_runner: BetfairRunner | None = None
    for r in market.runners:
        if r.selection_id == DRAW_SELECTION_ID or (
            r.name is not None and aliases.canonical(r.name) == "draw"
        ):
            draw_runner = r
            continue
        if r.name is None:
            continue
        canon = aliases.canonical(r.name)
        if canon == target_home:
            if home_runner is not None:
                return None  # ambiguous
            home_runner = r
        elif canon == target_away:
            if away_runner is not None:
                return None
            away_runner = r
    if home_runner is None or away_runner is None:
        return None

    result: str | None = None
    if home_runner.won:
        result = "H"
    elif away_runner.won:
        result = "A"
    elif draw_runner is not None and draw_runner.won:
        result = "D"
    return HdaClose(
        home_close=home_runner.close_price,
        draw_close=draw_runner.close_price if draw_runner is not None else None,
        away_close=away_runner.close_price,
        result=result,
    )


def load_betfair_dir(path: Path) -> list[BetfairMarketClose]:
    """Read-only: parse every operator-placed market file in ``path``.

    Reads ``*.bz2`` (compressed) and plain ``*.json`` / ``*.jsonl`` / ``*.txt``
    (one market per file). Unparseable files are skipped with a log line. An
    absent directory returns ``[]`` (the caller prints the operator instruction).
    Sorted by (kickoff, market_id) for determinism."""
    if not path.is_dir():
        return []
    markets: list[BetfairMarketClose] = []
    patterns = ("*.bz2", "*.json", "*.jsonl", "*.txt")
    for f in sorted({p for pat in patterns for p in path.glob(pat)}):
        try:
            if f.suffix == ".bz2":
                text = bz2.decompress(f.read_bytes()).decode("utf-8", errors="replace")
            else:
                text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            logger.warning("skip betfair file %s: %s", f.name, type(exc).__name__)
            continue
        market = parse_market_stream(text.splitlines())
        if market is not None and market.runners:
            markets.append(market)
    markets.sort(key=lambda m: (m.kickoff_utc or datetime.min.replace(tzinfo=UTC), m.market_id))
    return markets


@dataclass(frozen=True, slots=True)
class JoinStats:
    """Data-quality counters for the football-data <- Betfair-close join."""

    n_fd_rows: int
    n_markets: int
    n_joined: int
    n_unmatched: int
    n_result_conflict: int


def _fd_date_to_utc(raw: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def attach_betfair_close(
    fd_rows: list[dict],
    markets: list[BetfairMarketClose],
    *,
    aliases: AliasTable,
    max_day_drift: int = 1,
) -> tuple[list[dict], JoinStats]:
    """Join Betfair sharp closes onto football-data pre-match rows.

    For each fd row (HomeTeam/AwayTeam/Date + pre-match Max/PS odds) find the
    UNIQUE Betfair market with the same canonical teams within ``max_day_drift``
    days (``app/resolution/matching.match_event``, ordered, strict — never
    guesses), then OVERWRITE the closing slots (``PSCH/PSCD/PSCA`` and
    ``MaxCH/MaxCD/MaxCA``) with the Betfair close decimal odds. Pre-match columns
    are left untouched. A row is DROPPED (data-quality gate) when the Betfair
    settled result disagrees with the football-data ``FTR`` — a mismatch means a
    bad join or a bad result and must not silently corrupt CLV.

    Returns (joined_rows, stats). Only successfully-joined rows are returned, so
    every returned row carries a genuine sharp close.
    """
    from app.resolution.matching import EventCandidate, match_event

    candidates: list[EventCandidate] = []
    by_ref: dict[str, BetfairMarketClose] = {}
    for m in markets:
        if m.kickoff_utc is None:
            continue
        # home/away = the two non-draw runners, ordered by sortPriority (Betfair
        # lists the home side first in soccer MATCH_ODDS).
        non_draw = [
            r for r in m.runners if r.selection_id != DRAW_SELECTION_ID and r.name is not None
        ]
        if len(non_draw) < 2:
            continue
        home_r, away_r = non_draw[0], non_draw[1]
        ref = m.market_id or f"mkt-{len(by_ref)}"
        by_ref[ref] = m
        candidates.append(
            EventCandidate(
                ref=ref, home=home_r.name or "", away=away_r.name or "", kickoff=m.kickoff_utc
            )
        )

    joined: list[dict] = []
    n_unmatched = 0
    n_conflict = 0
    for row in fd_rows:
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        kickoff = _fd_date_to_utc(row.get("Date") or "")
        if not home or not away or kickoff is None:
            n_unmatched += 1
            continue
        match = match_event(
            home, away, kickoff, candidates, aliases=aliases, max_day_drift=max_day_drift
        )
        if match is None:
            n_unmatched += 1
            continue
        close = home_draw_away_close(by_ref[match.ref], home, away, aliases=aliases)
        if close is None or close.home_close is None or close.away_close is None:
            n_unmatched += 1
            continue
        # Data-quality gate: Betfair result must agree with football-data FTR.
        ftr = (row.get("FTR") or "").strip()
        if close.result is not None and ftr in ("H", "D", "A") and close.result != ftr:
            n_conflict += 1
            continue
        new_row = dict(row)
        new_row["PSCH"] = str(close.home_close)
        new_row["PSCA"] = str(close.away_close)
        new_row["MaxCH"] = str(close.home_close)
        new_row["MaxCA"] = str(close.away_close)
        if close.draw_close is not None:
            new_row["PSCD"] = str(close.draw_close)
            new_row["MaxCD"] = str(close.draw_close)
        new_row["BetfairMarketId"] = match.ref
        joined.append(new_row)

    return joined, JoinStats(
        n_fd_rows=len(fd_rows),
        n_markets=len(markets),
        n_joined=len(joined),
        n_unmatched=n_unmatched,
        n_result_conflict=n_conflict,
    )


__all__ = [
    "BASKETBALL_EVENT_TYPE_ID",
    "DRAW_SELECTION_ID",
    "SOCCER_EVENT_TYPE_ID",
    "BetfairMarketClose",
    "BetfairRunner",
    "HdaClose",
    "JoinStats",
    "attach_betfair_close",
    "home_draw_away_close",
    "load_betfair_dir",
    "parse_market_stream",
]
