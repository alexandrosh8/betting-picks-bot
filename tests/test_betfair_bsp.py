"""Betfair historical STREAM loader — pure-parser tests (no network, no DB).

Covers the documented Betfair Exchange Stream market-change (``mcm``) format
(``marketDefinition`` + per-runner ``rc`` price changes): closing-price
derivation (last pre-in-play best-back, BSP preferred when reconciled),
in-play-turn detection, WINNER/LOSER settlement, .bz2 handling, the 1x2 /
moneyline runner->home/draw/away mapping, and the join that attaches the
Betfair sharp close to football-data pre-match rows. All fixtures are SYNTHETIC
(the live data is account-gated / operator-placed), built to the documented
schema. No bets are ever placed.
"""

from __future__ import annotations

import bz2
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from app.ingestion.betfair_bsp import (
    DRAW_SELECTION_ID,
    BetfairMarketClose,
    attach_betfair_close,
    home_draw_away_close,
    load_betfair_dir,
    parse_market_stream,
)
from app.resolution.matching import default_aliases

HOME_ID = 111
AWAY_ID = 222
MARKET_TIME = "2026-06-28T18:00:00.000Z"


def _mcm(pt: int, *, market_def: dict | None = None, rc: list[dict] | None = None) -> str:
    mc: dict = {"id": "1.234567890"}
    if market_def is not None:
        mc["marketDefinition"] = market_def
    if rc is not None:
        mc["rc"] = rc
    return json.dumps({"op": "mcm", "clk": "AAA", "pt": pt, "mc": [mc]})


def _runner(
    sid: int, name: str, sort: int, *, status: str = "ACTIVE", bsp: float | None = None
) -> dict:
    r: dict = {"id": sid, "name": name, "sortPriority": sort, "status": status}
    if bsp is not None:
        r["bsp"] = bsp
    return r


def _market_def(
    *, in_play: bool, status: str, runners: list[dict], bsp_reconciled: bool = False
) -> dict:
    return {
        "marketType": "MATCH_ODDS",
        "eventTypeId": "1",
        "eventName": "Arsenal v Chelsea",
        "competition": {"id": "10", "name": "English Premier League"},
        "marketTime": MARKET_TIME,
        "openDate": MARKET_TIME,
        "status": status,
        "inPlay": in_play,
        "bspMarket": True,
        "bspReconciled": bsp_reconciled,
        "runners": runners,
    }


def _rc(sid: int, *, back: float | None = None, ltp: float | None = None) -> dict:
    d: dict = {"id": sid}
    if back is not None:
        d["batb"] = [[0, back, 100.0]]
    if ltp is not None:
        d["ltp"] = ltp
    return d


def _soccer_stream(*, with_bsp: bool = False) -> list[str]:
    active = [
        _runner(HOME_ID, "Arsenal", 1),
        _runner(AWAY_ID, "Chelsea", 2),
        _runner(DRAW_SELECTION_ID, "The Draw", 3),
    ]
    settled = [
        _runner(HOME_ID, "Arsenal", 1, status="WINNER", bsp=2.05 if with_bsp else None),
        _runner(AWAY_ID, "Chelsea", 2, status="LOSER", bsp=4.10 if with_bsp else None),
        _runner(DRAW_SELECTION_ID, "The Draw", 3, status="LOSER", bsp=3.55 if with_bsp else None),
    ]
    return [
        # pre-in-play prices
        _mcm(
            1_700_000_000_000,
            market_def=_market_def(in_play=False, status="OPEN", runners=active),
            rc=[
                _rc(HOME_ID, back=2.00),
                _rc(AWAY_ID, back=4.00),
                _rc(DRAW_SELECTION_ID, back=3.50),
            ],
        ),
        # last pre-in-play snapshot (this is the CLOSE when no BSP)
        _mcm(
            1_700_000_060_000,
            rc=[
                _rc(HOME_ID, back=2.10),
                _rc(AWAY_ID, back=3.80),
                _rc(DRAW_SELECTION_ID, back=3.60),
            ],
        ),
        # in-play turn — prices here and after must NOT be the close
        _mcm(
            1_700_000_120_000,
            market_def=_market_def(in_play=True, status="OPEN", runners=active),
            rc=[_rc(HOME_ID, back=1.50)],
        ),
        # settlement
        _mcm(
            1_700_000_900_000,
            market_def=_market_def(
                in_play=True, status="CLOSED", runners=settled, bsp_reconciled=with_bsp
            ),
        ),
    ]


def test_parse_close_is_last_pre_inplay_best_back() -> None:
    market = parse_market_stream(_soccer_stream(with_bsp=False))
    assert market is not None
    assert isinstance(market, BetfairMarketClose)
    assert market.market_type == "MATCH_ODDS"
    assert market.event_name == "Arsenal v Chelsea"
    assert market.competition == "English Premier League"
    assert market.settled is True
    by_id = {r.selection_id: r for r in market.runners}
    # close = last pre-in-play best-back (the 2nd message), in-play 1.50 ignored
    assert by_id[HOME_ID].close_price == Decimal("2.10")
    assert by_id[AWAY_ID].close_price == Decimal("3.80")
    assert by_id[DRAW_SELECTION_ID].close_price == Decimal("3.60")
    assert by_id[HOME_ID].won is True
    assert by_id[AWAY_ID].won is False


def test_kickoff_and_inplay_are_utc_aware() -> None:
    market = parse_market_stream(_soccer_stream())
    assert market is not None
    assert market.kickoff_utc == datetime(2026, 6, 28, 18, 0, 0, tzinfo=UTC)
    assert market.kickoff_utc.tzinfo is not None  # never naive
    assert market.in_play_utc is not None
    assert market.in_play_utc.tzinfo is not None
    # in-play turn = pt of the 3rd message
    assert market.in_play_utc == datetime.fromtimestamp(1_700_000_120_000 / 1000, tz=UTC)


def test_bsp_preferred_as_close_when_reconciled() -> None:
    market = parse_market_stream(_soccer_stream(with_bsp=True))
    assert market is not None
    by_id = {r.selection_id: r for r in market.runners}
    # BSP overrides the pre-in-play best-back as the settled sharp close
    assert by_id[HOME_ID].bsp == Decimal("2.05")
    assert by_id[HOME_ID].close_price == Decimal("2.05")
    assert by_id[AWAY_ID].close_price == Decimal("4.10")


def test_skips_non_mcm_and_blank_lines() -> None:
    lines = ["", '{"op":"status","id":1}', *_soccer_stream(), "   "]
    market = parse_market_stream(lines)
    assert market is not None
    assert len(market.runners) == 3


def test_home_draw_away_close_maps_by_name_and_draw_id() -> None:
    market = parse_market_stream(_soccer_stream())
    assert market is not None
    res = home_draw_away_close(market, "Arsenal", "Chelsea", aliases=default_aliases())
    assert res is not None
    assert res.home_close == Decimal("2.10")
    assert res.draw_close == Decimal("3.60")
    assert res.away_close == Decimal("3.80")
    assert res.result == "H"


def _basketball_stream() -> list[str]:
    active = [_runner(HOME_ID, "Boston Celtics", 1), _runner(AWAY_ID, "Miami Heat", 2)]
    settled = [
        _runner(HOME_ID, "Boston Celtics", 1, status="LOSER"),
        _runner(AWAY_ID, "Miami Heat", 2, status="WINNER"),
    ]
    bdef = {
        "marketType": "MATCH_ODDS",
        "eventTypeId": "7522",
        "eventName": "Celtics @ Heat",
        "competition": {"id": "9", "name": "NBA"},
        "marketTime": MARKET_TIME,
        "status": "OPEN",
        "inPlay": False,
        "bspMarket": False,
        "runners": active,
    }
    return [
        _mcm(
            1_700_000_000_000,
            market_def=bdef,
            rc=[_rc(HOME_ID, back=1.80), _rc(AWAY_ID, back=2.10)],
        ),
        _mcm(
            1_700_000_120_000,
            market_def={**bdef, "inPlay": True, "status": "CLOSED", "runners": settled},
        ),
    ]


def test_basketball_two_way_has_no_draw() -> None:
    market = parse_market_stream(_basketball_stream())
    assert market is not None
    assert market.event_type_id == "7522"
    res = home_draw_away_close(market, "Boston Celtics", "Miami Heat", aliases=default_aliases())
    assert res is not None
    assert res.home_close == Decimal("1.80")
    assert res.away_close == Decimal("2.10")
    assert res.draw_close is None
    assert res.result == "A"


def test_load_betfair_dir_reads_bz2_and_plain(tmp_path: Path) -> None:
    plain = tmp_path / "1.234567890.json"
    plain.write_text("\n".join(_soccer_stream()), encoding="utf-8")
    compressed = tmp_path / "1.999999999.bz2"
    compressed.write_bytes(bz2.compress("\n".join(_basketball_stream()).encode("utf-8")))
    markets = load_betfair_dir(tmp_path)
    assert len(markets) == 2
    types = {m.event_type_id for m in markets}
    assert types == {"1", "7522"}


def test_load_betfair_dir_absent_is_empty(tmp_path: Path) -> None:
    assert load_betfair_dir(tmp_path / "nope") == []


def test_attach_betfair_close_joins_and_rejects_result_mismatch() -> None:
    market = parse_market_stream(_soccer_stream())
    assert market is not None
    aliases = default_aliases()
    # fd-style pre-match rows (football-data Max soft) — one match, one mismatch
    good = {
        "Date": "28/06/2026",
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "MaxH": "2.20",
        "MaxD": "3.70",
        "MaxA": "3.90",
        "PSH": "2.05",
        "PSD": "3.50",
        "PSA": "3.70",
        "FTR": "H",
        "FTHG": "1",
        "FTAG": "0",
    }
    mismatch = {**good, "FTR": "A"}  # football-data says Away, Betfair says Home -> drop
    joined, stats = attach_betfair_close([good, mismatch], [market], aliases=aliases)
    assert stats.n_fd_rows == 2
    assert stats.n_markets == 1
    assert stats.n_joined == 1
    assert stats.n_result_conflict == 1
    assert len(joined) == 1
    row = joined[0]
    # Betfair sharp close written into the closing slots (decimal odds)
    assert float(row["PSCH"]) == 2.10
    assert float(row["PSCD"]) == 3.60
    assert float(row["PSCA"]) == 3.80
    # pre-match Max prices preserved untouched
    assert row["MaxH"] == "2.20"
