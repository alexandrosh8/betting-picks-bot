"""DB integration for the cross-source resolver (compose Postgres; skip absent).

resolve_pinnacle_close_snaps strict-matches a pick's fixture to its
`pinnacle_<sport>` archive event and returns that event's close re-keyed to the
pick's event_id + selection vocabulary. These tests prove the happy path
(alias + re-key), and the cardinal-sin guards at the DB layer (no match -> [],
ambiguous -> [], out-of-window -> []). No live network.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.base import EventTeams
from app.schemas.base import Market
from app.schemas.odds import OddsSnapshotIn
from app.storage.repositories import persist_odds_snapshots, resolve_pinnacle_close_snaps

DB_URL = "postgresql+asyncpg://betting_ai:betting_ai@localhost:5433/betting_ai"
KO = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)
CAPTURED = KO - timedelta(hours=2)


@pytest.fixture
async def factory():  # type: ignore[no-untyped-def]
    engine = create_async_engine(DB_URL)
    try:
        async with engine.connect() as probe:
            await probe.exec_driver_sql("SELECT 1")
    except Exception:
        await engine.dispose()
        pytest.skip("compose Postgres not reachable on :5433")
    async with engine.connect() as conn:
        trans = await conn.begin()
        maker = async_sessionmaker(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield maker
        finally:
            await trans.rollback()
    await engine.dispose()


def _pin_snap(selection: str, odds: float, event: str) -> OddsSnapshotIn:
    return OddsSnapshotIn(
        event_id=event,
        bookmaker="Pinnacle",
        market=Market.H2H,
        selection=selection,
        decimal_odds=odds,
        captured_at=CAPTURED,
        ingested_at=CAPTURED,
    )


async def _seed_pinnacle_event(factory, ref: str, home: str, away: str) -> None:  # type: ignore[no-untyped-def]
    snaps = [_pin_snap(home, 2.10, ref), _pin_snap("Draw", 3.40, ref), _pin_snap(away, 3.60, ref)]
    teams = {ref: EventTeams(home=home, away=away, league="pin", starts_at=KO)}
    await persist_odds_snapshots(factory, snaps, teams, "pinnacle_soccer", "pinnacle_soccer")


async def test_resolver_matches_via_alias_and_rekeys_selections(factory) -> None:  # type: ignore[no-untyped-def]
    await _seed_pinnacle_event(factory, "pin-mu-che", "Manchester United", "Chelsea")
    async with factory() as session:
        out = await resolve_pinnacle_close_snaps(
            session,
            pinnacle_sport_key="pinnacle_soccer",
            pick_external_ref="evt-pick",
            home="Man Utd",  # OddsPortal-style abbreviation -> alias -> Manchester United
            away="Chelsea",
            kickoff=KO,
        )
    by_sel = {s.selection: s for s in out}
    # re-keyed to the PICK's selection vocabulary (home collapses to "Man Utd")
    assert set(by_sel) == {"Man Utd", "Draw", "Chelsea"}
    assert all(s.event_id == "evt-pick" for s in out)
    assert all(s.bookmaker == "Pinnacle" for s in out)
    assert by_sel["Man Utd"].decimal_odds == pytest.approx(2.10)
    assert by_sel["Draw"].decimal_odds == pytest.approx(3.40)


async def test_resolver_no_match_returns_empty(factory) -> None:  # type: ignore[no-untyped-def]
    await _seed_pinnacle_event(factory, "pin-alpha-beta", "Alpha", "Beta")
    async with factory() as session:
        out = await resolve_pinnacle_close_snaps(
            session,
            pinnacle_sport_key="pinnacle_soccer",
            pick_external_ref="evt-pick",
            home="Gamma",
            away="Delta",
            kickoff=KO,
        )
    assert out == []


async def test_resolver_ambiguous_archive_returns_empty(factory) -> None:  # type: ignore[no-untyped-def]
    # two archive events with the SAME teams + kickoff -> cannot disambiguate -> []
    await _seed_pinnacle_event(factory, "pin-dup-1", "Alpha", "Beta")
    await _seed_pinnacle_event(factory, "pin-dup-2", "Alpha", "Beta")
    async with factory() as session:
        out = await resolve_pinnacle_close_snaps(
            session,
            pinnacle_sport_key="pinnacle_soccer",
            pick_external_ref="evt-pick",
            home="Alpha",
            away="Beta",
            kickoff=KO,
        )
    assert out == []


async def test_resolver_kickoff_outside_window_returns_empty(factory) -> None:  # type: ignore[no-untyped-def]
    await _seed_pinnacle_event(factory, "pin-far", "Alpha", "Beta")
    async with factory() as session:
        out = await resolve_pinnacle_close_snaps(
            session,
            pinnacle_sport_key="pinnacle_soccer",
            pick_external_ref="evt-pick",
            home="Alpha",
            away="Beta",
            kickoff=KO + timedelta(days=4),  # archive event is days away
            max_day_drift=1,
        )
    assert out == []


def _pin_snap_at(selection: str, odds: float, event: str, captured: datetime) -> OddsSnapshotIn:
    return OddsSnapshotIn(
        event_id=event,
        bookmaker="Pinnacle",
        market=Market.H2H,
        selection=selection,
        decimal_odds=odds,
        captured_at=captured,
        ingested_at=captured,
    )


async def test_resolver_caps_cutoff_at_arcadia_kickoff(factory) -> None:  # type: ignore[no-untyped-def]
    # The arcadia event kicks off a DAY before the pick (within the match
    # window). A Pinnacle row captured AFTER the arcadia kickoff (in-play) must
    # NOT become the close — the cutoff is capped at the arcadia kickoff.
    arc_ko = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)
    pick_ko = arc_ko + timedelta(days=1)
    pre = arc_ko - timedelta(hours=2)  # valid pre-kickoff close
    inplay = arc_ko + timedelta(hours=1)  # in-play; must be excluded
    ref = "pin-cutoff"
    snaps = [
        _pin_snap_at("Alpha", 2.10, ref, pre),
        _pin_snap_at("Draw", 3.40, ref, pre),
        _pin_snap_at("Beta", 3.60, ref, pre),
        _pin_snap_at("Alpha", 1.50, ref, inplay),  # later in-play home price
    ]
    teams = {ref: EventTeams(home="Alpha", away="Beta", league="pin", starts_at=arc_ko)}
    await persist_odds_snapshots(factory, snaps, teams, "pinnacle_soccer", "pinnacle_soccer")
    async with factory() as session:
        out = await resolve_pinnacle_close_snaps(
            session,
            pinnacle_sport_key="pinnacle_soccer",
            pick_external_ref="evt-pick",
            home="Alpha",
            away="Beta",
            kickoff=pick_ko,
            max_day_drift=1,
        )
    by_sel = {s.selection: s for s in out}
    # the Alpha close must be the PRE-kickoff 2.10, NOT the in-play 1.50
    assert by_sel["Alpha"].decimal_odds == pytest.approx(2.10)
