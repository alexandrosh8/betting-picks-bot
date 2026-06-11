"""Pick persistence against a real Postgres (compose). Skips if DB is absent.

Uses a savepoint-style rollback: each test runs in a transaction that is
rolled back, so the warehouse is never mutated by the suite.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.base import EventTeams
from app.schemas.base import Market
from app.schemas.picks import PickOut, StakeBreakdownOut
from app.storage.models import Event, Pick
from app.storage.repositories import (
    latest_picks_with_events,
    persist_pick,
    refresh_event_kickoffs,
)

DB_URL = "postgresql+asyncpg://betting_ai:betting_ai@localhost:5433/betting_ai"


def make_pick(
    event_id: str = "evt-persist-test",
    tier: str = "premium",
    decimal_odds: float = 2.10,
    edge: float = 0.05,
) -> PickOut:
    return PickOut(
        pick_id="p-1",
        sport="soccer",
        league="test-league-persist",
        event="Alpha FC vs Beta United",
        event_id=event_id,
        market=Market.H2H,
        selection="Alpha FC",
        bookmaker="testbook",
        decimal_odds=decimal_odds,
        model_probability=0.55,
        fair_probability=0.50,
        edge=edge,
        ev=0.155,
        confidence=0.70,
        recommended_stake_fraction=0.02,
        recommended_stake_amount=Decimal("20.00"),
        stake_breakdown=StakeBreakdownOut(raw_kelly=0.1, fractional=0.025, capped=True, final=0.02),
        odds_age_seconds=30.0,
        liquidity=None,
        reason_summary="persistence test",
        tier=tier,
        created_at=datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
async def session():  # type: ignore[no-untyped-def]
    engine = create_async_engine(DB_URL)
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    except Exception:
        await engine.dispose()
        pytest.skip("compose Postgres not reachable on :5433")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        await s.begin()
        try:
            yield s
        finally:
            await s.rollback()
    await engine.dispose()


async def test_persist_pick_inserts_then_dedupes(session) -> None:  # type: ignore[no-untyped-def]
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")

    inserted = await persist_pick(session, make_pick(), teams, "dixon-coles", "test-1")
    assert inserted == "inserted"

    count = await session.scalar(
        select(func.count()).select_from(Pick).where(Pick.bookmaker == "testbook")
    )
    assert count == 1

    # Same natural key -> deduped (no second row)
    again = await persist_pick(session, make_pick(), teams, "dixon-coles", "test-1")
    assert again == "duplicate"
    count2 = await session.scalar(
        select(func.count()).select_from(Pick).where(Pick.bookmaker == "testbook")
    )
    assert count2 == 1


async def test_persisted_pick_roundtrips_fields(session) -> None:  # type: ignore[no-untyped-def]
    teams = EventTeams(home="Alpha FC", away="Beta United")
    await persist_pick(session, make_pick("evt-roundtrip"), teams, "dixon-coles", "test-2")
    row = await session.scalar(
        select(Pick).where(Pick.bookmaker == "testbook").order_by(Pick.id.desc())
    )
    assert row is not None
    assert row.selection == "Alpha FC"
    assert row.market == "h2h"
    assert row.status == "alerted"
    assert row.decimal_odds == Decimal("2.1000")
    assert row.stake_breakdown["final"] == 0.02


async def test_version_bump_supersedes_older_open_pick(session) -> None:  # type: ignore[no-untyped-def]
    # Same (event, market, selection) under a NEW strategy version must not
    # show twice on the dashboard: the old open row flips to 'superseded'.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    await persist_pick(session, make_pick("evt-supersede"), teams, "value-sharp-vs-soft", "v2")
    await persist_pick(session, make_pick("evt-supersede"), teams, "value-sharp-vs-soft", "v3")

    rows = (
        (
            await session.execute(
                select(Pick.status).where(Pick.bookmaker == "testbook").order_by(Pick.id)
            )
        )
        .scalars()
        .all()
    )
    assert rows == ["superseded", "alerted"]


async def test_volume_pick_persists_with_tier_and_serializes_it(session) -> None:  # type: ignore[no-untyped-def]
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    outcome = await persist_pick(
        session,
        make_pick("evt-tier-vol", tier="volume", edge=0.02),
        teams,
        "value-sharp-vs-soft",
        "tier-t1",
    )
    assert outcome == "inserted"
    row = await session.scalar(select(Pick).where(Pick.bookmaker == "testbook"))
    assert row is not None
    assert row.tier == "volume"
    assert row.status == "alerted"  # lifecycle is shared; tier scopes behavior

    payload = await latest_picks_with_events(session, limit=200)
    ours = [p for p in payload if p["bookmaker"] == "testbook"]
    assert ours and ours[0]["tier"] == "volume"  # API exposes the tier


async def test_premium_key_is_shielded_from_volume_redetection(session) -> None:  # type: ignore[no-untyped-def]
    # The unique key (event, market, selection, model) collides across tiers
    # BY DESIGN; tier may only ratchet upward. A premium row must never be
    # downgraded or touched by a later volume candidate.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    assert (
        await persist_pick(
            session, make_pick("evt-tier-shield"), teams, "value-sharp-vs-soft", "tier-t2"
        )
        == "inserted"
    )
    outcome = await persist_pick(
        session,
        make_pick("evt-tier-shield", tier="volume", decimal_odds=2.40, edge=0.02),
        teams,
        "value-sharp-vs-soft",
        "tier-t2",
    )
    assert outcome == "duplicate"
    row = await session.scalar(select(Pick).where(Pick.bookmaker == "testbook"))
    assert row is not None
    assert row.tier == "premium"  # untouched
    assert row.decimal_odds == Decimal("2.1000")  # original market numbers kept


async def test_volume_to_premium_upgrade_promotes_row_in_place(session) -> None:  # type: ignore[no-untyped-def]
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    assert (
        await persist_pick(
            session,
            make_pick("evt-tier-upgrade", tier="volume", decimal_odds=2.10, edge=0.02),
            teams,
            "value-sharp-vs-soft",
            "tier-t3",
        )
        == "inserted"
    )
    before = await session.scalar(select(Pick).where(Pick.bookmaker == "testbook"))
    assert before is not None
    created_before = before.created_at

    outcome = await persist_pick(
        session,
        make_pick("evt-tier-upgrade", tier="premium", decimal_odds=2.30, edge=0.05),
        teams,
        "value-sharp-vs-soft",
        "tier-t3",
    )
    assert outcome == "upgraded"
    rows = (await session.execute(select(Pick).where(Pick.bookmaker == "testbook"))).scalars().all()
    assert len(rows) == 1  # promoted IN PLACE — never a second row
    row = rows[0]
    assert row.tier == "premium"
    assert row.status == "alerted"
    assert row.decimal_odds == Decimal("2.3000")  # the alert quotes the row
    assert row.edge == Decimal("0.050000")
    # created_at advances to the upgrade moment (exposure-seeding invariant)
    assert row.created_at > created_before
    # stale revalidation verdicts (priced on the old odds) are reset
    assert row.clv_log is None
    assert row.current_odds is None
    assert row.revalidated_at is None


async def test_settled_volume_row_is_not_upgraded(session) -> None:  # type: ignore[no-untyped-def]
    # Once the lifecycle moved past "alerted" the market moment is gone —
    # a late premium detection on the same key is a plain duplicate.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    await persist_pick(
        session,
        make_pick("evt-tier-settled", tier="volume", edge=0.02),
        teams,
        "value-sharp-vs-soft",
        "tier-t4",
    )
    await session.execute(
        sa_update(Pick).where(Pick.bookmaker == "testbook").values(status="settled")
    )
    outcome = await persist_pick(
        session,
        make_pick("evt-tier-settled", tier="premium", edge=0.05),
        teams,
        "value-sharp-vs-soft",
        "tier-t4",
    )
    assert outcome == "duplicate"
    row = await session.scalar(select(Pick).where(Pick.bookmaker == "testbook"))
    assert row is not None
    assert row.tier == "volume"
    assert row.status == "settled"


async def test_volume_insert_never_supersedes_open_premium(session) -> None:  # type: ignore[no-untyped-def]
    # Cross-version supersede respects tier: a NEW volume row (new strategy
    # version) must not flip an older OPEN premium row to 'superseded' —
    # while a premium insert supersedes any older open row, volume included.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    await persist_pick(session, make_pick("evt-tier-sup"), teams, "value-sharp-vs-soft", "v-old")
    assert (
        await persist_pick(
            session,
            make_pick("evt-tier-sup", tier="volume", edge=0.02),
            teams,
            "value-sharp-vs-soft",
            "v-new",
        )
        == "inserted"
    )
    statuses = dict(
        (
            await session.execute(
                select(Pick.tier, Pick.status).where(Pick.bookmaker == "testbook")
            )
        ).all()
    )
    assert statuses == {"premium": "alerted", "volume": "alerted"}  # premium survives

    # ...and the reverse: a premium insert under yet another version
    # supersedes BOTH older open rows (premium and volume).
    assert (
        await persist_pick(
            session, make_pick("evt-tier-sup"), teams, "value-sharp-vs-soft", "v-newest"
        )
        == "inserted"
    )
    rows = (
        await session.execute(
            select(Pick.status).where(Pick.bookmaker == "testbook").order_by(Pick.id)
        )
    ).scalars()
    assert sorted(rows) == ["alerted", "superseded", "superseded"]


async def test_refresh_event_kickoffs_upgrades_placeholder(session) -> None:  # type: ignore[no-untyped-def]
    # Events created before the source kickoff was known carry a pick-time
    # placeholder; a later scrape with the real kickoff must correct them.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    await persist_pick(session, make_pick("evt-kickoff-fix"), teams, "value-sharp-vs-soft", "t-4")

    real_kickoff = datetime(2026, 6, 25, 1, 0, tzinfo=UTC)
    changed = await refresh_event_kickoffs(session, {"evt-kickoff-fix": real_kickoff})
    assert changed == 1
    row = await session.scalar(select(Event).where(Event.external_ref == "evt-kickoff-fix"))
    assert row is not None
    assert row.starts_at == real_kickoff
    # idempotent: same kickoff again -> no change
    assert await refresh_event_kickoffs(session, {"evt-kickoff-fix": real_kickoff}) == 0


async def test_unknown_kickoff_persists_null_and_serializes_null(session) -> None:  # type: ignore[no-untyped-def]
    # When the source reports no kickoff, the event stores NULL — the REAL
    # "kickoff unknown" signal. The old placeholder (pick-time timestamp) was
    # undetectable client-side: Pick.created_at is a separate clock read, so
    # starts_at == created_at never held and TBD rows rendered as started
    # matches with live settle buttons.
    teams = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    assert teams.starts_at is None
    await persist_pick(session, make_pick("evt-kickoff-null"), teams, "value-sharp-vs-soft", "t-5")

    row = await session.scalar(select(Event).where(Event.external_ref == "evt-kickoff-null"))
    assert row is not None
    assert row.starts_at is None  # NULL, not a fake pick-time kickoff

    payload = await latest_picks_with_events(session, limit=200)
    ours = [p for p in payload if p["event_id"] == row.id]
    assert ours, "persisted pick missing from the /picks payload"
    assert ours[0]["starts_at"] is None  # dashboard renders TBD, no settle


async def test_unknown_kickoff_upgrades_once_source_reports_it(session) -> None:  # type: ignore[no-untyped-def]
    # A NULL kickoff must heal through BOTH paths: a later pick that knows
    # the kickoff (_get_or_create_event) and the per-cycle refresh.
    teams_unknown = EventTeams(home="Alpha FC", away="Beta United", league="test-league-persist")
    await persist_pick(
        session, make_pick("evt-kickoff-heal"), teams_unknown, "value-sharp-vs-soft", "t-6"
    )

    kickoff = datetime(2026, 6, 27, 18, 30, tzinfo=UTC)
    teams_known = EventTeams(
        home="Alpha FC", away="Beta United", league="test-league-persist", starts_at=kickoff
    )
    # same event re-detected, now with a known kickoff (dedupe path)
    await persist_pick(
        session, make_pick("evt-kickoff-heal"), teams_known, "value-sharp-vs-soft", "t-6"
    )
    row = await session.scalar(select(Event).where(Event.external_ref == "evt-kickoff-heal"))
    assert row is not None
    assert row.starts_at == kickoff

    # and refresh_event_kickoffs upgrades a NULL row too
    await persist_pick(
        session, make_pick("evt-kickoff-heal2"), teams_unknown, "value-sharp-vs-soft", "t-6"
    )
    assert await refresh_event_kickoffs(session, {"evt-kickoff-heal2": kickoff}) == 1
    row2 = await session.scalar(select(Event).where(Event.external_ref == "evt-kickoff-heal2"))
    assert row2 is not None
    assert row2.starts_at == kickoff


async def test_latest_picks_payload_carries_event_fields(session) -> None:  # type: ignore[no-untyped-def]
    # The dashboard needs match label / league / kickoff — not bare event ids.
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=UTC)
    teams = EventTeams(
        home="Alpha FC", away="Beta United", league="test-league-persist", starts_at=kickoff
    )
    await persist_pick(session, make_pick("evt-dashboard"), teams, "value-sharp-vs-soft", "t-3")

    payload = await latest_picks_with_events(session, limit=200)
    ours = [p for p in payload if p["bookmaker"] == "testbook"]
    assert ours, "persisted pick missing from the /picks payload"
    p = ours[0]
    assert p["event"] == "Alpha FC vs Beta United"
    assert p["league"] == "test-league-persist"
    assert p["starts_at"] == kickoff.isoformat()  # real kickoff, UTC ISO-8601
    assert p["selection"] == "Alpha FC"
    assert p["reason_summary"] == "persistence test"
    # API payload is data-only; the safety reminder lives in alerts
    # (app/schemas/picks.py, safety_audit check 8) and the dashboard banner.
    assert "manual_betting_reminder" not in p
