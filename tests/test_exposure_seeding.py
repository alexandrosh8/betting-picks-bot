"""Exposure-ledger startup seeding: today's persisted picks must preload the
in-memory daily cap, or a mid-day restart doubles recommendable exposure.

Uses the compose Postgres with savepoint isolation (skips when absent) —
same fixture pattern as tests/test_clv_trueup.py. The dev warehouse may
already hold real picks created today, so assertions are baseline-relative.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.base import EventTeams
from app.risk.exposure import DailyExposureLedger
from app.scheduler import seed_exposure_ledger
from app.schemas.base import Market
from app.schemas.picks import PickOut, StakeBreakdownOut
from app.storage.models import Pick
from app.storage.repositories import persist_pick

DB_URL = "postgresql+asyncpg://betting_ai:betting_ai@localhost:5433/betting_ai"
NOW = datetime.now(tz=UTC)


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


def make_pick(event_id: str, stake_fraction: float, marker: str) -> PickOut:
    return PickOut(
        pick_id=f"p-{marker}",
        sport="soccer",
        league="test-league-seeding",
        event="Home FC vs Away FC",
        event_id=event_id,
        market=Market.H2H,
        selection="Home FC",
        bookmaker="SoftBook",
        decimal_odds=2.50,
        model_probability=0.45,
        fair_probability=0.40,
        edge=0.05,
        ev=0.125,
        confidence=0.9,
        recommended_stake_fraction=stake_fraction,
        recommended_stake_amount=Decimal("20.00"),
        stake_breakdown=StakeBreakdownOut(
            raw_kelly=0.1, fractional=0.025, capped=True, final=stake_fraction
        ),
        odds_age_seconds=30.0,
        liquidity=None,
        reason_summary=marker,
        created_at=NOW,
    )


async def _persist_with_created_at(
    factory: async_sessionmaker,
    event_id: str,
    stake_fraction: float,
    marker: str,
    created_at: datetime,
) -> None:
    async with factory() as session:
        await persist_pick(
            session,
            make_pick(event_id, stake_fraction, marker),
            EventTeams(home="Home FC", away="Away FC"),
            "value-sharp-vs-soft",
            "v-seeding-test",
        )
        # server_default now() stamps transaction time; pin the row to the
        # intended day explicitly so the boundary query is exercised.
        await session.execute(
            sa_update(Pick).where(Pick.reason_summary == marker).values(created_at=created_at)
        )
        await session.commit()


async def test_seeding_counts_only_todays_picks(factory) -> None:  # type: ignore[no-untyped-def]
    today_utc = NOW.date()

    baseline_ledger = DailyExposureLedger(max_daily_fraction=1.0)
    await seed_exposure_ledger(baseline_ledger, factory)
    baseline = baseline_ledger.used(today_utc)

    now = datetime.now(tz=UTC)
    await _persist_with_created_at(factory, "evt-seed-today-1", 0.02, "seeding-today-1", now)
    await _persist_with_created_at(factory, "evt-seed-today-2", 0.01, "seeding-today-2", now)
    await _persist_with_created_at(
        factory, "evt-seed-yesterday", 0.04, "seeding-yesterday", now - timedelta(days=1)
    )

    ledger = DailyExposureLedger(max_daily_fraction=1.0)
    await seed_exposure_ledger(ledger, factory)
    # today's two picks count; yesterday's 0.04 must NOT leak into today
    assert ledger.used(today_utc) == pytest.approx(baseline + 0.03, abs=1e-9)


async def test_seeding_is_idempotent_preload_not_accumulate(factory) -> None:  # type: ignore[no-untyped-def]
    today_utc = NOW.date()
    now = datetime.now(tz=UTC)
    await _persist_with_created_at(factory, "evt-seed-idem", 0.02, "seeding-idem", now)

    ledger = DailyExposureLedger(max_daily_fraction=1.0)
    await seed_exposure_ledger(ledger, factory)
    first = ledger.used(today_utc)
    await seed_exposure_ledger(ledger, factory)
    assert ledger.used(today_utc) == pytest.approx(first, abs=1e-9)
