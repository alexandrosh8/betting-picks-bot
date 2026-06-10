"""Notifications: idempotency, fan-out, never-raise sinks, reminder presence."""

from datetime import UTC, datetime
from decimal import Decimal

import fakeredis.aioredis

from app.notifications.base import Alert, build_pick_alert
from app.notifications.dedupe import InMemoryIdempotencyStore, RedisIdempotencyStore
from app.notifications.dispatcher import AlertDispatcher
from app.schemas.base import Market
from app.schemas.picks import MANUAL_BETTING_REMINDER, PickOut, StakeBreakdownOut


class RecordingSink:
    name = "recording"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


class ExplodingSink:
    name = "exploding"

    async def send(self, alert: Alert) -> bool:
        raise RuntimeError("boom")


def make_alert(key: str = "k1") -> Alert:
    return Alert(pick_id="p1", title="t", body="b", dedupe_key=key)


def make_pick() -> PickOut:
    return PickOut(
        pick_id="pick-1",
        sport="soccer",
        league="EPL",
        event="Alpha FC vs Beta United",
        event_id="evt-abc",
        market=Market.H2H,
        selection="Alpha FC",
        bookmaker="bookie_one",
        decimal_odds=2.1,
        model_probability=0.55,
        fair_probability=0.50,
        edge=0.05,
        ev=0.155,
        confidence=0.75,
        recommended_stake_fraction=0.02,
        recommended_stake_amount=Decimal("20.00"),
        stake_breakdown=StakeBreakdownOut(
            raw_kelly=0.10, fractional=0.025, capped=True, final=0.02
        ),
        odds_age_seconds=60.0,
        liquidity=None,
        reason_summary="model edge over devigged market",
        created_at=datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
    )


async def test_duplicate_alert_suppressed() -> None:
    sink = RecordingSink()
    dispatcher = AlertDispatcher([sink], InMemoryIdempotencyStore())
    first = await dispatcher.dispatch(make_alert())
    second = await dispatcher.dispatch(make_alert())
    assert first.skipped_duplicate is False
    assert second.skipped_duplicate is True
    assert len(sink.sent) == 1


async def test_different_keys_both_deliver() -> None:
    sink = RecordingSink()
    dispatcher = AlertDispatcher([sink], InMemoryIdempotencyStore())
    await dispatcher.dispatch(make_alert("k1"))
    await dispatcher.dispatch(make_alert("k2"))
    assert len(sink.sent) == 2


async def test_sink_failure_does_not_raise_or_block_others() -> None:
    recording = RecordingSink()
    dispatcher = AlertDispatcher([ExplodingSink(), recording], InMemoryIdempotencyStore())
    result = await dispatcher.dispatch(make_alert())
    assert result.sink_results[0] == ("exploding", False)
    assert result.sink_results[1] == ("recording", True)
    assert len(recording.sent) == 1


async def test_redis_idempotency_store_claims_once() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    store = RedisIdempotencyStore(redis)
    assert await store.claim("key-1") is True
    assert await store.claim("key-1") is False
    assert await store.claim("key-2") is True


def test_pick_alert_contains_required_fields_and_reminder() -> None:
    alert = build_pick_alert(make_pick())
    assert MANUAL_BETTING_REMINDER in alert.body
    for fragment in (
        "Alpha FC vs Beta United",
        "Model probability: 0.550",
        "Fair (vig-free) probability: 0.500",
        "Edge: +0.050",
        "EV: +0.155",
        "Recommended stake: 2.00%",
        "informational only",
        "Odds age: 60s",
    ):
        assert fragment in alert.body, fragment
    assert len(alert.dedupe_key) == 32


def test_same_pick_same_dedupe_key() -> None:
    assert build_pick_alert(make_pick()).dedupe_key == build_pick_alert(make_pick()).dedupe_key
