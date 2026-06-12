"""Notifications: idempotency, fan-out, never-raise sinks, reminder presence."""

from datetime import UTC, datetime
from decimal import Decimal

import fakeredis.aioredis

from app.notifications.base import Alert, build_pick_alert
from app.notifications.dedupe import (
    DEFAULT_TTL_SECONDS,
    InMemoryIdempotencyStore,
    RedisIdempotencyStore,
)
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


class FlakySink:
    """Raises on the first send, delivers afterwards — a transient outage."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient outage")
        self.sent.append(alert)
        return True


class UnconfiguredSink:
    """Mirrors Telegram/Webhook sinks with no token/url: skips by design."""

    name = "unconfigured"
    configured = False

    async def send(self, alert: Alert) -> bool:
        return False


async def test_failed_dispatch_releases_claim_so_next_cycle_retries() -> None:
    # Claim-leak regression: claim succeeds, the send raises -> the key must
    # be RELEASED, so the pipeline's next-cycle re-dispatch of the same
    # market state retries instead of being suppressed forever.
    sink = FlakySink()
    dispatcher = AlertDispatcher([sink], InMemoryIdempotencyStore())
    first = await dispatcher.dispatch(make_alert())
    assert first.skipped_duplicate is False
    assert first.sink_results == (("flaky", False),)

    second = await dispatcher.dispatch(make_alert())  # next cycle, same key
    assert second.skipped_duplicate is False  # claim was released
    assert second.sink_results == (("flaky", True),)
    assert len(sink.sent) == 1

    third = await dispatcher.dispatch(make_alert())  # delivered -> claim sticks
    assert third.skipped_duplicate is True
    assert len(sink.sent) == 1


async def test_partial_delivery_keeps_claim() -> None:
    # One channel delivered: releasing would duplicate the alert to the
    # healthy channel on the next cycle.
    recording = RecordingSink()
    dispatcher = AlertDispatcher([ExplodingSink(), recording], InMemoryIdempotencyStore())
    await dispatcher.dispatch(make_alert())
    second = await dispatcher.dispatch(make_alert())
    assert second.skipped_duplicate is True
    assert len(recording.sent) == 1


async def test_unconfigured_sinks_do_not_release_claim() -> None:
    # No-channel deployments: an unconfigured sink's False is a skip by
    # design, not a delivery failure — without this, every alert would
    # re-dispatch every cycle forever.
    dispatcher = AlertDispatcher([UnconfiguredSink()], InMemoryIdempotencyStore())
    first = await dispatcher.dispatch(make_alert())
    assert first.skipped_duplicate is False
    second = await dispatcher.dispatch(make_alert())
    assert second.skipped_duplicate is True


async def test_redis_idempotency_store_claims_once() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    store = RedisIdempotencyStore(redis)
    assert await store.claim("key-1") is True
    assert await store.claim("key-1") is False
    assert await store.claim("key-2") is True


async def test_redis_idempotency_store_release_reopens_key() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    store = RedisIdempotencyStore(redis)
    assert await store.claim("key-1") is True
    await store.release("key-1")
    assert await store.claim("key-1") is True  # claimable again after release


async def test_redis_idempotency_ttl_keeps_unchanged_odds_quiet_for_seven_days() -> None:
    # Unchanged odds on a still-open pick must NOT re-alert daily: the claim
    # TTL is 7 days by default (a price move mints a new key regardless).
    assert DEFAULT_TTL_SECONDS == 7 * 24 * 60 * 60
    redis = fakeredis.aioredis.FakeRedis()
    store = RedisIdempotencyStore(redis)
    assert await store.claim("key-ttl") is True
    assert await redis.ttl("alert:dedupe:key-ttl") == DEFAULT_TTL_SECONDS
    assert await store.claim("key-ttl") is False  # quiet within the TTL

    custom = RedisIdempotencyStore(redis, ttl_seconds=3600)
    assert await custom.claim("key-custom") is True
    assert await redis.ttl("alert:dedupe:key-custom") == 3600


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


def test_pick_alert_still_ev_line_from_value_min_edge() -> None:
    # Value-pipeline alerts carry the execution helper: model_probability is
    # the sharp fair prob there, so the floor is 1/(0.55-0.03)=1.923 -> 1.93
    # after the round-UP display rule (rounding down would lose the edge).
    alert = build_pick_alert(make_pick(), value_min_edge=0.03)
    assert "Still +EV down to: 1.93 at bookie_one" in alert.body
    assert "skip" in alert.body
    # default (model strategy / legacy callers): no line, identical key
    plain = build_pick_alert(make_pick())
    assert "Still +EV down to" not in plain.body
    assert plain.dedupe_key == alert.dedupe_key


def test_pick_alert_omits_still_ev_line_when_no_price_retains_edge() -> None:
    # fair prob at/below the threshold: the helper has no honest floor to
    # print, and the alert must not invent one.
    pick = make_pick().model_copy(update={"model_probability": 0.02})
    alert = build_pick_alert(pick, value_min_edge=0.03)
    assert "Still +EV down to" not in alert.body
