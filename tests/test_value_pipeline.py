"""Value pipeline: multi-book snapshots -> anchor -> value pick -> alert."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.edge.gates import GatePolicy
from app.ingestion.base import EventDirectory, EventTeams
from app.models.base import NullModel
from app.notifications.base import Alert
from app.notifications.dedupe import InMemoryIdempotencyStore
from app.notifications.dispatcher import AlertDispatcher
from app.pipeline import PipelineDeps, run_value_pipeline
from app.risk.exposure import DailyExposureLedger
from app.risk.staking import StakePolicy
from app.schemas.base import Market
from app.schemas.odds import OddsSnapshotIn

NOW = datetime.now(tz=UTC)

POLICY = GatePolicy(
    min_edge=0.0,
    min_ev=0.0,
    min_confidence=0.0,
    max_odds_age_seconds=300,
    min_liquidity=0.0,
)


def snap(book: str, sel: str, odds: float, age_s: float = 30.0) -> OddsSnapshotIn:
    return OddsSnapshotIn(
        event_id="evt-1",
        bookmaker=book,
        market=Market.H2H,
        selection=sel,
        decimal_odds=odds,
        captured_at=NOW - timedelta(seconds=age_s),
        ingested_at=NOW,
    )


class FakeLoader:
    def __init__(self, snapshots: list[OddsSnapshotIn]) -> None:
        self._snapshots = snapshots

    async def fetch_odds(self, sport_key: str) -> Sequence[OddsSnapshotIn]:
        return self._snapshots


class RecordingSink:
    name = "recording"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


def market_snapshots(age_s: float = 30.0) -> list[OddsSnapshotIn]:
    # Pinnacle prices a tight 3-way; SoftBook is too generous on Home.
    return [
        snap("Pinnacle", "Home FC", 2.50, age_s),
        snap("Pinnacle", "Draw", 3.30, age_s),
        snap("Pinnacle", "Away FC", 3.10, age_s),
        snap("SoftBook", "Home FC", 2.90, age_s),
        snap("SoftBook", "Draw", 3.20, age_s),
        snap("SoftBook", "Away FC", 2.95, age_s),
    ]


def make_deps(sink: RecordingSink, loader: FakeLoader) -> PipelineDeps:
    directory = EventDirectory()
    directory.register("evt-1", EventTeams(home="Home FC", away="Away FC"))
    return PipelineDeps(
        loader=loader,
        model=NullModel(),
        dispatcher=AlertDispatcher([sink], InMemoryIdempotencyStore()),
        gate_policy=POLICY,
        stake_policy=StakePolicy(),
        ledger=DailyExposureLedger(max_daily_fraction=0.05),
        bankroll=Decimal("1000"),
        directory=directory,
        value_min_edge=0.015,
        value_min_odds=1.30,
    )


async def test_value_pipeline_produces_pick_and_alert() -> None:
    sink = RecordingSink()
    picks = await run_value_pipeline(make_deps(sink, FakeLoader(market_snapshots())), "soccer")
    assert len(picks) == 1
    pick = picks[0]
    assert pick.selection == "Home FC"
    assert pick.bookmaker == "SoftBook"
    assert pick.decimal_odds == 2.90
    # model_probability carries the SHARP fair prob; edge = fair - implied
    assert pick.model_probability > pick.fair_probability
    assert pick.edge >= 0.015
    assert pick.confidence == 0.9  # named sharp anchor (Pinnacle)
    assert pick.event == "Home FC vs Away FC"
    assert len(sink.sent) == 1
    assert "does not place bets" in sink.sent[0].body
    assert "value: Pinnacle fair" in pick.reason_summary


async def test_value_pipeline_rerun_dedupes_alert() -> None:
    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))
    first = await run_value_pipeline(deps, "soccer")
    second = await run_value_pipeline(deps, "soccer")
    assert len(first) == len(second) == 1
    assert len(sink.sent) == 1  # same market state -> one alert


async def test_value_pipeline_skips_stale_odds() -> None:
    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots(age_s=400.0)))  # > 300s gate
    picks = await run_value_pipeline(deps, "soccer")
    assert picks == []
    assert sink.sent == []


async def test_value_pipeline_handles_future_captured_at() -> None:
    # Live scrapes stamp captured_at DURING the multi-minute fetch; a snapshot
    # "newer than now" must clamp to age 0, not crash PickOut validation.
    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots(age_s=-90.0)))  # future
    picks = await run_value_pipeline(deps, "soccer")
    assert len(picks) == 1
    assert picks[0].odds_age_seconds == 0.0


async def test_value_pipeline_no_anchor_no_picks() -> None:
    # Only two books and neither is a named sharp -> no trustworthy anchor.
    snaps = [s for s in market_snapshots() if s.bookmaker != "Pinnacle"]
    snaps += [
        snap("OtherBook", "Home FC", 2.55),
        # OtherBook prices only one selection -> not a full-market book
    ]
    sink = RecordingSink()
    picks = await run_value_pipeline(make_deps(sink, FakeLoader(snaps)), "soccer")
    assert picks == []
