"""Value pipeline: multi-book snapshots -> anchor -> value pick -> alert."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

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
        self.snapshots = snapshots
        # Mirrors OddsPortalLoader's liveness contract read by _record_poll.
        self.last_fetch_matches: dict[str, int] = {}

    async def fetch_odds(self, sport_key: str) -> Sequence[OddsSnapshotIn]:
        self.last_fetch_matches[sport_key] = len({s.event_id for s in self.snapshots})
        return self.snapshots


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


async def test_value_pipeline_records_poll_liveness() -> None:
    # The dashboard/health must be able to tell "engine alive" from "engine
    # dead showing day-old picks" — every cycle records itself, including
    # per-market snapshot counts and the loader's listing count so a selector
    # break (matches found, zero odds parsed) is visible, not silent.
    from app.pipeline import LAST_POLL

    sink = RecordingSink()
    await run_value_pipeline(make_deps(sink, FakeLoader(market_snapshots())), "soccer")
    poll = LAST_POLL["soccer"]
    assert poll["finished_at"] is not None
    assert poll["snapshots"] > 0
    assert poll["picks"] == 1
    assert poll["matches_found"] == 1
    assert poll["per_market"] == {"h2h": 6}
    assert poll["degraded"] is False


async def test_poll_record_flags_degraded_on_matches_without_snapshots() -> None:
    """Selector/DOM break (or anti-bot wall): listings parse, every odds row
    is missed. Cycles still complete, so finished_at alone looks healthy —
    the poll record must carry an explicit degraded flag for /health."""
    from app.pipeline import LAST_POLL

    class BrokenScrapeLoader(FakeLoader):
        async def fetch_odds(self, sport_key: str) -> Sequence[OddsSnapshotIn]:
            self.last_fetch_matches[sport_key] = 7  # listings parsed fine
            return []  # ...but zero odds rows survived parsing

    sink = RecordingSink()
    await run_value_pipeline(make_deps(sink, BrokenScrapeLoader([])), "soccer")
    poll = LAST_POLL["soccer"]
    assert poll["matches_found"] == 7
    assert poll["snapshots"] == 0
    assert poll["per_market"] == {}
    assert poll["degraded"] is True


async def test_poll_record_without_listing_count_is_not_degraded() -> None:
    # Loaders that don't report listing counts (odds_api, plain fakes) must
    # not be flagged degraded on an empty day — unknown is not broken.
    from app.pipeline import LAST_POLL

    class CountlessLoader:
        async def fetch_odds(self, sport_key: str) -> Sequence[OddsSnapshotIn]:
            return []

    sink = RecordingSink()
    await run_value_pipeline(make_deps(sink, CountlessLoader()), "soccer")  # type: ignore[arg-type]
    poll = LAST_POLL["soccer"]
    assert poll["matches_found"] is None
    assert poll["degraded"] is False


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


class FakeSessionFactory:
    """Minimal async-contextmanager session; revalidation calls against
    it raise and are swallowed by the pipeline's try/except."""

    def __call__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aexit__(self, *exc):  # type: ignore[no-untyped-def]
        return False

    async def commit(self) -> None:
        return None


def patch_persist_dedupe_after_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_pick inserts on the FIRST call, dedupes every later call —
    the DB unique key (event, market, selection, model) ignores odds."""
    import app.storage.repositories as repos

    calls = {"n": 0}

    async def fake_persist_pick(session, pick, teams, model_name, model_version):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(repos, "persist_pick", fake_persist_pick)


async def test_duplicate_pick_releases_exposure_and_unchanged_odds_stay_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1 regression: a pick already persisted (DB dedupe) must hand its
    daily-exposure grant back — leaking one grant per cycle exhausts the
    daily cap within minutes. The alert is still DISPATCHED (so a failed
    first delivery self-heals); with unchanged odds the idempotency store
    suppresses it, so exactly one alert reaches the sink."""
    patch_persist_dedupe_after_first(monkeypatch)

    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))
    deps.session_factory = FakeSessionFactory()  # type: ignore[assignment]

    day = datetime.now(tz=UTC).date()
    first = await run_value_pipeline(deps, "soccer")
    assert len(first) == 1
    used_after_first = deps.ledger.used(day)
    assert used_after_first > 0.0

    second = await run_value_pipeline(deps, "soccer")
    assert second == []  # duplicate is not a new pick this cycle
    assert deps.ledger.used(day) == pytest.approx(used_after_first)  # grant returned
    assert len(sink.sent) == 1  # idempotency (key includes odds) suppressed it


async def test_duplicate_pick_with_price_move_realerts_and_still_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alert dedupe key deliberately includes decimal_odds (notifications/
    base.py): a material price move on a pick the DB already knows must
    RE-ALERT — skipping dispatch on DB dedupe killed that design. Exposure is
    still handed back: a re-priced duplicate is not new exposure."""
    patch_persist_dedupe_after_first(monkeypatch)

    sink = RecordingSink()
    loader = FakeLoader(market_snapshots())
    deps = make_deps(sink, loader)
    deps.session_factory = FakeSessionFactory()  # type: ignore[assignment]

    day = datetime.now(tz=UTC).date()
    first = await run_value_pipeline(deps, "soccer")
    assert len(first) == 1
    assert len(sink.sent) == 1
    used_after_first = deps.ledger.used(day)

    # SoftBook moves its Home price 2.90 -> 2.95: same DB row (dedupe ignores
    # odds), materially different market state.
    loader.snapshots = [
        snap("SoftBook", "Home FC", 2.95)
        if s.bookmaker == "SoftBook" and s.selection == "Home FC"
        else s
        for s in market_snapshots()
    ]
    second = await run_value_pipeline(deps, "soccer")
    assert second == []  # still not a NEW pick
    assert deps.ledger.used(day) == pytest.approx(used_after_first)  # grant returned
    assert len(sink.sent) == 2  # price move re-alerted
    assert "2.95" in sink.sent[1].title


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


async def test_value_pipeline_prices_half_line_handicap_directly() -> None:
    # Half-line AH is a full 2-way market: direct devig anchor, line kept
    # separate via market_detail. Pinnacle tight, SoftBook generous on home.
    def ah(book: str, sel: str, odds: float) -> OddsSnapshotIn:
        return OddsSnapshotIn(
            event_id="evt-1",
            bookmaker=book,
            market=Market.SPREADS,
            selection=sel,
            decimal_odds=odds,
            captured_at=NOW - timedelta(seconds=30),
            ingested_at=NOW,
            market_detail="asian_handicap_-1_5",
        )

    snaps = [
        ah("Pinnacle", "Home FC -1.5", 2.00),
        ah("Pinnacle", "Away FC +1.5", 1.95),
        ah("SoftBook", "Home FC -1.5", 2.35),
        ah("SoftBook", "Away FC +1.5", 1.70),
    ]
    sink = RecordingSink()
    picks = await run_value_pipeline(make_deps(sink, FakeLoader(snaps)), "soccer")
    assert len(picks) == 1
    assert picks[0].selection == "Home FC -1.5"
    assert picks[0].market == Market.SPREADS
    assert picks[0].bookmaker == "SoftBook"


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
