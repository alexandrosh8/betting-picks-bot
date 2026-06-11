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
        return "inserted" if calls["n"] == 1 else "duplicate"

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


def test_pick_tier_boundaries() -> None:
    """Tier floors are INCLUSIVE (>= mirrors the backtests' gates): edge
    exactly 0.03 is premium, a hair under is volume, under 0.015 is no pick;
    equal floors disable the volume tier entirely."""
    from app.pipeline import pick_tier

    assert pick_tier(0.03, 0.03, 0.015) == "premium"
    assert pick_tier(0.0299, 0.03, 0.015) == "volume"
    assert pick_tier(0.015, 0.03, 0.015) == "volume"
    assert pick_tier(0.0149, 0.03, 0.015) is None
    assert pick_tier(0.02, 0.03, 0.03) is None  # equal floors: tier off
    assert pick_tier(0.03, 0.03, 0.03) == "premium"


def patch_persist_recording(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[str]
) -> list[tuple[str, str]]:
    """persist_pick fake returning scripted outcomes; records (selection,
    tier) per call so tests can assert what reached the repository."""
    import app.storage.repositories as repos

    seen: list[tuple[str, str]] = []
    script = iter(outcomes)

    async def fake_persist_pick(session, pick, teams, model_name, model_version):  # type: ignore[no-untyped-def]
        seen.append((pick.selection, pick.tier))
        return next(script)

    monkeypatch.setattr(repos, "persist_pick", fake_persist_pick)
    return seen


async def test_volume_tier_pick_persists_without_alert_or_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shadow tier's contract: persisted (with the informational stake
    breakdown computed) but (a) NO alert dispatch and (b) NO exposure-ledger
    reservation — it must never consume the cap premium picks need."""
    seen = patch_persist_recording(monkeypatch, ["inserted"])

    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))
    deps.session_factory = FakeSessionFactory()  # type: ignore[assignment]
    deps.value_min_edge = 0.10  # the ~4.5% edge cannot reach premium
    deps.value_volume_min_edge = 0.015

    day = datetime.now(tz=UTC).date()
    picks = await run_value_pipeline(deps, "soccer")

    assert [p.tier for p in picks] == ["volume"]
    assert seen == [("Home FC", "volume")]
    assert sink.sent == []  # (a) never alerted
    assert deps.ledger.used(day) == 0.0  # (b) never on the ledger
    assert picks[0].stake_breakdown.final > 0.0  # stake computed, informational
    from app.pipeline import LAST_POLL

    assert LAST_POLL["soccer"]["picks"] == 0  # headline count stays premium
    assert LAST_POLL["soccer"]["volume_picks"] == 1


async def test_volume_tier_dropped_when_persistence_unavailable() -> None:
    # A volume pick that cannot reach the DB accumulates no CLV evidence —
    # its only purpose — so it is dropped silently: no pick, no alert.
    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))  # no session_factory
    deps.value_min_edge = 0.10
    deps.value_volume_min_edge = 0.015
    picks = await run_value_pipeline(deps, "soccer")
    assert picks == []
    assert sink.sent == []
    assert deps.ledger.used(datetime.now(tz=UTC).date()) == 0.0


async def test_volume_redetection_of_existing_key_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'duplicate' covers both a volume re-detection AND a key already held
    by a PREMIUM row — the shadow tier must never alert, never touch the
    ledger, and never displace the premium row."""
    patch_persist_recording(monkeypatch, ["duplicate"])

    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))
    deps.session_factory = FakeSessionFactory()  # type: ignore[assignment]
    deps.value_min_edge = 0.10
    deps.value_volume_min_edge = 0.015

    picks = await run_value_pipeline(deps, "soccer")
    assert picks == []
    assert sink.sent == []
    assert deps.ledger.used(datetime.now(tz=UTC).date()) == 0.0


async def test_volume_to_premium_upgrade_alerts_and_reserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upgrade transition: a key first persisted as volume later clears
    the premium threshold -> the repository promotes the row ('upgraded')
    and the pipeline treats it as a NEW premium pick — alert dispatched,
    exposure reserved (the shadow row never held a reservation)."""
    seen = patch_persist_recording(monkeypatch, ["inserted", "upgraded"])

    sink = RecordingSink()
    deps = make_deps(sink, FakeLoader(market_snapshots()))
    deps.session_factory = FakeSessionFactory()  # type: ignore[assignment]
    deps.value_min_edge = 0.10  # cycle 1: candidate lands in the volume band
    deps.value_volume_min_edge = 0.015

    day = datetime.now(tz=UTC).date()
    first = await run_value_pipeline(deps, "soccer")
    assert [p.tier for p in first] == ["volume"]
    assert sink.sent == []
    assert deps.ledger.used(day) == 0.0

    # cycle 2: the same candidate now clears premium (threshold change here;
    # a price move in production) — the volume row upgrades in place.
    deps.value_min_edge = 0.03
    second = await run_value_pipeline(deps, "soccer")
    assert [p.tier for p in second] == ["premium"]
    assert seen == [("Home FC", "volume"), ("Home FC", "premium")]
    assert len(sink.sent) == 1  # the upgrade IS the alert moment
    assert deps.ledger.used(day) > 0.0  # exposure reserved on upgrade


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


def _detail_snap(
    book: str, market: Market, sel: str, odds: float, detail: str | None
) -> OddsSnapshotIn:
    return OddsSnapshotIn(
        event_id="evt-1",
        bookmaker=book,
        market=market,
        selection=sel,
        decimal_odds=odds,
        captured_at=NOW - timedelta(seconds=30),
        ingested_at=NOW,
        market_detail=detail,
    )


def test_event_fair_probs_expanded_markets_devig_per_line_and_derive_dc() -> None:
    """The expanded market set round-trips devig per (market, line) group:
    every direct group sums to 1.0 within ITS line; double chance is never
    devigged directly (legs overlap, quotes sum ~200%) — its fair value is
    DERIVED from the 1X2 anchor's pairwise sums."""
    from app.pipeline import event_fair_probs, group_market_prices
    from app.probabilities.devig import DevigMethod

    snaps = [
        # 1X2 anchor (full 3-way at the sharp book)
        _detail_snap("Pinnacle", Market.H2H, "Home FC", 2.50, None),
        _detail_snap("Pinnacle", Market.H2H, "Draw", 3.30, None),
        _detail_snap("Pinnacle", Market.H2H, "Away FC", 3.10, None),
        # two totals lines — must anchor as separate 2-way books
        _detail_snap("Pinnacle", Market.TOTALS, "Over 2.5", 1.95, "over_under_2_5"),
        _detail_snap("Pinnacle", Market.TOTALS, "Under 2.5", 1.95, "over_under_2_5"),
        _detail_snap("Pinnacle", Market.TOTALS, "Over 3.5", 2.80, "over_under_3_5"),
        _detail_snap("Pinnacle", Market.TOTALS, "Under 3.5", 1.45, "over_under_3_5"),
        # 3-way European handicap line (devig-sound at any integer line)
        _detail_snap("Pinnacle", Market.SPREADS, "Home FC -1", 3.10, "european_handicap_-1"),
        _detail_snap("Pinnacle", Market.SPREADS, "Draw (-1)", 3.60, "european_handicap_-1"),
        _detail_snap("Pinnacle", Market.SPREADS, "Away FC +1", 2.10, "european_handicap_-1"),
        # double-chance quotes: NEVER a direct devig input
        _detail_snap("SoftBook", Market.DOUBLE_CHANCE, "Home FC or Draw", 1.42, "double_chance"),
        _detail_snap("SoftBook", Market.DOUBLE_CHANCE, "Home FC or Away FC", 1.36, "double_chance"),
        _detail_snap("SoftBook", Market.DOUBLE_CHANCE, "Draw or Away FC", 1.60, "double_chance"),
    ]
    fair = event_fair_probs(group_market_prices(snaps), DevigMethod.POWER)

    for market, detail, n_outcomes in (
        (Market.H2H, None, 3),
        (Market.TOTALS, "over_under_2_5", 2),
        (Market.TOTALS, "over_under_3_5", 2),
        (Market.SPREADS, "european_handicap_-1", 3),
    ):
        anchor_book, by_sel = fair[("evt-1", market, detail)]
        assert anchor_book == "Pinnacle"
        assert len(by_sel) == n_outcomes
        assert sum(by_sel.values()) == pytest.approx(1.0)
    # symmetric 2.5-line book devigs to exactly 0.5 within its OWN line
    assert fair[("evt-1", Market.TOTALS, "over_under_2_5")][1]["Over 2.5"] == pytest.approx(0.5)

    h2h_fair = fair[("evt-1", Market.H2H, None)][1]
    dc_anchor, dc_fair = fair[("evt-1", Market.DOUBLE_CHANCE, "double_chance")]
    assert dc_anchor == "Pinnacle"  # inherited from the 1X2 anchor
    assert dc_fair["Home FC or Draw"] == pytest.approx(h2h_fair["Home FC"] + h2h_fair["Draw"])
    assert dc_fair["Home FC or Away FC"] == pytest.approx(h2h_fair["Home FC"] + h2h_fair["Away FC"])
    assert dc_fair["Draw or Away FC"] == pytest.approx(h2h_fair["Draw"] + h2h_fair["Away FC"])
    # overlapping legs by design: DC fair sums to 2.0, not 1.0
    assert sum(dc_fair.values()) == pytest.approx(2.0)
