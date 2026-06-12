"""Timing-window analyzer (scripts/reports/timing_windows.py) on synthetic
snapshot fixtures: bucket boundaries, state reconstruction under change-only
persistence, drift math, hypothetical edge-then vs realized CLV, the
qualified-event rule, and the INSUFFICIENT DATA gate.

Loads the script by path (scripts/ is not a package) — the same pattern as
tests/test_ml_dataset.py. No network, no DB: pure functions only.
"""

import importlib.util
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.probabilities.devig import DevigMethod

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reports" / "timing_windows.py"
_spec = importlib.util.spec_from_file_location("timing_windows", _SCRIPT)
assert _spec is not None and _spec.loader is not None
tw: Any = importlib.util.module_from_spec(_spec)
sys.modules["timing_windows"] = tw
_spec.loader.exec_module(tw)

KICKOFF = datetime(2026, 6, 10, 18, 0, tzinfo=UTC)


def snap(book: str, sel: str, odds: float, hours_before: float, market: str = "1x2") -> Any:
    return tw.SnapshotRow(
        bookmaker=book,
        market=market,
        selection=sel,
        decimal_odds=odds,
        captured_at=KICKOFF - timedelta(hours=hours_before),
    )


def test_bucket_label_boundaries() -> None:
    assert tw.bucket_label(72.0) == ">48h"
    assert tw.bucket_label(48.0) == ">48h"  # lo-inclusive
    assert tw.bucket_label(47.999) == "24-48h"
    assert tw.bucket_label(24.0) == "24-48h"
    assert tw.bucket_label(5.0) == "5-24h"
    assert tw.bucket_label(1.0) == "1-5h"
    assert tw.bucket_label(0.0) == "<1h"
    assert tw.bucket_label(-0.1) is None  # post-kickoff


def test_state_at_takes_last_observation_per_key() -> None:
    rows = [
        snap("SoftBook", "home", 2.50, 72.0),
        snap("SoftBook", "home", 2.30, 10.0),
        snap("SoftBook", "home", 2.10, 0.5),
    ]
    # change-only persistence: an old row IS the state until a newer one
    state = tw.state_at(rows, KICKOFF - timedelta(hours=24))
    assert state["1x2"]["home"]["SoftBook"] == 2.50
    assert tw.state_at(rows, KICKOFF)["1x2"]["home"]["SoftBook"] == 2.10
    # degenerate odds never enter a state
    assert tw.state_at([snap("B", "home", 1.0, 2.0)], KICKOFF) == {}


def test_event_qualifies_needs_two_times_and_a_close() -> None:
    two_times_fresh = tw.EventSnapshots(
        "A vs B", KICKOFF, (snap("S", "home", 2.5, 30.0), snap("S", "home", 2.4, 0.5))
    )
    assert tw.event_qualifies(two_times_fresh)
    one_time = tw.EventSnapshots("A vs B", KICKOFF, (snap("S", "home", 2.5, 0.5),))
    assert not tw.event_qualifies(one_time)
    stale_close = tw.EventSnapshots(
        "A vs B", KICKOFF, (snap("S", "home", 2.5, 30.0), snap("S", "home", 2.4, 5.0))
    )
    assert not tw.event_qualifies(stale_close)  # last capture 5h > 4h gap


def test_drift_is_log_close_over_then_excluding_the_close_row() -> None:
    event = tw.EventSnapshots(
        "A vs B",
        KICKOFF,
        (
            snap("SoftBook", "home", 2.50, 72.0),  # >48h
            snap("SoftBook", "home", 2.40, 30.0),  # 24-48h
            snap("SoftBook", "home", 2.30, 10.0),  # 5-24h
            snap("SoftBook", "home", 2.20, 2.0),  # 1-5h
            snap("SoftBook", "home", 2.10, 0.5),  # the close row -> excluded
        ),
    )
    drift = tw.drift_by_bucket(event)
    assert drift[">48h"] == [pytest.approx(math.log(2.10 / 2.50))]
    assert drift["24-48h"] == [pytest.approx(math.log(2.10 / 2.40))]
    assert drift["5-24h"] == [pytest.approx(math.log(2.10 / 2.30))]
    assert drift["1-5h"] == [pytest.approx(math.log(2.10 / 2.20))]
    assert "<1h" not in drift  # only the close itself lived there


def _value_event() -> Any:
    # Pinnacle holds 2.00/2.00 (fair 0.5/0.5) from 72h out. SoftBook posts a
    # soft 2.30 on home at 30h (edge 0.5 - 1/2.30 = +0.065) that closes at
    # 2.05 (edge +0.012 < 0.03): detection at 24-48h catches CLV the <1h
    # window no longer offers.
    return tw.EventSnapshots(
        "A vs B",
        KICKOFF,
        (
            snap("Pinnacle", "home", 2.00, 72.0),
            snap("Pinnacle", "away", 2.00, 72.0),
            snap("SoftBook", "home", 2.30, 30.0),
            snap("SoftBook", "away", 1.80, 30.0),
            snap("SoftBook", "home", 2.05, 0.5),
        ),
    )


def test_edge_then_vs_realized_clv() -> None:
    picks = tw.edge_vs_clv_by_bucket(
        _value_event(), min_edge=0.03, min_odds=1.30, devig_method=DevigMethod.POWER
    )
    assert set(picks) == {"24-48h"}  # >48h has no soft book yet; <1h edge gone
    (edge_then, clv), *rest = picks["24-48h"]
    assert not rest
    assert edge_then == pytest.approx(0.5 - 1 / 2.30, abs=1e-9)
    # realized CLV vs the devigged close (Pinnacle 2.00/2.00 -> fair 0.5):
    # ln(2.30 * 0.5) — effective == raw at commission-free books.
    assert clv == pytest.approx(math.log(2.30 * 0.5), abs=1e-9)


def test_analyze_events_counts_only_qualified_events() -> None:
    unqualified = tw.EventSnapshots("C vs D", KICKOFF, (snap("S", "home", 2.0, 0.5),))
    qualified, aggs = tw.analyze_events(
        [_value_event(), unqualified],
        min_edge=0.03,
        min_odds=1.30,
        devig_method=DevigMethod.POWER,
    )
    assert qualified == 1
    assert len(aggs["24-48h"].picks) == 1


def test_report_withholds_estimates_until_min_events() -> None:
    qualified, aggs = tw.analyze_events(
        [_value_event()], min_edge=0.03, min_odds=1.30, devig_method=DevigMethod.POWER
    )
    report = tw.render_report(
        qualified, aggs, min_events=30, min_edge=0.03, devig_method=DevigMethod.POWER
    )
    assert "INSUFFICIENT DATA: n=1 events with >=2 snapshot times and a close" in report
    assert "withheld" in report
    # counts still visible (depth must be watchable week over week)
    assert "24-48h" in report

    full = tw.render_report(
        qualified, aggs, min_events=1, min_edge=0.03, devig_method=DevigMethod.POWER
    )
    assert "INSUFFICIENT DATA" not in full
    assert "withheld" not in full
