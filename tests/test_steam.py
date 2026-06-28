"""Line-movement / steam-awareness gate (app/edge/steam.py) — pure math.

Invariants under test:
  - soft price converging TOWARD the anchor (edge correcting) trips the gate;
  - a STALE anchor (last observation older than the freshness window) trips it;
  - a FRESH anchor + a soft book holding its edge PASSES;
  - a soft book steaming sharply AWAY is FLAGGED but does not trip by default;
  - no future leakage: a captured_at > now is ignored (pick-time discipline);
  - insufficient history is conservative (cannot judge -> no trip).
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.edge.steam import (
    SteamPolicy,
    build_trajectories,
    evaluate_steam,
    implied_prob,
)
from app.schemas.base import Market
from app.schemas.odds import OddsSnapshotIn

NOW = datetime.now(tz=UTC)


def _t(seconds_ago: float) -> datetime:
    return NOW - timedelta(seconds=seconds_ago)


POLICY = SteamPolicy(
    enabled=False,
    lookback_seconds=21600.0,
    min_points=2,
    soft_toward_anchor_close_frac=0.5,
    min_initial_gap=0.01,
    anchor_staleness_seconds=7200.0,
    soft_steam_away_delta=0.04,
)


# --- implied_prob boundary --------------------------------------------------


def test_implied_prob_rejects_degenerate_odds() -> None:
    with pytest.raises(ValueError):
        implied_prob(1.0)
    with pytest.raises(ValueError):
        implied_prob(0.5)
    assert implied_prob(2.0) == pytest.approx(0.5)


# --- soft converging toward the anchor (edge correcting) --------------------


def test_soft_moving_toward_anchor_trips() -> None:
    # Anchor steady at 2.00 (implied 0.50). Soft opened generous at 2.50
    # (implied 0.40, gap 0.10) and has CORRECTED to 2.10 (implied ~0.476,
    # gap ~0.024) -> ~76% of the original edge already gone -> trip.
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 2.50), (_t(60), 2.10)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is True
    assert v.tripped is True
    assert "soft_toward_anchor" in v.reasons
    assert v.closed_fraction is not None and v.closed_fraction >= 0.5


def test_soft_holding_its_edge_passes() -> None:
    # Anchor steady 2.00; soft holds 2.50 the whole window -> gap unchanged,
    # nothing corrected, anchor fresh -> pass.
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 2.50), (_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is False
    assert v.stale_anchor is False
    assert v.tripped is False
    assert v.reasons == ()


def test_partial_convergence_below_fraction_passes() -> None:
    # Only ~24% of the edge closed (2.50 -> 2.40), below the 0.5 fraction -> pass.
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 2.50), (_t(60), 2.40)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is False
    assert v.tripped is False


def test_anchor_dropping_toward_soft_also_closes_gap() -> None:
    # The soft holds 2.50; the ANCHOR drifts up 2.00 -> 2.45 (implied 0.50 ->
    # 0.408), collapsing the gap from the anchor side. The edge is evaporating
    # regardless of which side moved -> trip.
    anchor = [(_t(3600), 2.00), (_t(60), 2.45)]
    fill = [(_t(3600), 2.50), (_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is True
    assert v.tripped is True


def test_no_initial_edge_does_not_trip_on_convergence() -> None:
    # Soft opened at/above the anchor's implied (no value to begin with): the
    # convergence signal must not fire (gap_open below min_initial_gap).
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 1.99), (_t(60), 2.05)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is False
    assert v.tripped is False


# --- stale anchor -----------------------------------------------------------


def test_stale_anchor_trips() -> None:
    # Anchor's last observation is 3h old (> 2h window); soft holds its edge.
    anchor = [(_t(14000), 2.00), (_t(10800), 2.00)]
    fill = [(_t(3600), 2.50), (_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.stale_anchor is True
    assert v.tripped is True
    assert "stale_anchor" in v.reasons
    assert v.anchor_age_seconds is not None and v.anchor_age_seconds > 7200.0


def test_fresh_anchor_not_stale() -> None:
    anchor = [(_t(3600), 2.00), (_t(120), 2.00)]
    fill = [(_t(3600), 2.50), (_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.stale_anchor is False


def test_empty_anchor_is_treated_as_stale() -> None:
    # No anchor observation in the window at all: cannot confirm freshness ->
    # conservative stale (a phantom edge against an unverifiable anchor).
    fill = [(_t(3600), 2.50), (_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=[], now=NOW, policy=POLICY)
    assert v.stale_anchor is True
    assert v.tripped is True


# --- soft steamed away (flag only) ------------------------------------------


def test_soft_steamed_away_flags_but_does_not_trip_by_default() -> None:
    # Soft lengthens 2.00 -> 2.60 (implied 0.50 -> 0.385, a 0.115 drop): a sharp
    # move AWAY (maybe reacting to real info). Flagged, NOT tripped by default.
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 2.00), (_t(60), 2.60)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_steamed is True
    assert "soft_steamed_away" in v.reasons
    assert v.soft_toward_anchor is False
    assert v.tripped is False


def test_soft_steamed_away_trips_when_policy_demotes_on_it() -> None:
    policy = SteamPolicy(demote_on_soft_steam=True)
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill = [(_t(3600), 2.00), (_t(60), 2.60)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=policy)
    assert v.soft_steamed is True
    assert v.tripped is True


# --- leakage & insufficient-data discipline ---------------------------------


def test_future_points_are_ignored_no_leakage() -> None:
    # A future-stamped row (captured_at > now) must NEVER enter the judgement —
    # pick-time discipline. With only the past point left, fill has < min_points
    # and the gate cannot judge convergence.
    anchor = [(_t(3600), 2.00), (_t(60), 2.00)]
    fill_with_future = [(_t(60), 2.50), (NOW + timedelta(seconds=120), 2.05)]
    v = evaluate_steam(
        fill_trajectory=fill_with_future, anchor_trajectory=anchor, now=NOW, policy=POLICY
    )
    assert v.soft_toward_anchor is False  # the "correction" was a future row


def test_insufficient_history_is_conservative_pass() -> None:
    # Only the current point for the fill book: no movement can be measured ->
    # no convergence trip (anchor fresh -> not stale either).
    anchor = [(_t(60), 2.00)]
    fill = [(_t(60), 2.50)]
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=POLICY)
    assert v.soft_toward_anchor is False
    assert v.stale_anchor is False
    assert v.tripped is False


def test_points_outside_lookback_window_excluded() -> None:
    # An old "open" row beyond the lookback is dropped: the visible fill series
    # then has only the recent point -> cannot judge -> pass.
    policy = SteamPolicy(lookback_seconds=1800.0)  # 30 min
    anchor = [(_t(600), 2.00)]
    fill = [(_t(7200), 2.50), (_t(120), 2.10)]  # 2h-old open is outside 30min
    v = evaluate_steam(fill_trajectory=fill, anchor_trajectory=anchor, now=NOW, policy=policy)
    assert v.soft_toward_anchor is False


# --- build_trajectories (pure grouping helper) ------------------------------


def _snap(
    book: str, sel: str, odds: float, age_s: float, detail: str | None = None
) -> OddsSnapshotIn:
    return OddsSnapshotIn(
        event_id="evt-1",
        bookmaker=book,
        market=Market.H2H,
        selection=sel,
        decimal_odds=odds,
        captured_at=_t(age_s),
        ingested_at=NOW,
        market_detail=detail,
    )


def test_build_trajectories_groups_and_sorts() -> None:
    snaps = [
        _snap("Pinnacle", "Home FC", 2.00, 60),
        _snap("Pinnacle", "Home FC", 2.05, 3600),
        _snap("SoftBook", "Home FC", 2.50, 120),
    ]
    traj = build_trajectories(snaps, NOW, lookback_seconds=21600.0)
    key = ("evt-1", Market.H2H, None)
    assert key in traj
    pin = traj[key][("Home FC", "pinnacle")]
    assert [o for _, o in pin] == [2.05, 2.00]  # sorted oldest-first
    assert ("Home FC", "softbook") in traj[key]


def test_build_trajectories_drops_out_of_window_and_future() -> None:
    snaps = [
        _snap("Pinnacle", "Home FC", 2.00, 60),
        _snap("Pinnacle", "Home FC", 2.20, 999999),  # ancient -> dropped
        OddsSnapshotIn(
            event_id="evt-1",
            bookmaker="Pinnacle",
            market=Market.H2H,
            selection="Home FC",
            decimal_odds=2.30,
            captured_at=NOW + timedelta(seconds=300),  # future -> dropped
            ingested_at=NOW,
        ),
    ]
    traj = build_trajectories(snaps, NOW, lookback_seconds=21600.0)
    pin = traj[("evt-1", Market.H2H, None)][("Home FC", "pinnacle")]
    assert [o for _, o in pin] == [2.00]
