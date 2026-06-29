"""H6 (WNBA 'W' marker veto) — wrong-game safety invariants + deferral marker.

P4 coverage found WNBA picks miss a Pinnacle anchor (+5 picks): the OddsPortal
slug drops the league-wide "W" on one side, so "Las Vegas Aces" (slug) vs "Las
Vegas Aces W" (display) trip the one-sided women-marker veto inside the
all-women WNBA league.

H6 is DEFERRED (2026-06-29): the safe scoping the fix requires — an all-women
league confirmed on BOTH sides — is NOT reliably available on the path that
anchors a WNBA Pinnacle close. The live loader (resolve_pinnacle_close_snaps)
and the wrong-game audit BOTH call match_event_hardened with
``league=None, candidate_leagues=None`` (cross-source taxonomies are
incomparable), and the candidate pool is the MIXED ``pinnacle_basketball``
namespace (NBA + WNBA conflated via arcadia_base_sport). A league-gated
suppression is therefore inert on that path; a blanket W-suppression is
forbidden because it would weaken men<->women separation inside that mixed pool.

These tests PIN the invariant the eventual fix must never weaken (a men's team
NEVER matches a women's team, youth/reserve vetoes unchanged) and mark the WNBA
recovery as an expected-fail so a future, safely-scoped fix surfaces as xpass.
"""

from datetime import UTC, datetime

import pytest

from app.resolution.matching import (
    EventCandidate,
    _markers_conflict,
    default_aliases,
    distinguishing_markers,
    match_event_hardened,
)

KO = datetime(2026, 6, 29, 23, 0, tzinfo=UTC)


def _match(qh: str, qa: str, ch: str, ca: str, *, league: str | None = None) -> bool:
    """True iff the hardened matcher anchors the candidate to the query."""
    cand = [EventCandidate(ref="1", home=ch, away=ca, kickoff=KO)]
    cl = {"1": league} if league is not None else None
    matched = match_event_hardened(
        qh,
        qa,
        KO,
        cand,
        aliases=default_aliases(),
        ordered=True,
        league=league,
        candidate_leagues=cl,
    )
    return matched is not None


# --- ABSOLUTE CONSTRAINT: men's X never matches women's X W -------------------
def test_mens_team_never_matches_womens_team_unknown_league() -> None:
    # No league context (the live anchor path): a one-sided "W" MUST veto so a
    # men's pick never anchors onto a women's close (or vice versa).
    assert _markers_conflict(
        "Los Angeles Lakers", "Boston Celtics", "Los Angeles Lakers W", "Boston Celtics W"
    )
    assert not _match(
        "Los Angeles Lakers", "Boston Celtics", "Los Angeles Lakers W", "Boston Celtics W"
    )


def test_mens_team_never_matches_womens_team_mixed_basketball_namespace() -> None:
    # Even with a league string present, a men's-vs-women's "W" difference must
    # still veto: the women marker is the only categorical separator here.
    assert not _match(
        "Phoenix Suns",
        "Dallas Mavericks",
        "Phoenix Suns W",
        "Dallas Mavericks W",
        league="basketball",
    )


def test_one_sided_w_difference_vetoes_when_only_one_side_carries_it() -> None:
    # The exact slug-drop shape, but for a generic (non-WNBA) pairing: a single
    # missing "W" must still veto without a confirmed all-women league.
    assert _markers_conflict("Some Club", "Other Club", "Some Club W", "Other Club")
    assert not _match("Some Club", "Other Club", "Some Club W", "Other Club")


# --- youth / reserve vetoes are UNCHANGED ------------------------------------
def test_youth_marker_veto_still_fires() -> None:
    assert _markers_conflict("Argentina U19", "Brazil U19", "Argentina", "Brazil")
    assert "youth" in distinguishing_markers("Argentina U19")


def test_reserve_marker_veto_still_fires() -> None:
    assert _markers_conflict("Real Madrid B", "Barcelona B", "Real Madrid", "Barcelona")
    assert "reserve" in distinguishing_markers("Real Madrid B")
    assert _markers_conflict("Bayern II", "Dortmund II", "Bayern", "Dortmund")


# --- the DEFERRED hole: WNBA-to-WNBA recovery (expected-fail until a safe fix) -
@pytest.mark.xfail(
    reason="H6 DEFERRED: all-women league not confirmable bilaterally on the "
    "league=None live anchor path / mixed pinnacle_basketball namespace.",
    strict=True,
)
def test_wnba_to_wnba_one_sided_w_should_eventually_match() -> None:
    # A future, safely-scoped fix (e.g. a WNBA-roster confirmation, or a
    # WNBA-specific archive namespace + bilateral league tags) should let the
    # slug-dropped "W" match WNBA-to-WNBA. Until then this stays xfail.
    assert _match("Las Vegas Aces", "New York Liberty", "Las Vegas Aces W", "New York Liberty W")
