"""Golden tests for the fixture-confirmed NAME-FORM alias additions (2026-06-29).

The alias-coverage push (scripts/research/probe_unmatched_split.py) found ~75
unmatched cross-source fixtures whose counterparty DID exist but differed only by
name FORM (sponsor tail / club-type prefix / abbreviation / connector). The
vetted, UNAMBIGUOUS same-club pairs were added to the seed via
scripts/research/exotic_slate_aliases.py + import_alias_datasets.py.

This module locks in BOTH directions of the wrong-game-safety contract:

  POSITIVE  — every auto-added feed<->Pinnacle pair canonicalizes to the SAME
              club (so the strict matcher now links the fixture).
  NEGATIVE  — the new aliases NEVER cause a match across a women/youth/reserve
              marker or between two DISTINCT clubs, and the deliberately-OMITTED
              ambiguous pairs were NOT auto-added.

All pairs are real strings from the live instrument run (instrument.txt).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.resolution import (
    AliasTable,
    EventCandidate,
    default_aliases,
    match_event,
    match_event_hardened,
)

# Feed (OddsPortal display) -> Pinnacle-archive form. Both must canonicalize to
# the SAME normalized club after the alias merge (the strict-tier match).
_AUTO_ADDED: list[tuple[str, str]] = [
    ("Cangrejeros", "Cangrejeros de Santurce"),
    ("Pelita Jaya", "Pelita Jaya Jakarta"),
    ("Taranaki Airs", "Taranaki Mountainairs"),
    ("Berkane", "RS Berkane"),
    ("KAC Marrakech", "Kawkab Marrakech"),
    ("D. Puerto Montt", "Deportes Puerto Montt"),
    ("Puerto Montt", "Deportes Puerto Montt"),
    ("U. Espanola", "Union Espanola"),
    ("Binacional", "Deportivo Binacional"),
    ("Njardvik", "UMF Njardvik"),
    ("Arsenal Sarandi", "Arsenal de Sarandi"),
    ("Heidelberg Utd", "Heidelberg United"),
    ("Broadbeach Utd.", "Broadbeach United"),
    ("Cumberland Utd.", "Cumberland United"),
    ("West Torrens", "West Torrens Birkalla"),
    ("Gjoevik-Lyn", "SK Gjovik-Lyn"),
    ("Flint City", "Flint City Bucks"),
    ("Hudson Valley", "Hudson Valley Hammers"),
    ("Defensores de Vilelas", "Defensores de Puerto Vilelas"),
    ("Kimberley", "Kimberley Mar del Plata"),
    ("Sunnersta", "Sunnersta AIF"),
    ("Grobina", "Grobinas SC/LFS"),
]

# Deliberately NOT auto-added (ambiguous / wrong-game-unsafe) — these MUST stay
# DISTINCT after the merge (see the review/omit list in the run report).
_NOT_ADDED: list[tuple[str, str]] = [
    ("Racing", "Racing Beirut"),  # bare "Racing" — many distinct Racing clubs
    ("Everton", "Everton Vina del Mar"),  # collides with English Everton
    ("Fenix", "Club Atletico Fenix"),  # bare "Fenix" ambiguous
    ("Jazz Pori", "Jazz"),  # cross-sport Jazz (Utah Jazz) collision
    ("Gigantes San Francisco", "Indios de San Francisco de Macoris"),  # distinct clubs
    ("Bayswater", "Bayswater City"),  # bare + disambiguating "City"
    ("Redlands", "Redlands United"),  # bare + disambiguating "United"
    ("Playford Patriots", "Playford City"),  # "City" — possibly distinct clubs
]


@pytest.fixture(scope="module")
def aliases() -> AliasTable:
    return default_aliases()


@pytest.mark.parametrize(("feed", "pinnacle"), _AUTO_ADDED)
def test_nameform_alias_canonicalizes_same_club(
    aliases: AliasTable, feed: str, pinnacle: str
) -> None:
    """Each vetted pair resolves to ONE canonical club (the strict-tier link)."""
    assert aliases.canonical(feed) == aliases.canonical(pinnacle)
    assert aliases.canonical(feed)  # non-empty


@pytest.mark.parametrize(("feed", "pinnacle"), _AUTO_ADDED)
def test_nameform_alias_strict_match_event(aliases: AliasTable, feed: str, pinnacle: str) -> None:
    """The strict exact-on-alias matcher now links the fixture (shared opponent)."""
    ko = datetime(2026, 6, 29, 18, 0, tzinfo=UTC)
    cand = EventCandidate(ref="x", home=pinnacle, away="Opponent United", kickoff=ko)
    matched = match_event(feed, "Opponent United", ko, [cand], aliases=aliases)
    assert matched is cand


@pytest.mark.parametrize(("a", "b"), _NOT_ADDED)
def test_ambiguous_pairs_were_not_auto_added(aliases: AliasTable, a: str, b: str) -> None:
    """The OMITTED/REVIEW pairs stay DISTINCT — no ambiguous auto-merge."""
    assert aliases.canonical(a) != aliases.canonical(b)


@pytest.mark.parametrize(
    "marker_suffix",
    ["Women", "W", "Femenino", "U19", "U20", "Youth", "II", "B", "Reserves"],
)
@pytest.mark.parametrize(("feed", "pinnacle"), _AUTO_ADDED)
def test_nameform_alias_never_crosses_a_marker(
    aliases: AliasTable, feed: str, pinnacle: str, marker_suffix: str
) -> None:
    """A new alias must NEVER link a senior side to a women/youth/reserve side of
    the SAME club. The marker-bearing variant is a DIFFERENT fixture."""
    ko = datetime(2026, 6, 29, 18, 0, tzinfo=UTC)
    # pick = senior feed name; candidate = SAME club + a distinguishing marker.
    cand = EventCandidate(ref="x", home=f"{pinnacle} {marker_suffix}", away="Opponent", kickoff=ko)
    assert match_event_hardened(feed, "Opponent", ko, [cand], aliases=aliases, ordered=True) is None


def test_nameform_alias_never_merges_distinct_clubs(aliases: AliasTable) -> None:
    """Distinct clubs sharing a token / region must not merge via the new aliases."""
    ko = datetime(2026, 6, 29, 18, 0, tzinfo=UTC)
    distinct_pairs = [
        ("RS Berkane", "FUS Rabat"),  # two distinct Botola clubs
        ("Flint City Bucks", "Hudson Valley Hammers"),  # two USL2 clubs
        ("Everton", "Everton de Vina del Mar"),  # English vs Chilean Everton
        ("West Torrens Birkalla", "Western Knights"),  # distinct Australian clubs
        ("Heidelberg United", "Heidelberg Women"),  # marker
    ]
    for home, cand_home in distinct_pairs:
        cand = EventCandidate(ref="x", home=cand_home, away="Common Rival", kickoff=ko)
        assert (
            match_event_hardened(home, "Common Rival", ko, [cand], aliases=aliases, ordered=True)
            is None
        )
