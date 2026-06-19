"""Diagnose the Arcadia↔picks name-match gap (read-only).

`GET /resolution/match-rate` reports `unmatched_with_candidates`: picks where a
Pinnacle `pinnacle_<sport>` archive event exists in the kickoff window but the
STRICT matcher (exact normalized names + alias table + day window) found no
unique match. This prints the actual OddsPortal vs Pinnacle name pairs for those
cases so the specific mismatches (and any systematic pattern) are visible —
the input to expanding `app/resolution/aliases_seed.json`.

Replays the exact query of `repositories.shadow_match_rate_outcomes`; writes
nothing. Run:  uv run python scripts/research/probe_arcadia_match.py
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import aliased

MAX_DAY_DRIFT = 1


async def _main() -> None:
    from app.config import get_settings
    from app.database import create_engine, create_session_factory
    from app.resolution import EventCandidate, default_aliases, match_event
    from app.resolution.matching import normalize_name
    from app.resolution.shadow import arcadia_base_sport
    from app.storage.models import Event, League, Pick, Sport, Team

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    aliases = default_aliases()
    window = timedelta(days=MAX_DAY_DRIFT + 1)

    try:
        async with session_factory() as session:
            home_t, away_t = aliased(Team), aliased(Team)
            pick_rows = (
                await session.execute(
                    select(
                        Pick.id, Sport.key, League.key, home_t.name, away_t.name, Event.starts_at
                    )
                    .select_from(Pick)
                    .join(Event, Pick.event_id == Event.id)
                    .join(Sport, Event.sport_id == Sport.id)
                    .join(League, Event.league_id == League.id)
                    .join(home_t, Event.home_team_id == home_t.id)
                    .join(away_t, Event.away_team_id == away_t.id)
                    .where(Event.starts_at.is_not(None))
                )
            ).all()

            by_ns: dict[str, list] = {}
            for row in pick_rows:
                by_ns.setdefault(f"pinnacle_{arcadia_base_sport(row[1])}", []).append(row)

            unmatched: list[tuple] = []
            no_cands = 0
            matched = 0
            for ns, picks in by_ns.items():
                kickoffs = [p[5] for p in picks]
                ah, aw = aliased(Team), aliased(Team)
                arc_rows = (
                    await session.execute(
                        select(ah.name, aw.name, Event.starts_at)
                        .join(Sport, Event.sport_id == Sport.id)
                        .join(ah, Event.home_team_id == ah.id)
                        .join(aw, Event.away_team_id == aw.id)
                        .where(
                            Sport.key == ns,
                            Event.starts_at.is_not(None),
                            Event.starts_at >= min(kickoffs) - window,
                            Event.starts_at <= max(kickoffs) + window,
                        )
                    )
                ).all()
                archive = [
                    EventCandidate(ref=str(i), home=h, away=a, kickoff=ko)
                    for i, (h, a, ko) in enumerate(arc_rows)
                ]
                for _pid, sk, lk, home, away, ko in picks:
                    in_window = [
                        c
                        for c in archive
                        if abs((c.kickoff.date() - ko.date()).days) <= MAX_DAY_DRIFT
                    ]
                    is_match = (
                        match_event(
                            home, away, ko, in_window, aliases=aliases, max_day_drift=MAX_DAY_DRIFT
                        )
                        is not None
                    )
                    if is_match:
                        matched += 1
                    elif in_window:
                        unmatched.append((sk, lk, home, away, ko, in_window))
                    else:
                        no_cands += 1
    finally:
        await engine.dispose()

    print(
        f"\npicks={matched + no_cands + len(unmatched)}  matched={matched}  "
        f"no_archive_candidates={no_cands}  unmatched_with_candidates={len(unmatched)}\n"
    )

    # An UNMATCHED pick is an ALIAS gap only if some in-window Pinnacle event is
    # PLAUSIBLY the same fixture (shares >=1 normalized team token with the pick)
    # but under a name the strict matcher rejected. If NO candidate shares a
    # token, the "candidates" are just same-day noise and it's a pure COVERAGE
    # gap (Pinnacle doesn't carry that fixture) — aliases would not help.
    def _tokens(name: str) -> set[str]:
        return set(normalize_name(name).split())

    alias_suspects = 0
    coverage_only = 0
    print("=== UNMATCHED-WITH-CANDIDATES (real alias suspects only) ===")
    for sk, lk, home, away, ko, cands in unmatched:
        pick_tokens = _tokens(home) | _tokens(away)
        suspects = [c for c in cands if (_tokens(c.home) | _tokens(c.away)) & pick_tokens]
        if not suspects:
            coverage_only += 1
            continue
        alias_suspects += 1
        print(f"\n[{sk} / {lk}]  {ko:%Y-%m-%d %H:%M}")
        print(f"  ODDSPORTAL : {home!r} vs {away!r}")
        print(f"     norm    : {normalize_name(home)!r} vs {normalize_name(away)!r}")
        for c in suspects[:5]:
            dd = (c.kickoff.date() - ko.date()).days
            print(f"  PINNACLE   : {c.home!r} vs {c.away!r}  (day {dd:+d})")
            print(f"     norm    : {normalize_name(c.home)!r} vs {normalize_name(c.away)!r}")

    print(
        f"\nSUMMARY of the {len(unmatched)} unmatched-with-candidates: "
        f"{alias_suspects} have a plausible same-fixture candidate (ALIAS gap), "
        f"{coverage_only} are same-day noise only (COVERAGE gap — Pinnacle "
        f"doesn't carry the fixture; aliases would NOT help)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
