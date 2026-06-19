"""Import withqwerty/reep (CC0) soccer team-name aliases into our cross-source
alias seed, BOUNDED to teams we actually scrape.

WHY
  The shadow match-rate harness keeps surfacing soccer fixtures lost purely to a
  NAME gap (OddsPortal "Köln" vs Pinnacle "1. FC Köln"). reep ships a CC0
  Wikidata-derived club alias set; the bounded subset that touches OUR scraped
  teams is free, high-confidence cross-source bridge data.

DOCTRINE (read before editing)
  - DATA-only. The matcher stays STRICT: exact normalized canonical<->alias only.
    This script NEVER introduces a fuzzy/containment pair. Every emitted pair is
    an EXACT alias whose normalize_name() differs from the canonical's, both
    drawn from reep, and at least one surface form of the reep group exactly
    (normalized) equals a team we actually scrape.
  - BOUNDED. A reep club group is imported only if SOME surface name in it
    (canonical or any alias) normalize_name()-equals a scraped soccer team. This
    keeps the seed small and relevant — 45k reep teams collapse to the few we see.
  - COLLISION-SAFE. A new alias->canonical pair is SKIPPED (and logged) if its
    normalized alias already maps to a DIFFERENT canonical (in the existing seed
    or earlier in this import), or if it would collapse two DISTINCT scraped
    teams into one. A wrong close corrupts CLV — we drop the pair rather than risk it.

USAGE
  uv run python scripts/research/import_reep_soccer_aliases.py            # dry-run report
  uv run python scripts/research/import_reep_soccer_aliases.py --write    # rewrite the seed

The reep CSV is fetched read-only from raw.githubusercontent.com (LICENSE = CC0
1.0; data/meta.json source = "Wikidata SPARQL + custom verified mappings").
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import urllib.request
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.resolution.matching import _SEED_PATH, normalize_name
from app.storage.models import Sport, Team

# reep CC0 names file. master is the published branch (no releases/tags as of
# import); data_version is recorded in data/meta.json.
_REEP_NAMES_URL = "https://raw.githubusercontent.com/withqwerty/reep/master/data/names.csv"
_REEP_TEAM_PREFIX = "reep_t"  # team entities; reep_p = people, reep_c = competitions

# Local Postgres (read-only). Soccer is captured under TWO sport keys — the
# OddsPortal side ("soccer") and the Pinnacle archive side ("pinnacle_soccer");
# the whole point of the alias table is to bridge those two, so we union both.
_DB_URL = "postgresql+asyncpg://betting_ai:betting_ai@localhost:5433/betting_ai"
_SOCCER_SPORT_KEYS = ("soccer", "pinnacle_soccer")


def _fetch_reep_team_groups() -> dict[str, list[str]]:
    """`{canonical reep name -> [alias, ...]}` for team entities only.

    reep rows are `reep_id,key_wikidata,name,alias`; a team has one row per alias,
    all sharing the same `name`. We key by the human-readable `name` (the seed
    convention) and gather its aliases.
    """
    with urllib.request.urlopen(_REEP_NAMES_URL, timeout=60) as resp:  # noqa: S310 (fixed https host)
        text = resp.read().decode("utf-8")
    groups: dict[str, list[str]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if not row["reep_id"].startswith(_REEP_TEAM_PREFIX):
            continue
        name = (row.get("name") or "").strip()
        alias = (row.get("alias") or "").strip()
        if not name or not alias:
            continue
        groups[name].append(alias)
    return dict(groups)


async def _fetch_scraped_soccer_norms() -> set[str]:
    """Normalized forms of every soccer team name we actually scrape (both sources)."""
    engine = create_async_engine(_DB_URL)
    try:
        session_maker = async_sessionmaker(engine, expire_on_commit=False)
        async with session_maker() as session:
            sport_ids = (
                (await session.execute(select(Sport.id).where(Sport.key.in_(_SOCCER_SPORT_KEYS))))
                .scalars()
                .all()
            )
            names = (
                (
                    await session.execute(
                        select(Team.name).where(Team.sport_id.in_(sport_ids)).distinct()
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()
    return {n for n in (normalize_name(name) for name in names) if n}


def _load_existing_seed() -> dict[str, list[str]]:
    data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    teams: dict[str, list[str]] = data.get("teams", {})
    return teams


def build_import(
    reep_groups: dict[str, list[str]],
    scraped_norms: set[str],
    existing_seed: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    """Return (new_pairs_by_canonical, collisions).

    new_pairs_by_canonical: `{canonical_label -> [alias_label, ...]}` to ADD.
    collisions: `(alias, attempted_canonical, blocking_canonical)` skipped pairs.
    """
    # Existing alias->canonical map, fully normalized, including each canonical's
    # own normalized self-key (a canonical is its own alias). `canon_label_by_norm`
    # remembers the EXISTING human-readable label per normalized canonical so a
    # reep group whose canonical normalizes to one we already have merges UNDER
    # the existing label instead of minting a second key (which would trip
    # test_seed_alias_canonicals_do_not_collide).
    alias_to_canon: dict[str, str] = {}
    canon_label_by_norm: dict[str, str] = {}
    for canonical, aliases in existing_seed.items():
        canon_norm = normalize_name(canonical)
        if not canon_norm:
            continue
        canon_label_by_norm.setdefault(canon_norm, canonical)
        for surface in [canonical, *aliases]:
            s = normalize_name(surface)
            if s:
                alias_to_canon[s] = canon_norm

    new_pairs: dict[str, list[str]] = defaultdict(list)
    collisions: list[tuple[str, str, str]] = []
    # Deterministic order so the emitted seed is stable across runs.
    for reep_canonical in sorted(reep_groups):
        aliases = reep_groups[reep_canonical]
        canon_norm = normalize_name(reep_canonical)
        if not canon_norm:
            continue
        surfaces_norm = {canon_norm, *(normalize_name(a) for a in aliases)}
        surfaces_norm.discard("")
        # BOUND: import only groups that touch a team we actually scrape.
        if surfaces_norm.isdisjoint(scraped_norms):
            continue
        # The canonical self-key must not already belong to a DIFFERENT canonical.
        prior = alias_to_canon.get(canon_norm)
        if prior is not None and prior != canon_norm:
            collisions.append((reep_canonical, reep_canonical, prior))
            continue
        # Reuse the existing label for this normalized canonical when one exists
        # (e.g. seed "San Martin de San Juan" vs reep "San Martín de San Juan")
        # so we extend that entry rather than create an accent-variant duplicate.
        canonical = canon_label_by_norm.setdefault(canon_norm, reep_canonical)
        alias_to_canon.setdefault(canon_norm, canon_norm)
        for alias in aliases:
            a_norm = normalize_name(alias)
            if not a_norm or a_norm == canon_norm:
                continue  # empty or already-resolvable -> nothing new
            existing_canon = alias_to_canon.get(a_norm)
            if existing_canon is not None:
                if existing_canon != canon_norm:
                    collisions.append((alias, canonical, existing_canon))
                continue  # already maps here (dedupe) or elsewhere (collision, logged)
            alias_to_canon[a_norm] = canon_norm
            if alias not in new_pairs[canonical]:
                new_pairs[canonical].append(alias)
    return {k: v for k, v in new_pairs.items() if v}, collisions


def merge_into_seed(
    existing_seed: dict[str, list[str]], new_pairs: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {k: list(v) for k, v in existing_seed.items()}
    for canonical, aliases in new_pairs.items():
        bucket = merged.setdefault(canonical, [])
        for alias in aliases:
            if alias not in bucket:
                bucket.append(alias)
    return dict(sorted(merged.items()))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite aliases_seed.json")
    args = parser.parse_args()

    reep_groups = _fetch_reep_team_groups()
    scraped_norms = await _fetch_scraped_soccer_norms()
    existing_seed = _load_existing_seed()

    new_pairs, collisions = build_import(reep_groups, scraped_norms, existing_seed)
    n_new_aliases = sum(len(v) for v in new_pairs.values())

    print(f"reep team entities (with aliases) : {len(reep_groups)}")
    print(f"scraped soccer normalized names   : {len(scraped_norms)}")
    print(f"canonicals gaining aliases        : {len(new_pairs)}")
    print(f"NEW alias->canonical pairs        : {n_new_aliases}")
    print(f"collisions skipped                : {len(collisions)}")
    for alias, attempted, blocking in collisions:
        print(f"  SKIP collision: {alias!r} -> {attempted!r} (already -> {blocking!r})")
    print("\nNEW pairs:")
    for canonical in sorted(new_pairs):
        print(f"  {canonical}: {new_pairs[canonical]}")

    if args.write and new_pairs:
        data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        data["teams"] = merge_into_seed(existing_seed, new_pairs)
        _SEED_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {_SEED_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
