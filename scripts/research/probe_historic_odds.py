"""Probe: does OddsHarvester HISTORIC mode give us, for FREE, a real CLOSING
line (and ideally Pinnacle open+close) for already-played matches?

Read-only validation step gating any historic backfill (ADR-research
2026-06-19). Scrapes ONE finished football league / season, bounded to a
single results page, and reports:

  * does the historic page return per-bookmaker odds at all?
  * is **Pinnacle** among the bookmakers? (the data gate's sharp anchor)
  * with --history, does each book carry an opening-vs-current odds history?

Uses OddsHarvester exactly as the live loader does (its own pacing, no
anti-bot bypass) — same site we already scrape for upcoming matches. Nothing
is written to the DB. Run:

    uv run python scripts/research/probe_historic_odds.py
    uv run python scripts/research/probe_historic_odds.py --history   # + opening odds (slower)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

# Mirror the live loader's mandatory upstream setup + coherent (non-bypass)
# browser fingerprint. Importing the module triggers the quirk patches.
from app.ingestion.oddsportal import _patch_upstream_quirks, register_extra_leagues

LEAGUE = "england-premier-league"
SEASON = "2023-2024"
MARKETS = ["1x2"]


def _summarize(matches: list[dict[str, Any]]) -> None:
    books: set[str] = set()
    history_books: set[str] = set()
    matches_with_odds = 0
    sample_entry: dict[str, Any] | None = None
    for m in matches:
        entries = m.get("1x2_market") or []
        if entries:
            matches_with_odds += 1
        for e in entries:
            if not isinstance(e, dict):
                continue
            name = str(e.get("bookmaker_name") or "")
            if name:
                books.add(name)
            if e.get("odds_history_data"):
                history_books.add(name)
            if sample_entry is None:
                sample_entry = e

    pinn = sorted(b for b in books if "pinnacle" in b.lower())
    print("\n================ HISTORIC PROBE RESULT ================")
    print(f"league/season         : {LEAGUE} {SEASON}")
    print(f"matches returned      : {len(matches)}")
    print(f"matches with 1x2 odds : {matches_with_odds}")
    print(f"distinct bookmakers   : {len(books)}")
    print(f"PINNACLE present      : {'YES -> ' + ', '.join(pinn) if pinn else 'NO'}")
    print(
        f"books w/ odds_history : {len(history_books)}"
        + (f" (incl. {sorted(history_books)[:3]})" if history_books else "")
    )
    if books:
        print(f"bookmaker sample      : {sorted(books)[:12]}")
    if sample_entry is not None:
        print("\nsample 1x2 entry keys :", sorted(sample_entry.keys()))
        print("sample 1x2 entry      :")
        print(json.dumps(sample_entry, indent=2, default=str)[:1400])
    print("=======================================================\n")
    # Decisive verdict for the data gate.
    if pinn and matches_with_odds:
        print(
            "VERDICT: historic results carry per-book CLOSING odds INCLUDING "
            "Pinnacle — the sharp anchor + close are free here. Backfill is "
            "worth building (next: confirm OPENING via --history)."
        )
    elif matches_with_odds:
        print(
            "VERDICT: historic results carry per-book closing odds but NO "
            "Pinnacle — a real closing line, but not a sharp anchor. Useful "
            "for grading existing picks' CLV; does NOT close the Pinnacle gap."
        )
    else:
        print(
            "VERDICT: no per-book odds parsed from the historic page — either "
            "the DOM shifted or the scrape was blocked. Re-run / inspect before "
            "any conclusion (absence of data is not a negative result)."
        )


async def _main(with_history: bool) -> None:
    register_extra_leagues()
    _patch_upstream_quirks()
    from oddsharvester.core.scraper_app import run_scraper
    from oddsharvester.utils.command_enum import CommandEnum

    print(
        f"scraping HISTORIC {LEAGUE} {SEASON} (1 page, markets={MARKETS}, "
        f"odds_history={with_history}) — read-only, OddsHarvester pacing…"
    )
    result = await run_scraper(
        command=CommandEnum.HISTORIC,
        sport="football",
        leagues=[LEAGUE],
        season=SEASON,
        markets=MARKETS,
        max_pages=1,
        scrape_odds_history=with_history,
        headless=True,
        browser_timezone_id="UTC",
        browser_locale_timezone="en-GB",
        concurrency_tasks=2,
        request_delay=1.5,
    )
    matches = list(getattr(result, "success", None) or [])
    failed = list(getattr(result, "failed", None) or [])
    if failed:
        print(f"(note: {len(failed)} match link(s) failed to parse — expected on a fragile scrape)")
    _summarize(matches)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--history",
        action="store_true",
        help="also scrape per-bookmaker odds history (opening odds) — slower",
    )
    args = ap.parse_args()
    asyncio.run(_main(args.history))
