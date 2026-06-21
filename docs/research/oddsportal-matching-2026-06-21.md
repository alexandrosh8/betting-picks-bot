# OddsPortal data model + cross-source matching (2026-06-21)

Goal: read how OddsPortal encodes team identity to lift the Pinnacle->soft match
rate (was 28% basketball, 36% after alias growth). Read-only GETs of public
aggregator pages + OddsHarvester source/fixtures + 2-agent web/GitHub research.
NO anti-bot bypass, NO bet surface.

## What OddsPortal exposes

1. **Match-URL slug + stable 8-char participant ID.** Every match link is
   `/<sport>/h2h/<home-slug>-<ID>/<away-slug>-<ID>/#<eventID>`. The 8-char token
   after each team slug is a **stable per-team participant ID** (the SAME id
   recurs for the same team across fixtures — e.g. `minnesota-lynx-AeUcuq4r`,
   `las-vegas-aces-nZjYLTCd`). The `#`-fragment is the event id.
2. **The slug is cleaner than the display name.** OddsPortal's URL slug omits the
   women-league `W` suffix the scraped display name carries
   (`Los Angeles Sparks W` display vs `los-angeles-sparks` slug) and is
   consistently lowercased — a better match key.
3. **Embedded JSON** (`div#react-event-header`, `data=...`) carries `eventData`:
   `{id, ukeyBase:'E-...', tournamentId, home, away, sportId}` — three stable ids.
4. **OddsPortal + BetExplorer share the SAME operator (LiveSport s.r.o.) AND the
   EXACT SAME participant + event ids.** BetExplorer team pages
   `/basketball/team/minnesota-lynx-w/AeUcuq4r/` carry IDENTICAL ids — and
   BetExplorer's slug keeps the `-w` suffix while OddsPortal's omits it, proving
   the **id is more canonical than any name string**.

## DONE — slug fallback (this commit)

`app/resolution/matching.py:oddsportal_slug_names()` parses `(home, away)` from
the OddsPortal `external_ref` (strips the 8-char id). `resolve_pinnacle_close_snaps`

- `shadow_match_rate_outcomes` now **retry with the slug when the display name
  fails** — still STRICT + unique-candidate (cannot attach a wrong close).
  Empirical: display 87/236, **display-OR-slug 96/236 -> 36% -> 41%**, GENERICALLY
  (every women's league + sponsor case, not just the 13 hand-aliased WNBA teams).

## The bigger lever (future, not built)

**Key the alias table by OddsPortal participant ID -> Pinnacle canonical name.**
Once `minnesota-lynx-AeUcuq4r -> Pinnacle "Minnesota Lynx"` is confirmed, it is
permanent and immune to display drift (W-suffix, sponsor, transliteration) — a
per-stable-id table instead of per-display-string. **Bootstrap it from
BetExplorer** (same ids, its `/team/<slug>-<id>/` pages enumerate every team's
canonical name + id) — a free, read-only, offline seed.

**Caveat (must go in the ADR):** the participant id is OddsPortal/BetExplorer-
internal and is **NOT shared with Pinnacle**, so it cannot directly key the
cross-source join — Pinnacle is still matched by name. The id's value is making
the OddsPortal SIDE's identity canonical so the alias is learned ONCE, never a
runtime fuzzy guess (the cardinal "never guess a close" rule is preserved).

## Effort / next step

Persist the parsed participant ids on the OddsPortal event at ingest
(`app/ingestion/oddsportal.py`, pure-string parse, no extra network), then build
the id->canonical alias map seeded from a one-off BetExplorer harvest. Medium
effort; the slug fallback already banks most of the immediate gain.
