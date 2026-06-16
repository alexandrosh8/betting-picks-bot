# ADR-0014: Strict Cross-Source CLV Resolution — attach the Pinnacle archive close to OddsPortal picks

- **Status:** accepted
- **Date:** 2026-06-16
- **Deciders:** GodFather (Alexis) — "Build a new PURE `app/resolution/` module
  that attaches captured Pinnacle arcadia closes … to the matching OddsPortal
  events/picks … STRICT, deterministic matching ONLY. Fuzzy/loose string-distance
  joins are FORBIDDEN."

## Context

ADR-0013 shipped the Pinnacle arcadia capture: sharp closes accumulate under an
ISOLATED `pinnacle_<sport>` warehouse namespace, deliberately NOT joined to the
live pick path. The deferred step — the prerequisite for measuring incremental
CLV vs a real sharp close (and eventually validating NBA/tennis) — is to attach
a Pinnacle archive close to the matching OddsPortal pick. The events come from
different sources with different ids and different team-name spellings, and the
project's cardinal sin is a **wrong** close: it corrupts CLV. So the join must
be STRICT — it must prefer leaving a fixture unmatched over guessing.

The 2026-06-16 repo sweep (`docs/research/repo-sweep-2026-06-16.md`) surfaced
the clean-room building blocks: USSoccerFederation/glass_onion (BSD-3,
deterministic event join), probberechts/soccerdata (Apache-2.0, alias-table
pattern), withqwerty/reep (CC0, alias data).

## Decision

A new **pure** `app/resolution/` module (numpy/stdlib only, no IO):

- `normalize_name` — NFKD accent-strip → casefold → alphanumerics → drop
  club-form noise (`FC`/`AFC`/…). **Women's/youth markers are preserved on
  purpose** (stripping them, as glass_onion does, would conflate "Arsenal
  Women" with the men's "Arsenal" — a wrong-close defect).
- `AliasTable` — `{normalized_alias → canonical}` with bidirectional helpers
  (the soccerdata pattern), seeded from a small hand-curated
  `aliases_seed.json` starter (expand from reep's CC0 data over time).
- `match_event` — returns the UNIQUE candidate with exact canonical home/away
  equality AND kickoff within a small day window, else **None**. No fuzzy, no
  containment, no best-available fallback; **ambiguous (>1 candidate) → None**;
  `ordered=True` (soccer/NBA) rejects a home/away swap; `ordered=False` (tennis)
  matches the unordered pair.

IO lives outside the pure module:

- `repositories.resolve_pinnacle_close_snaps` — strict-matches the pick's
  fixture to its `pinnacle_<sport>` archive event and returns that event's CLOSE
  re-keyed to the pick's `event_id` + selection vocabulary (bookmaker stays
  "Pinnacle"); selections that cannot be mapped to the pick's home/away/Draw are
  **dropped, never mis-attached**. `[]` on no/ambiguous match.
- `clv_trueup.finalize_closing_from_snapshots` injects that close into the
  settlement-time close set behind `Settings.clv_use_pinnacle_archive`
  (**default FALSE**). When present, `bookmaker="Pinnacle"` auto-anchors the
  fair (`value.SHARP_BOOKS[0]`) — incremental CLV vs a real sharp close, with
  zero devig-math changes.

## Clean-room provenance

The deterministic exact-key + bounded-date-tolerance approach is adapted from
glass_onion's `match.py`; the alias-table pattern from soccerdata. NO code was
copied (a reviewer fetched both and confirmed zero shared expression); the
forbidden TF-IDF/cosine fuzzy passes are omitted. `aliases_seed.json` is a
~24-entry hand-curated starter, not a bulk copy of reep's dataset; `reep-cli`
is never installed.

## Default OFF — why

Enabling the injection changes `anchor_type`/CLV for matched picks. Per the
project's anchor-change doctrine, that needs evidence (match rate + CLV
stability), so `CLV_USE_PINNACLE_ARCHIVE=false` is the default and the wiring is
byte-identical to before when off. A **low** match rate is safe (fewer picks get
a Pinnacle-anchored close); a **wrong** match would corrupt CLV — prevented by
the strict matcher.

## Verification

18 pure matcher tests (`tests/test_resolution.py`) + 4 DB integration tests
(`tests/test_resolution_db.py`, compose Postgres) covering alias match, re-key,
and the cardinal-sin guards (ambiguous → [], no-match → [], out-of-window → [],
women ≠ men, swap → None). ruff/mypy/safety-audit green; full suite passes.

## Consequences

- **v1 = soccer (ordered), moneyline.** Tennis (`ordered=False` + name-order
  reconciliation) and NBA come later; the seed alias table grows from reep.
- Before flipping `CLV_USE_PINNACLE_ARCHIVE=true`, validate the soccer match
  rate against live data (how many picks find a strict Pinnacle match).
- The matcher is reusable for the football-data.co.uk results join and any other
  cross-source attach; it supersedes nothing — `app/settlement/results.py`'s
  `normalize_team`/`ScoreBook` stay as-is for now.
