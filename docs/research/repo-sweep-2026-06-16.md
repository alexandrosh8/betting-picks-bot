# Repo Sweep — "best repos to use on the project" (2026-06-16)

- **Method:** 4 parallel `repo-researcher` agents (file-by-file inspection, no
  clones), applying the project gate (strategy shape → data gate → no-autobet →
  license → "don't duplicate what we have"). The ~25 already-settled repos
  (`.claude/memory/decisions.md`) were excluded, not re-evaluated.
- **Bottom line:** no new runtime dependency is warranted — the project already
  owns the heavy machinery (OddsHarvester, penaltyblog devig, our backtester +
  CLV/Kelly, the new clean-room Pinnacle arcadia capture). But three findings
  are genuinely actionable as **clean-room patterns / data to lift**, and they
  de-risk the deferred cross-source CLV join.

## Headline: the cross-source matcher stack (unblocks the deferred CLV join)

Attaching a Pinnacle close (from the new `pinnacle_<sport>` archive) to the
matching OddsPortal pick is the deferred step in ADR-0013, and fuzzy joins are
forbidden (a wrong close corrupts CLV). These three compose into the exact
strict matcher we need — **port the algorithms/data, do not add the packages**
(keep the numpy/stdlib pure-math boundary):

| Repo                               | License    | ★ / pushed        | What to take                                                                                                                                     | Caveat                                                                                                                                                 |
| ---------------------------------- | ---------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **USSoccerFederation/glass_onion** | BSD-3      | 22 · 2026-06-03   | The **event JOIN**: exact `(date, home_id, away_id)` merge + bounded **±3-day** tolerance + matchday fallback (`match.py: synchronize_pair`)     | **Skip** its `team.py`/`player.py` TF-IDF-cosine passes (pass-3 runs at threshold 0.0 = best-available — the forbidden fuzzy join)                     |
| **probberechts/soccerdata**        | Apache-2.0 | 1759 · 2026-06-14 | The **alias-table PATTERN**: a `{alias→canonical}` dict + bidirectional `add_alt_team_names` / `add_standardized_team_name`                      | Shipped dict is empty; data sources fail the data gate (never an odds dep)                                                                             |
| **withqwerty/reep**                | **CC0**    | 170 · 2026-05-04  | The **alias DATA** to seed it: Wikidata register, 488K people / 45K teams, incl. "A. Surname" forms; `custom_aliases.json` provider-keyed schema | No Pinnacle/OddsPortal key column → bridges canonical↔mainstream; we normalize our two sources _into_ it. CSV/JSON only — never pip-install `reep-cli` |

Recommended build: a new pure `app/resolution/` module — glass_onion's
deterministic join + soccerdata's alias pattern + a reep-seeded alias table,
extending the existing `app/settlement/results.py::normalize_team` (NFKD +
containment) with suffix/prefix stripping. Strict-only; tennis stays
rank/seed-aware, never loose string distance.

## Robustness reference for the new arcadia capture

| Repo                | License | Verdict       | Take                                                                                                                                                                                                                                                                                                                                         |
| ------------------- | ------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pinnapi/pinnapi** | MIT     | adopt-pattern | Typed error hierarchy (`AuthError` 401/403, `RateLimitError` 429 reading `retry_after_ms`) vs our single `PinnacleArcadiaError` + blind tenacity backoff. Adapt the **patterns**, not the URLs (they're pinnapi's, not arcadia's). Also confirms Pinnacle's public dev API closed 2025-07-23 → the guest arcadia path is the only live route |

## Backtester realism upgrade

| Repo                                       | License          | Verdict                    | Take                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------ | ---------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **betcode-org/flumine**                    | MIT              | adopt-pattern (clean-room) | Queue-position-aware fill model (`_piq` decremented before fill + traded-volume halving + `size_remaining`/book-max caps + BSP/LAPSE) — materially more realistic than our full-size-at-quoted-price assumption. Re-implement the PATTERN in `app/backtesting/`; **do NOT** add flumine/betfairlightweight (`betfairexecution.py` places live orders) |
| evan-kolberg/prediction-market-backtesting | NOASSERTION/LGPL | reference-only             | Idea only: probability-scale entry/exit taker-slippage. Autobet + copyleft + heavy NautilusTrader — write clean-room if wanted                                                                                                                                                                                                                        |

## Devig / CLV — no gap

penaltyblog (already a dependency) consolidates the **entire** devig space:
multiplicative, power, additive, **Shin**, differential-margin-weighting
(worst-case/longshot), odds-ratio, logarithmic — devigging to 1e-8. So:

- **mberk/shin** (MIT, 103★) — correct standalone Shin solver with z/insider
  diagnostics; keep as a **clean-room cross-check only**, not a dependency.
- **REJECTED:** `neeljshah/clvtrack` (vaporware — README advertises a full CLV
  bootstrap tool; the code is a 22-byte `__version__` stub), `p2w-math`
  (ad-hoc/wrong Shin + unliftable NOASSERTION), `ian-shepherd/pybettor` (CLV on
  _vigged_ probs — wrong for our no-vig doctrine), `deltaray-io/kelly-criterion`
  (equities mean-variance Kelly — wrong shape).
- The one real gap (a packaged CLV beat-rate calc with CIs) — no inspected repo
  implements it, and we already compute clustered/block-bootstrap CLV CIs.

## Data gate (free sharp+close for tennis/NBA) — STILL NONE

Re-confirmed by code inspection. Everything carrying **both** a Pinnacle anchor
**and** a closing line is **paid** (bettingiscool, PinBook, prop-line, SharpAPI,
Odds Warehouse). `sportsdataverse/hoopR` exposes open/close columns but
**ESPN-soft providers only — no Pinnacle** (fails the sharp-anchor condition,
same class as nflverse consensus). ParlayAPI's free "pinnacle" rows are its own
forward self-capture — the same kind as our arcadia capture. **The only
doctrine-valid path for NBA/tennis remains prospective self-captured Pinnacle
snapshots — already shipped (ADR-0013).**

## Rejected on no-autobet (hard rule)

`rozzac90/pinnacle` (POSTs real orders to `/v2/bets/straight`),
`chrisgillam/polymarket_gambot` (signs + submits live Polymarket orders + paid
Pinnacle proxy + no close), the prediction-market/flumine **execution** modules.
Their betting/order code is unusable; only backtest/devig math is salvageable,
clean-room.

## Surfaced but NOT inspected (never recommended)

JeffSackmann/tennis_atp (404/removed — the canonical ATP/WTA player crosswalk; a
real tennis-side gap if it reappears), JohnComonitski/SoccerAPI (wraps the
SUSPENDED API-Football), Tennismylife/TML-Database (CC-BY-NC-SA, restricted),
nautechsystems/nautilus_trader (LGPL + autobet adapters), `canonmap`/fuzzywuzzy
matchers (loose-fuzzy — forbidden).

## Recommended next action

Build the strict cross-source matcher (`app/resolution/`) from the glass_onion +
soccerdata + reep stack — it converts the Pinnacle archive from "accumulating"
into "attaches a real sharp close to OddsPortal picks," the prerequisite for
NBA/tennis CLV. Optional later: clean-room flumine's PIQ fill model into the
backtester; add pinnapi's typed-error pattern to `pinnacle_arcadia.py`.
