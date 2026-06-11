# NBA repo evaluations — phase-5 inputs (2026-06-11)

Two user-suggested repositories inspected file-by-file (repo-researcher
agents, GitHub MCP). Full scoring tables in the session transcript; verdicts
and the parts that survive scrutiny below. Baseline for judgment: ADR-0005 /
ADR-0009 (LightGBM + isotonic, walk-forward CLV evaluation, leakage-safe
point-in-time features, nba_api ingestion).

## kyleskom/NBA-Machine-Learning-Sports-Betting — REFERENCE-ONLY (reject code)

1,663★, pushed 2026-01. **No LICENSE (`license: null`) — code may not be
adapted, period.** Methodology fails our bar anyway:

- **Leakage twice over**: per-day team stats fetched with `DateTo=` the game
  date itself (stats.nba.com is inclusive → the game's own box score is in
  its features); the totals model keeps the (closing) `OU` line as a feature.
- **Accuracy-only evaluation**: no CLV, no ROI backtest, no reliability
  diagrams; models literally named by test accuracy.
- **No devig** in EV math; **full Kelly** with no fraction/caps.
- Inference bug: UO feature columns silently misaligned vs training order.
- Safety: clean — GET-only ingestion, display-only Flask; no placement code.

Ideas (not code) worth keeping: `sbrscrape`/SBR as a free per-day NBA odds
source for recent seasons (needs its own ToS review); pushes as an explicit
third class in totals models.

## NBA-Betting/NBA_Betting — MINE-FOR-PARTS (MIT)

206★, ARCHIVED (author moved to NBA-Betting/NBA_AI, uninspected — evaluate
separately before phase 5). Modeling/eval layers are what ADR-0005 was
written to avoid (AutoGluon, accuracy metric, zero calibration, single
season split, flat −110 pseudo-Kelly, no closing lines → CLV-blind, naive
Mountain-time datetimes). Safety: clean (no placement code).

**Re-implement these patterns in our stack (MIT permits; credit here):**

1. **Point-in-time nba_api snapshots** — `src/data_sources/team/
nbastats_fetcher.py`: `LeagueDashTeamStats(date_to_nullable=D)` per
   calendar date in dual windows (season-to-date + last-2-weeks), 0.6s
   rate-limit with exponential backoff + jitter, DB-resume on restart.
   The answer to "leakage-free historical team stats" for phase 5.
2. **Point-in-time join rule** — `src/etl/main_etl.py`: `merge_date =
stats.to_date + 1 day` then backward `merge_asof` — stats through D−1
   only ever attach to games on D. Adopt as the canonical stats→game join.
3. **Feature catalogue cross-check** — `src/etl/feature_creation.py`:
   shift(1)-disciplined rolling/expanding features, rest-diff; also a
   negative example (`win_pct` mislabeled net-result rate) for our tests.
4. **Model-cutoff registry rule** — `src/predictions/generator.py`:
   never let a model score a game dated on/before its training-end.

**Do not port:** AutoGluon trainer, accuracy evaluation, covers.com spider,
flat −110 ROI/pseudo-Kelly, naive local datetimes.
