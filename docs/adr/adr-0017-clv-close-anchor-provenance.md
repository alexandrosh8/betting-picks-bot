# ADR-0017 — CLV close-anchor provenance (trust only genuine sharp closes)

- **Status:** Accepted (2026-06-20)
- **Relates to:** ADR-0013 (Pinnacle ARCADIA archive), ADR-0014 (cross-source
  CLV resolution). Builds on `Pick.anchor_type` (the CREATION anchor).

## Context

CLV (Closing Line Value) is the platform's trust currency: a sport only earns
alerting picks after its held-out incremental CLV clears > 2 SE, and the live
`/performance` + `live_evidence` reports are how a strategy version is judged.
An adversarial review found that the reported CLV was **partly contaminated**:

1. A pick's `anchor_type` is the **creation** anchor, fixed once. The **close**
   is computed independently later (`revalidate_open_picks`,
   `finalize_closing_from_snapshots`) and can be anchored by a **different**
   book — a soft-book **consensus median** when no named sharp book prices the
   market at close. That consensus close was reported under the creation
   anchor's stratum, contaminating per-anchor CLV.
2. When no snapshot close exists, the close falls back to the last poll-time
   re-scrape fair (`closing_odds` stays NULL) — a **revalidation fallback**, not
   a true market close. Both consumers counted these in the headline
   stake-weighted CLV / beat-close rate with **no provenance filter**.

The alerting decision itself is made at pick time on the real anchor, so this is
a reported-metric integrity defect, not a pick-admission defect — but the
version-trust gate feeds future strategy decisions, so honest CLV matters.

## Decision

Track the **close** anchor and surface a **trusted** CLV subset, without
discarding any data (additive + feature-detected, mirroring `anchor_type`):

- New nullable column `Pick.closing_anchor_type` (pinnacle / sharp / consensus),
  set by **both** CLV write paths from the anchor `event_fair_probs` returns
  (previously discarded). Migration `d3b8f1c7a9e2`.
- A close is **trusted ("sharp close")** iff it is **snapshot-sourced**
  (`closing_odds` NOT NULL — not a poll-time fallback) **AND** anchored by a
  **named sharp book** (`closing_anchor_type` in {pinnacle, sharp} — not a
  consensus median). `SettledPickRow.sharp_close` encodes this.
- `live_evidence_report` gains a `sharp_close` stratum (the trusted CLV the
  platform can stand behind — always reported; n=0 honestly says "none yet")
  and a `by_close_anchor` grouping (CLV stratified by the anchor that produced
  each CLOSE, the anchor CLV is actually measured against). `by_anchor` keeps
  its existing CREATION-anchor contract (the consensus-fallback forward test).
- `performance_report` / `_aggregate_settled` gain `n_sharp_close`,
  `sharp_stake_weighted_clv_log`, `sharp_beat_close_rate` over the same trusted
  subset, alongside the existing blended headline.

## Why surface, not replace

The blended headline stays (continuity + the deliberate consensus-creation
forward test, decisions.md). The trusted figures are added so the operator can
**see** how much of the reported CLV is genuine sharp-close evidence vs
consensus/fallback — honest disclosure rather than a silent number. Nothing is
dropped; obscure/fallback closes remain visible, just no longer masquerading as
trusted CLV.

## Consequences

- Per-anchor and headline CLV can now be read honestly: the `sharp_close`
  subset is the number to trust; the blended headline is context.
- Additive + nullable + feature-detected — pre-column rows and not-yet-migrated
  DBs degrade to "no close anchor" (empty sharp-close subset), never an error.
- No change to pick admission, the strict matcher, or the edge gate.

## Alternatives considered

- **Restrict the close to named sharp books only (NULL otherwise):** simpler
  but throws away the consensus-fallback forward-test evidence the project
  deliberately accrues, and shrinks the CLV sample. Rejected in favour of
  label-and-surface.
- **Bucket `by_anchor` on the close anchor (replace creation):** would lose the
  intended consensus-CREATION analysis; added `by_close_anchor` instead so both
  views coexist.
