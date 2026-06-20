# ADR-0016 — Major-league premium gate (scope alerts to leagues with sharp coverage)

- **Status:** Accepted (2026-06-20)
- **Supersedes / relates to:** ADR-0013 (Pinnacle ARCADIA archive), ADR-0014
  (cross-source CLV resolution). Builds on the `picks.tier`
  premium/volume split.

## Context

The held-out CLV edge (sharp-vs-soft line shopping) only exists where a real
sharp anchor prices the market. On the free stack, sharp coverage is
**structurally ~37%** of picks: obscure slates (Australian NPL state leagues,
lower divisions, women's, U20, reserve) have **no free sharp close anywhere**,
and the strict cross-source matcher correctly refuses to alias them (a wrong
close mints fake CLV — worse than no close; see `.claude/memory/pitfalls.md`,
2026-06-20). Aggressive aliasing is therefore off the table.

Yet the value pipeline assigned `tier` purely on the edge threshold, with **no
league awareness**: a premium-edge pick in an obscure league was alerted and
reserved exposure with the same authority as one in a Pinnacle-covered major —
even though its "edge" can never be CLV-verified and its market may be illiquid.
This dilutes the alerting tier with picks that cannot honestly be high-ROI.

The lever for honest high-ROI alerts is therefore **scoping the premium tier to
major leagues**, not loosening the matcher.

## Decision

Add an **optional, default-OFF, config-driven major-league gate** to
`run_value_pipeline`, mirroring the existing `VALUE_ML_FILTER` demotion:

- `VALUE_MAJOR_LEAGUES` (csv of scraped `league_name` strings) enters the
  pure-math `ValuePolicy` as a frozen `major_leagues` tuple at the composition
  root. **Empty = gate disabled** (every league premium-eligible — current
  behavior; non-breaking on merge).
- When set, a `premium` candidate whose scraped per-event league name does not
  normalize into the set is **demoted to the `volume` (shadow) tier**: still
  persisted and CLV-tracked, but **never alerted and never reserving exposure**.
- Matching is **exact-on-normalized** (NFKD→ASCII, casefold, punctuation→space,
  whitespace collapse) — no fuzzy substring, so the gate can never _falsely_
  promote a minor league (same doctrine as the strict matcher).
- A blank/unknown scraped league name with the gate enabled is **not major**
  (cannot be confirmed sharp-covered → demoted).
- The operator curates the set from the existing `/resolution/match-rate`
  per-league (`by_league`) report, which is keyed on the same `league_name`
  the gate uses — so the allowlist reflects _measured_ sharp coverage, not
  guesses. `.env.example` ships a documented seed of unambiguous majors.

## Why demote, not drop

Demotion preserves the full evidence trail: obscure-league edges keep flowing as
silent volume picks that are still CLV-graded, so the gate's correctness is
auditable and nothing honest is lost. Dropping would destroy that evidence.
This is the honesty-maximizing choice and reversible (clear the env var).

## Consequences

- The premium (alerting + exposure) tier concentrates on leagues with real sharp
  coverage + liquidity, raising the _premium-tier_ matched-close rate toward the
  honest ceiling; obscure-league edges remain tracked, just silent.
- Off-season / thin slates will naturally alert fewer picks when the gate is on
  — correct, not a regression.
- No effect until the operator opts in; committed default keeps the gate off.
- Does **not** touch the strict matcher, CLV math, or pick-admission edge gate.

## Alternatives considered

- **Anchor-availability gate at pick time** (query the Pinnacle archive per
  candidate): more principled but adds a per-pick DB query (ADR-0014
  deliberately made anchor resolution settlement-time) and a timing risk (the
  archive may not have listed the event yet at pick time, suppressing valid
  picks). The league allowlist is a cheap, stable proxy curated from the _same_
  measured coverage, without those costs.
- **Higher edge floor outside majors:** softer; does not tie alerting to sharp
  availability, so it moves the matched-close rate far less.
- **Looser cross-source aliasing to raise coverage:** rejected — mints wrong
  sharp closes (fake CLV). The structural ~37% is the doctrine-correct ceiling.
