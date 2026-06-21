# Best solution for ALL basketball games (2026-06-21)

4-agent web + GitHub sweep (coverage / pricing / model-as-screen), judged against
the sharp-vs-soft CLV doctrine. See also multisport-modeling-2026-06-21.md.

## Verdict: it's COVERAGE, not models

All three angles independently concluded **adopt-nothing-new** for the
devig/anchor/EV math — that core is already at/near optimal (POWER/SHIN devig
correct for 2-way basketball). The only axis that adds +EV picks is **reach of the
sharp-vs-soft line-shop**, via a two-tier anchor:

- **Tier A — true sharp, alertable:** `devig(Pinnacle)` wherever Pinnacle prices
  the league — FAR broader than NBA: NBA, WNBA, G-League, EuroLeague, EuroCup,
  Spain ACB, Italy Lega A, France Pro A, German BBL, Greece A1, VTB, Israeli, Korea
  KBL, PBA, China CBA, Australia NBL, NCAAB. Pinnacle is the field-tightest book in
  every basketball market it prices (NBA ~2-2.5% vig, NCAAB 2.89% grade-A #1,
  EuroLeague 3.8% #1). So "devig Pinnacle = fair" is correct PER league.
- **Tier B — consensus-proxy, shadow only:** leagues Pinnacle structurally won't
  price (women's/youth/minor-Euro) -> no-vig multi-book CONSENSUS as a _proxy_ fair,
  with a WIDER edge floor + a `consensus_proxy` provenance flag — never graded as a
  true sharp close.

## Concrete improvements (prioritized)

1. **[DONE 2026-06-21] Grow the Pinnacle alias table** — the #1 lever. The 28%
   basketball match rate was an ALIAS problem on our side, not a Pinnacle coverage
   hole. Data-driven fix in `aliases_seed.json`: 13 WNBA "W"-suffix teams (Pinnacle
   `Los Angeles Sparks` vs OddsPortal `Los Angeles Sparks W`; WNBA names are
   gender-unique so no men's/women's conflation) + sponsor/city tails (Legia
   Warszawa, Barangay Ginebra San Miguel, Taipei Fubon Braves, Pau-Orthez).
   **Live match rate 28% -> 36%** (68->87 of 236) — recovered the in-season WNBA
   slate (now Tier-A alertable). The long tail (minor Israeli/Dominican/Argentine
   women's leagues) is ongoing grind, mostly shadow-tier.
2. **[medium] `consensus_proxy` close-anchor provenance** (extend ADR-0017 enum) —
   surfaces +EV in Pinnacle-absent leagues without faking sharp CLV.
3. **[small] TheSportsDB** free scores backfill where ESPN thins on non-NBA basketball.
4. **[small] Early-season NBA totals-UNDER bias** as an informational seasonal tag
   (Baryla 2007 56.7%; JPM wk1 58%) — must still clear forward CLV; never staked on
   the historical pattern alone.
5. **[small] WNBA = "soft-sharp"** — Pinnacle present but not always field-tightest.

## Honest truths

- A points/ratings/Elo/**ML model will NOT find +EV the Pinnacle close misses** —
  Hubacek/Sourek/Zelezny (2019, IJF) tested ML against this exact anchor and lost;
  Forrest-Goddard, Levitt agree. Settled, not folklore.
- **NBA is the only basketball league with a validated forward CLV history**; every
  other league lacks a free sharp close — a permanent free-source-bound limit.
- **Betfair/exchanges are a dead end** as a basketball anchor (NBA liquidity too
  thin, non-NBA effectively dead). Keep Betfair basketball as a secondary soft price.
- Consensus-proxy picks are genuinely lower-grade and must stay labeled + demoted.
- Do NOT add: a new sharp source, paid Pinnacle reseller, model-as-anchor,
  derivative markets (1H/1Q/team totals), SX Bet/Kalshi.
