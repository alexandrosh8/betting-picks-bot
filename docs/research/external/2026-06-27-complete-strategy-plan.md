# Complete Strategy Plan — sharp-ev-picks (2026-06-27)

Merges: the 14-finding correctness audit (PRs #100–110), the honest-premium diagnosis (`n_sharp=0` structural ceiling), the Jarvis +EV blueprint, and the 1,092-source research corpus. X-account data is **not included** (account has 0 API credits — 402; scraping alternatives need account login + violate ToS, so excluded).

---

## 0. Honest current state (one paragraph)
The strategy *math* is correct and the safety posture is intact. The problem is **coverage + anchor quality**, not the method. Live: `n_sharp = 0/50` trusted independent-sharp closes, so PROOF-OF-EDGE is (correctly) blank. Cause is **structural**: Pinnacle prices only ~21% of your priced soccer, `betfair_soccer` is 0 upcoming (off-season + thin), and the European majors are out of season. No labeling or code fix manufactures a sharp close the market never made.

## 1. Verdict — optimize, don't pivot
**Keep sharp-vs-soft line shopping. It is the solid strategy; the research is unanimous.** The "more solid alternative" you asked about does not exist as a *different method* — the candidates all fail:
- **Pure ML predictor** → does not beat the closing line OOS (your own ADR + the systematic-review literature; kyleskom-style accuracy is leakage-inflated).
- **Arbitrage / middling** → real but capital- and limit-constrained, and shrinks as books tighten.
- **Market-making** → forbidden (it *places orders*; violates HARD SAFETY RULE 1).

The "more solid" version of your strategy is **sharp-vs-soft executed on a better anchor, with strict coverage discipline.** That is the plan below.

## 2. The binding constraint
Honest premium volume is capped by **how many of your priced games have a genuine, independent sharp close.** Everything in §3 either (a) makes each anchored pick *more correct*, or (b) *expands genuine sharp coverage*. Only (b) raises the count; (a) protects integrity.

---

## 3. Optimization roadmap (phased; alpha vs marginal labelled)

### Phase A — Anchor quality (REAL EDGE; build first)
1. **Log-odds (logit) ensemble true line.** Replace single-anchor→consensus with `consensus_p = sigmoid(Σ wᵢ·logit(pᵢ))` across genuinely independent sharps (Pinnacle + Betfair back/lay **midpoint** + Smarkets). Linear averaging is provably under-confident on favorites/longshots (Gneiting-Ranjan) — the single biggest correctness win. *File: `app/edge/value.py` anchor path.*
2. **Inverse-variance weighting with shrinkage to equal** (`wᵢ = λ/K + (1−λ)·volᵢ/Σvol`, λ≈0.5). With only 4–6 sharps, learned weights are noise — shrinkage degrades gracefully to equal-weight log-odds when liquidity proxies are missing.
3. **Curate independent sharps, do NOT add more soft books.** Soft books copy/shade Pinnacle; equal-weighting many just re-weights Pinnacle with noise. Soft books are the *value target only*.
4. **Market-dependent devig.** Route **symmetric markets (basketball totals / Asian handicap) through Probit**; keep Shin/Power for FLB-prone 1X2/longshots. Multiplicative is fine for 1–3% sharp books (validated on 380 EPL closes). Harden `app/probabilities/devig.py:_shin` with the exact **n=2 closed form** (≡ Additive) + discriminant clamp. *Reuse `penaltyblog` (already a dep) as oracle.*
5. **Liquidity / empty-book guard on exchange anchors** (extends CLV-2). A NULL/thin-liquidity or just-firmed exchange line must not earn "sharp" grade — the Welwalo lesson.

### Phase B — Staking (REAL; informational only)
6. **Edge-uncertainty shrink = the Kelly fraction.** Full-Kelly on a shrunk edge ≡ fractional-Kelly on raw (Rising-Wyner). Make the flat ¼ a per-pick curve decreasing in `Var(p̂)` (Baker-McHale).
7. **Covariance-aware fractional Kelly across same-slate correlated picks.** Independent Kelly overbets ~2.3× once leg-correlation ρ>0.4. Use Busseti-Boyd risk-constrained Kelly with a Ledoit-Wolf-shrunk residual covariance (runnable ref: `cvxgrp/kelly_code`). Your per-event sub-cap is the crude proxy; this is the principled version.

### Phase C — Measurement (correctness; do alongside)
8. **CLV line-drift time-series.** Log bet-time → intermediate → close (vig-free) per pick; you currently keep mostly a single close snapshot.
9. **Good/bad variance 2×2 attribution.** sign(CLV) × sign(PnL), cut by market/time-bucket/Kelly-bucket (design ref: `neeljshah/clvtrack` report shape). Always de-vig the close first.
10. **Steam detection — MARGINAL here, cheap insurance.** Flag M-of-N independent sharps moving the same direction ≥X within T, **staleness-gated**. Cheapest real win: a **per-outcome `changedAt` timestamp** so one snapshot detects a stale soft book. *Honest: with one scraped request-response feed vs a 300s freshness window, sub-cycle latency edge is folklore until measured on your own CLV log.*

### Phase D — Coverage (the only lever that RAISES honest-premium count)
11. **Restore Betfair as the second independent sharp** using the **delayed read-only app key** (live key read-only is ToS-barred). Cheapest path to a second `close_independent_of_fill` source.
12. **Focus markets sharps actually price.** Premium should concentrate where Pinnacle/Betfair genuinely quote (mainstream football in-season). Obscure-league picks stay shadow.
13. **Flip `VALUE_REQUIRE_SHARP_ANCHOR` ON when coverage returns (~Aug 2026).** Off-season it zeroes premium; in-season it makes "premium" mean "sharp-anchored." `[HUMAN DECISION]`.

---

## 4. Sequenced build order (effort · impact · agent)
| # | Build | Effort | Impact | Agent |
|---|-------|--------|--------|-------|
| 1 | Log-odds ensemble true line (A1–A3) | M | **High** | vig-edge-math-engineer |
| 2 | Probit-for-symmetric + Shin n=2 closed form (A4) | S–M | Med | vig-edge-math-engineer |
| 3 | Exchange liquidity/empty-book guard (A5) | S | Med | odds-ingestion-engineer |
| 4 | Edge-uncertainty Kelly curve (B6) | S | Med | risk-kelly-engineer |
| 5 | Covariance Kelly for correlated slates (B7) | M–L | Med | risk-kelly-engineer |
| 6 | CLV time-series + 2×2 attribution (C8–C9) | M | Med (measurement) | data-engineer |
| 7 | Per-outcome `changedAt` timestamp (C10) | S | Low–Med | odds-ingestion-engineer |
| 8 | Restore Betfair delayed-key capture (D11) | M | **High (coverage)** | odds-ingestion-engineer |

Every item is TDD + wrong-game-verified + picks-only/read-only/informational. Items 1, 4, 6, 7 are pure-correctness, zero wrong-game risk → ship first.

## 5. Adding X data later (when funded / decided)
- **Clean path:** fund X API credits → `tweepy` (ToS-compliant; your keys work the moment credits exist).
- **No-credit path:** `twikit`/`Scweet` with an X **account login** — ToS-violating, ban-risk; only with your explicit informed consent and a throwaway account, never your main. Value of X data: sentiment/steam confirmation + injury-news latency — a *signal-confirmation* layer, not the core edge.

## 6. One-line honest summary
You don't need a new strategy — you need a **better anchor (log-odds ensemble), honest coverage discipline, and a second independent sharp (Betfair delayed key)**. The scoreboard is already honest; this is how you actually score.

---

## 7. X (Twitter) signal layer — Tavily-mined (web-indexed; API credit-blocked)

**Status:** X API returned `402 CreditsDepleted` (0 credits). Mined the **web-indexed slice** via Tavily (25 X-scoped queries → 318 unique X URLs). This is a confirmation/watchlist layer, **not** new mathematical alpha.

**What X CONFIRMS (no contradiction to §1–§3):**
- *"CLV is the only metric that matters"* — repeated verbatim by @BetsniperAUS, @KP_tipster, @OutlierDotBet. Your CLV-centric design is the consensus sharp view.
- *Fractional 1–2% of bankroll per bet* is the stated norm (@EVBetting) — matches your ¼-Kelly + 2% cap.
- *Fair-odds → minimum-odds +EV* method (@nishikoripicks) is exactly your pipeline.
- *Line shopping + middling* are the popular +EV tactics (@DGFantasy, @PickTheOdd).

**One actionable REFINEMENT (from @gamedaytrader_):** the strongest sharp signal is **limits going UP while odds go DOWN together** — a bare line move isn't sharp; it's confirmed only when Pinnacle *raised limits* into it. *Caveat:* you don't capture Pinnacle limit data in the scrape today, so this is aspirational until a limit-aware feed exists. It does, however, validate the §3-C10 steam design (require a sharp originator + staleness gate).

**X account watchlist** (mine for sentiment / steam / injury-latency confirmation when X is funded — signal-confirmation, NOT core edge):
`@nishikoripicks` `@OutlierDotBet` `@Pinnacle` `@ClosingDime` `@teddy_covers (VSiN)` `@EVBettors` `@PickTheOdd` `@gamedaytrader_` `@KP_tipster` `@BetsniperAUS` `@SmokeTheBooks` `@SportmarketPro` `@BettingIsCool` (+ its curated list: `@12Xpert`, `@Matthew_Trench`).

**Honest verdict:** X added *confirmation + a watchlist*, not a new edge. The plan is unchanged — the alpha is the anchor/coverage work in §3 (log-odds ensemble, market-dependent devig, covariance Kelly, Betfair-as-second-sharp).
