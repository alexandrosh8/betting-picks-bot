# Jarvis — +EV optimization blueprint (sharp-ev-picks)

## Phase 1: Research insights & alpha synthesis

### Key findings (math models, techniques, anomalies + source URL)

1. **De-vig BEFORE you aggregate, and pool in LOG-ODDS space — not linear.** Any nontrivial linear average of calibrated forecasts is provably under-confident (pulled toward 50%, loses tail sharpness; Ranjan & Gneiting 2010, Thm 1). The only non-extremizing, order-invariant pool is geometric-mean-of-odds / logit pooling: `consensus_p = sigmoid(Σ wᵢ·logit(pᵢ))`. This is the single most load-bearing design choice for a synthetic line, and it matters most exactly where this project lives — heavy NBA favorites and football 1X2 longshots. Source: Gneiting & Ranjan TR543 https://stat.uw.edu/sites/default/files/files/reports/2008/tr543.pdf ; Satopää et al. 2014, doi:10.1016/j.ijforecast.2013.09.009 .

2. **Correlated books are the #1 pitfall — "number of books" is NOT the lever.** Soft/regional books copy/shade Pinnacle, so equal-weighting 100 of them just re-weights Pinnacle with noise. Bates-Granger covariance-aware "optimal" weights blow up because Σ is poorly estimated; correlation-ignoring equal/shrinkage weights win in practice. Fix: curate a small set of genuinely independent sharps (Pinnacle, Circa, Betfair/Smarkets exchange, BetCRIS) and treat soft books only as the value target. Source: Wang/Hyndman/Li/Kang 2022, arXiv:2205.04216 ; Winkler & Makridakis 1983.

3. **Inverse-variance weighting with shrinkage toward equal weights** is the principled liquidity weighting: `wᵢ = λ/K + (1−λ)·volᵢ/Σvol`, λ≈0.5. With only 4–6 sharps, learned weights are noisy, so shrinkage fights the "forecast-combination puzzle" and degrades gracefully to equal-weight log-odds when limit/liquidity proxies are missing. Source: DerSimonian & Laird 1986, Controlled Clinical Trials 7(3):177-188 ; oddsraven.com/methodology (Model A).

4. **Shin's z is a fixed-point iteration, not a generic optimizer, and has an exact 2-outcome closed form.** n≥3: `z_{m+1} = (Σ √(z_m² + 4(1−z_m)·πᵢ²/B) − 2)/(n−2)`, z₀=0, πᵢ=1/oᵢ, B=Σπᵢ; then `pᵢ = (√(z² + 4(1−z)πᵢ²/B) − z)/(2(1−z))`. For 2 outcomes: `z = ((B−1)(D²−B))/(B(D²−1))`, D=π₁−π₂, and Shin ≡ Additive — zero solver cost. Source: mberk/shin https://github.com/mberk/shin ; Jullien & Salanié 1994 doi:10.2307/2235458 ; Strumbelj 2014 doi:10.1016/j.ijforecast.2014.02.008 .

5. **Method choice is market-dependent — Shin is NOT universally better.** On all 380 EPL 2024/25 closing 1X2 matches, RPS ranked multiplicative BEST (0.19724) and power WORST (0.19739); differences negligible. Shin/power earn their keep only on documented favourite-longshot-bias markets (racing, MMA, tennis dogs, longshot n≥3). For 1–3% vig sharp books, all methods converge — so multiplicative is defensible and the bigger lever is HOW MANY independent sharps you devig. Source: penaltyblog blog "From Biased Odds to Fair Probabilities" (2025-09-14) https://docs.pena.lt/y/implied .

6. **Edge-uncertainty shrinkage IS choosing the Kelly fraction (Rising–Wyner Thm V.1).** Full-Kelly on a shrinkage estimator `α·r_f + (1−α)·μ̂` is IDENTICAL to (1−α)-Kelly on raw μ̂. So "shrink the edge by α" ≡ "multiply λ by (1−α)". Baker & McHale: the optimal Kelly multiplier is monotonically DECREASING in Var(p̂). This converts the flat 0.25 into a defensible per-pick curve. Source: Rising & Wyner https://ryansbrill.com/pdf/statistics_in_sports_papers/Kelly1956_Wyner2012.pdf ; Baker & McHale 2013 doi:10.1287/deca.2013.0271 .

7. **Correlated overbetting is large and quantified.** Naive independent Kelly overbets ~2.3× once leg correlation ρ>0.4; a ρ=0.6 2-leg example: independent 5.2%/leg (8.3% ruin/100 bets) vs covariance-aware 2.8%/leg (1.9% ruin). Use Busseti-Boyd Risk-Constrained Kelly with a Ledoit-Wolf-shrunk residual covariance. Source: Busseti, Ryu & Boyd 2016 https://stanford.edu/~boyd/papers/pdf/kelly.pdf ; Han, Yu & Mathew 2019, Quantitative Finance 19(2).

8. **Steam = velocity × breadth × magnitude, gated by staleness.** Flag when M of N tracked books move the SAME direction by ≥X within window T (typ. 0.5–1.5pt / ~30bps, 60–180s, 3–4 of 8+), requiring a sharp originator. Three mandatory filters: magnitude, direction (else it's uncertainty, not steam), and staleness (drop books not updated in >5 min — their "move" is a delayed feed catching up). The cheapest accuracy win is a per-outcome `changedAt` timestamp: one snapshot detects a stale soft book. Source: agentbets.ai/sharp-betting/steam-move-detection-bot ; oddspapi.io/blog/steam-move-detector-python ; github.com/neeljshah/linewatch .

9. **Good-vs-bad variance is a 2×2 of sign(CLV)×sign(PnL), and settlement NEVER changes CLV.** Positive-CLV loss = good bet/bad variance; negative-CLV win = lucky/bad bet. Devig the close before comparing (raw CLV on a vigged close overstates edge by ~half the overround). Source: comparenbet.org/guide-closing-line-value ; project app/backtesting/clv.py already implements log-ratio CLV correctly.

### Validated links (critical resources, how each applies here)

- **mberk/shin** https://github.com/mberk/shin — drop-in Shin fixed-point + n=2 closed form. Apply: harden `app/probabilities/devig.py:155 _shin` with the closed-form 2-outcome branch and discriminant clamp.
- **cvxgrp/kelly_code** https://github.com/cvxgrp/kelly_code — real, runnable Risk-Constrained Kelly (CVXPY/ECOS). Apply: the portfolio path for ≥3 correlated simultaneous picks in `app/risk/`.
- **penaltyblog** (already a project dep, app/models/football_dc.py) https://docs.pena.lt/y/implied — `calculate_implied(odds, method=…)` returns method_params (Shin z, power k). Apply: reuse rather than reimplement; cross-check oracle for devig unit tests.
- **adoniscares7-sketch/p2w-math** `p2w_math/devig.py` (fetch via raw.githubusercontent.com/.../main/p2w_math/devig.py; root has a stray `._.git`) — fail-CLOSED de-vig WATERFALL (Pinnacle→Circa→multiplicative-median→Shin, returns None never fabricates). Closest existing pattern to a sharp-anchored synthetic line.
- **neeljshah/linewatch** https://github.com/neeljshah/linewatch — `SteamDetector(threshold_bps, window_minutes, min_books, require_pinnacle)` + Shin-devigged `MarketSnapshot`. Closest reference for the steam layer. (File-level vetting required before adoption.)
- **oddsraven.com/methodology** — cleanest spec of inverse-variance log-odds with shrinkage + 14-day stale-source exclusion.
- **neeljshah/clvtrack** https://github.com/neeljshah/clvtrack — `report() → {mean_clv_bps, t_stat (block bootstrap), p_value, n, by_market, by_time_bucket, by_kelly_bucket}`. Design reference for the CLV attribution cuts; `track.py` was NOT fetchable — treat as design reference only.
- **DO NOT ADOPT:** `michaelschecht/Edge-Radar` and `neeljshah/kellycorr` — Edge-Radar places Kalshi orders (read for weighting constants ONLY; importing execution violates HARD SAFETY RULE 1); kellycorr's `src/` is a 22-byte stub (cite the principle, build on cvxgrp/kelly_code). Betfair as a sharp anchor must use the **delayed** app key — support.developer.betfair.com states live-key read-only is not permitted (updated 2026-06-02).

---

## Phase 2: Architectural optimization blueprint

### Ingestion layer (read-only steam capture; app/ingestion + scheduler)

The honest constraint first: **the OddsPortal `.dat` feed is request-response — there is no public WebSocket**, and every source today is GET-poll (oddsportal_json.py:977/:1088, pinnacle_arcadia.py:176, betfair_exchange.py:437). "Faster sharp-move capture" must come from **tiered cadence + delta ingestion + dual timestamps**, not a socket. WebSocket/SSE scaffolding is only worth building if a paid push vendor (Odds-API.io, OpticOdds) is later adopted as `ODDS_SOURCE` — defer to a costed ADR.

Concrete changes:

1. **Imminence-tiered cadence (highest-leverage, low-risk).** Today `poll_odds` is a single global `IntervalTrigger(poll_interval_seconds=300)` (scheduler.py:436) over a uniform `oddsportal_days_ahead` window. Replace with three lanes keyed off kickoff time: a **fast lane** (e.g. 45–60s) for events kicking off within ~30 min, the existing 300s for same-day, and a slow lane (≥900s) for days-ahead. Implement as a `kickoff_tier(event)` classifier feeding distinct APScheduler jobs (or a single job that filters its candidate set per invocation). This is the only latency lever that respects the scrape's ToS/ban profile — scale fast-lane load across the existing rotating proxy pool (oddsportal.py:1185), not by hammering one IP (per CLAUDE.md).

2. **Dual timestamps now.** Persist `provider_ts` (the existing `d.time-base`, oddsportal_json.py:496) AND a nanosecond `received_at_ns` separately on every `OddsSnapshotIn`. Compute and log per-source latency `received_at − provider_ts` to detect a degrading scrape BEFORE it poisons steam signals. Aligns with the project UTC hard rule (naive datetime = bug). Add chrony to the VPS Docker host for cross-source clock discipline.

3. **Per-(market,book,outcome) change tracking + content-hash dedup.** Betfair already has `_seen_price` (betfair_exchange.py:594); generalize it. Tag each raw row with `event_id = sha256(provider+market+selection+price+provider_ts)[:16]` for idempotent dedup (the same price reappears every poll), and store `last_changed_at` per outcome so you can compute the linewatch staleness filter in ONE snapshot. Check whether the OddsPortal `.dat` JSON exposes a per-price change time; if not, diff consecutive polls from `odds_snapshots`.

4. **A read-only steam detector** (new `app/edge/steam.py`, pure-math boundary) consuming the snapshot delta: N-of-M same-direction ≥ threshold-bps within window, requiring a sharp originator, with mandatory magnitude + direction + staleness filters. Output is an informational surfacing signal on a Pick — never an action.

5. **Per-source token-bucket budget** (gap: no standing RPS object today; only per-cycle `json_concurrency=8` + `oddsportal_request_delay=1.0`). Add an explicit budget guard so fast-lane concurrency scaling can't silently raise ban risk.

### Math/analytics engine (Synthetic True Line; app/probabilities/devig.py + app/edge/value.py)

The current anchor is **winner-take-all**: `_named_sharp_anchor` (value.py:435–463) returns the FIRST sharp in priority order and discards every other sharp + all soft data; the consensus fallback blends in **odds space** with an **unweighted median** (value.py:478–492). Build a Synthetic True Line as a new pure function (route through `vig-edge-math-engineer`, failing-test-first, new ADR):

1. **Devig EACH book first, then pool in LOG-ODDS** (not odds-space median). New `synthetic_fair_probs(books, weights, devig_method)` in `app/edge/value.py`:
   - per book: `effective_odds` (commission-net, value.py:173) → `devig(...)` → per-book fair vector;
   - pool per selection: `consensus_p = sigmoid(Σ wᵢ·logit(pᵢ))`, then renormalize across the market so Σ=1. This strips each book's own overround before combination and fixes the tail miscalibration that odds-space median leaks.

2. **Curated independent-sharp set + shrinkage weights.** Restrict pool inputs to the SHARP_BOOKS set (value.py:29) — do NOT count correlated soft books as votes. Weight `wᵢ = λ/K + (1−λ)·volᵢ/Σvol` with λ≈0.5; where no liquidity proxy exists, degrade to equal-weight (λ=1). This is the lever the Pinnacle-coverage memo points at: extract sharp signal even when no single sharp prices the FULL market (~1% Pinnacle anchor, ~37% structural sharp coverage today).

3. **Blend Pinnacle + Betfair + Smarkets when all price the market** instead of hard-ordering them — they are currently never averaged.

4. **Harden Shin** (devig.py:155): add the exact n=2 closed form `z = ((B−1)(D²−B))/(B(D²−1))` (zero solver cost for the bulk of NBA/soccer 2-way side/total markets), and clamp the discriminant `z²+4(1−z)π²/B` ≥0 with multiplicative fallback on malformed/arb-crossed prices.

5. **Propagate dispersion, not just a point estimate.** `anchor_fair_probs` returns `{sel: float}` (value.py:251). Add a cross-book/cross-method **fair-prob variance** (you already compute the 3-method disagreement spread for value_filter.py:63 — reuse it). Feed that variance into (a) a continuous widening of `min_edge` on high-disagreement markets instead of the single binary swap-reject (value.py:356), and (b) the staking α below. Surface a quality flag when fair_prob fell back to emergency multiplicative (devig.py:80/103/125/146/169/187) so an ensemble can penalize it.

6. **Keep method policy per-market (ADR-0006):** multiplicative for sharp/efficient closing 1X2 and liquid NBA; Shin/power only for FLB-prone or skewed inputs — validate per-league on held-out closing odds (RPS/log-loss) before switching; do NOT assume Shin > multiplicative.

### Staking & risk module (dynamic Kelly; informational only; app/risk)

Keep `recommended_stake` (staking.py:90) as the single source of truth — all modulation is a multiplier on top of `f* → ×λ → cap`, and live + backtest must route through it.

1. **Per-pick dynamic λ via edge-shrinkage = fraction equivalence (Rising-Wyner).** Add `lambda_eff = 1 − α(uncertainty)` driven by a REAL uncertainty signal: the cross-book/cross-method fair-prob variance from the math engine, model calibration variance, and the Pinnacle-coverage/thin-data flags. Higher Var(p̂) → larger α → smaller stake (Baker-McHale monotone curve). Implement as a frozen policy field in `StakePolicy`, pure-math boundary. This replaces the flat 0.25 with a defensible curve and composes cleanly with `drawdown_constrained_multiplier` already built but default-off (staking.py:47–77).

2. **Time-to-event acts THROUGH the uncertainty channel, NOT as an ad-hoc multiplier.** The prompt's "bigger near tip-off" is a hypothesis, not established. The defensible mechanism: as an event nears, edge uncertainty falls (firmer lineups) → α falls → λ_eff rises naturally. Let `α(uncertainty)` read a time-to-event term only insofar as it lowers variance; validate via walk-forward before trusting.

3. **Portfolio Busseti-Boyd RCK for ≥3 correlated simultaneous picks** (same match, same-game props, shared slate driver). Today each pick is sized independently (staking.py) with only the per-event sub-cap (exposure.py) as a correlation backstop — this overbets at ρ>0.4. Add an optional `app/risk/portfolio.py` solving the convex RCK with a Ledoit-Wolf-shrunk covariance from MODEL RESIDUALS, gated behind a policy flag, built on cvxgrp/kelly_code (NOT kellycorr). Output stays an informational recommended stake.

4. **Liquidity/depth clip (practitioner, configurable — no formula).** `recommended_stake = min(f·λ_eff·bankroll, per_bet_cap, depth_fraction·visible_size)`. Where read-only feeds expose available size, dim/clip the displayed stake for thin markets. Applied AFTER Kelly; frozen policy from Settings, not a learned coefficient.

All four REDUCE exposure under uncertainty; none exceeds the existing quarter-Kelly + 2% per-bet / 5% daily / 4% per-event caps (config.py:196–205).

### Database/logging schema (lightweight CLV + line-drift time-series)

The recommended decision is **do NOT add a new time-series table** — `odds_snapshots` is already an append-only, change-only line-drift store (models.py, `idx_odds_event_market_captured`). The gap is the read-side aggregation, not storage.

1. **Read-only drift VIEW/aggregation** over `odds_snapshots` keyed `(event_id, market, selection)` producing per-pick **open / fill / close devigged probs + hours-to-close bucket + max favorable/adverse excursion**. Decompose the open→close move into the captured leg (`q_close − q_fill` = CLV) and residual post-fill drift. Persist only cheap scalars on `picks` if query cost matters.

2. **New columns on `picks`** (additive, nullable): `provider_ts`/`received_at_ns` provenance for the fill snapshot; `fair_prob_variance` (ensemble dispersion); `lambda_eff` and `stake_uncertainty_alpha` (audit the dynamic-Kelly decision in the existing `stake_breakdown` JSONB); `steam_flag` + `steam_originator` (read-only surfacing). Reuse the existing CLV provenance fields (`has_snapshot_close`, `closing_anchor_type`, `close_independent_of_fill`) — ADR-0017 already encodes trusted-close logic correctly.

3. **CLV attribution cuts.** The trusted subset (`_aggregate_settled`, repositories.py:799–930) already strata by anchor; add `by_market`, `by_league`/country (now captured), `by_book`, an hours-to-close bucket, and a **stake-vs-Kelly bucket** (cheap sizing audit: worse CLV on bigger bets = systematic mis-sizing). Report EVERY cell or correct for multiple comparisons — never cherry-pick. Add a derived `good_bad_variance` 2×2 label `(sign(clv_log), sign(pnl))` so the operator doesn't chase results down on process-correct picks.

4. **Watch the heuristics:** `_clv_row_is_fabricated` (repositories.py:772) drops `|clv_log|>0.5` / edge>0.20 — a genuine large-edge close can be silently excluded; recompute the per-bet CLV SD `s` (retail ~0.02–0.03 is folklore) from THIS project's settled history before trusting CI widths / the >2-SE alert gate. The in-memory exposure ledger (exposure.py) assumes single-process — a DB-backed ledger table is needed before any multi-process run.

---

## Phase 3: Pseudo-code implementation

All blocks are picks-only / read-only / informational. No order, login, exchange, or execution code.

### (a) Async multi-book line compiler → synthetic vig-free fair (log-odds pool)

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from app.probabilities.devig import devig, DevigMethod

@dataclass(frozen=True)
class BookQuote:
    book: str                       # canonical name
    selection: str
    decimal_odds: float             # effective (commission-net) odds
    provider_ts: datetime           # feed observation time (UTC-aware)
    received_at_ns: int             # local ingestion clock
    liquidity_proxy: float | None   # limit / matched volume, if known

# Curated INDEPENDENT sharps only — soft books are the value target, not votes.
SHARP_SET = ("pinnacle", "betfair exchange", "smarkets", "betcris")

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _shrinkage_weights(vols: list[float | None], lam: float = 0.5) -> np.ndarray:
    k = len(vols)
    eq = np.full(k, 1.0 / k)
    if any(v is None for v in vols):          # degrade gracefully to equal weight
        return eq
    v = np.asarray(vols, float)
    inv = v / v.sum() if v.sum() > 0 else eq
    return lam * eq + (1.0 - lam) * inv        # OddsRaven Model A

def synthetic_fair_probs(
    quotes_by_book: dict[str, dict[str, float]],   # {book: {selection: eff_odds}}
    liquidity: dict[str, float | None],
    method: DevigMethod = DevigMethod.MULTIPLICATIVE,  # sharp 1-3% vig => methods converge
    lam: float = 0.5,
    min_books: int = 2,
    outlier_pp: float = 0.04,                       # drop a sharp > 4pp off the median
) -> dict[str, float] | None:
    """Devig EACH sharp book, drop stale/outliers, pool in LOG-ODDS, renormalize.
    Returns None (fail-closed) if too few independent sharps price the full market."""
    selections = sorted({s for q in quotes_by_book.values() for s in q})

    # 1) per-book devig -> fair vectors (probability space, vig already stripped)
    per_book: dict[str, np.ndarray] = {}
    for book, q in quotes_by_book.items():
        if book not in SHARP_SET or any(s not in q for s in selections):
            continue                                # require FULL-market coverage
        odds = np.array([q[s] for s in selections], float)
        if np.any(odds <= 1.0):
            continue
        try:
            fair = np.array(list(devig(odds.tolist(), method).values()), float)
        except ValueError:
            continue
        per_book[book] = fair

    if len(per_book) < min_books:
        return None

    # 2) outlier rejection vs cross-book median (probability space)
    stack = np.vstack(list(per_book.values()))
    med = np.median(stack, axis=0)
    kept = {b: f for b, f in per_book.items()
            if np.max(np.abs(f - med)) <= outlier_pp}
    if len(kept) < min_books:
        kept = per_book                              # keep all rather than fabricate

    # 3) LOG-ODDS pool (non-extremizing; correct at the tails)
    books = list(kept)
    w = _shrinkage_weights([liquidity.get(b) for b in books], lam)
    L = np.vstack([_logit(kept[b]) for b in books])  # (n_books, n_sel)
    consensus = _sigmoid(w @ L)                      # weighted mean in logit space

    consensus = consensus / consensus.sum()          # renormalize across market
    return {s: float(p) for s, p in zip(selections, consensus)}
```

### (b) Shin's-method de-vig (fixed-point + exact n=2 closed form, clamped)

```python
import numpy as np

def shin_devig(decimal_odds: list[float], tol: float = 1e-12,
               max_iter: int = 1000) -> tuple[np.ndarray, float]:
    """Shin (1993) insider-adjusted fair probabilities + z (informed-money fraction).
    Pure numpy, no IO. Falls back to multiplicative on malformed prices."""
    o = np.asarray(decimal_odds, float)
    if o.size < 2 or np.any(o <= 1.0):
        raise ValueError("need >=2 outcomes and decimal odds > 1.0")
    pi = 1.0 / o
    B = pi.sum()
    n = o.size

    def _mult() -> tuple[np.ndarray, float]:
        return pi / B, 0.0

    if B <= 1.0:                                     # underround: Shin undefined
        return _mult()

    if n == 2:                                       # EXACT closed form (== additive)
        D = pi[0] - pi[1]
        denom = B * (D * D - 1.0)
        if abs(denom) < 1e-15:
            return _mult()
        z = ((B - 1.0) * (D * D - B)) / denom
        z = float(np.clip(z, 0.0, 0.999))
    else:                                            # Jullien-Salanie fixed point
        z = 0.0
        for _ in range(max_iter):
            disc = z * z + 4.0 * (1.0 - z) * pi * pi / B
            if np.any(disc < 0):                     # malformed / arb-crossed
                return _mult()
            z_new = (np.sqrt(disc).sum() - 2.0) / (n - 2)
            z_new = float(np.clip(z_new, 0.0, 0.999))
            if abs(z_new - z) < tol:
                z = z_new
                break
            z = z_new

    disc = z * z + 4.0 * (1.0 - z) * pi * pi / B
    if np.any(disc < 0):
        return _mult()
    p = (np.sqrt(disc) - z) / (2.0 * (1.0 - z))
    p = p / p.sum()                                  # absorb float dust
    return p, z                                      # persist z as a sharpness feature
```

### (c) Dynamic fractional-Kelly (uncertainty + liquidity + time-to-event scaled)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DynamicKellyPolicy:
    base_lambda: float = 0.25          # quarter-Kelly floor (pro standard)
    max_stake_fraction: float = 0.02   # per-bet cap (unchanged)
    alpha_uncertainty_gain: float = 1.0  # maps fair-prob variance -> shrinkage
    depth_fraction: float = 0.10       # clip to 10% of visible size

def kelly_fraction(p: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b)

def dynamic_recommended_fraction(
    model_p: float,
    market_p: float,             # no-edge prior to shrink toward
    dec_odds: float,
    fair_prob_variance: float,   # cross-book/cross-method dispersion (Phase 2.5)
    hours_to_close: float,       # acts THROUGH uncertainty, not as ad-hoc multiplier
    visible_size: float | None,
    bankroll: float,
    pol: DynamicKellyPolicy,
) -> dict[str, float]:
    """Informational stake only. Shrinkage<->fraction equivalence (Rising-Wyner):
    shrinking the edge by (1-alpha) == multiplying lambda by (1-alpha)."""
    # Baker-McHale: alpha monotonically INCREASES with edge variance -> smaller stake.
    # Time-to-event enters only by inflating effective variance when far out.
    far_out_penalty = min(1.0, hours_to_close / 48.0)
    eff_var = fair_prob_variance * (1.0 + far_out_penalty)
    alpha = min(0.95, pol.alpha_uncertainty_gain * eff_var)

    p_shrunk = market_p + (1.0 - alpha) * (model_p - market_p)   # shrink edge
    lam_eff = pol.base_lambda                                    # (1-alpha) folded into p_shrunk

    f_star = kelly_fraction(p_shrunk, dec_odds)
    frac = lam_eff * f_star
    frac = min(frac, pol.max_stake_fraction)                    # per-bet cap

    stake = frac * bankroll
    if visible_size is not None:                                # liquidity clip (after Kelly)
        stake = min(stake, pol.depth_fraction * visible_size)

    return {"fraction": frac, "alpha": alpha, "p_shrunk": p_shrunk,
            "recommended_stake": stake}   # informational — user places any bet manually
```

---

## What to build first (mapped to this repo, prioritized)

1. **Log-odds synthetic fair line** in `app/edge/value.py` (new `synthetic_fair_probs`, failing-test-first, new ADR via `vig-edge-math-engineer`). Highest alpha: directly attacks the ~1% Pinnacle / ~37% sharp-coverage bottleneck by blending all independent sharps in probability space instead of winner-take-all (value.py:435). Reuse the 3-method spread already computed at value_filter.py:63.
2. **Harden Shin** at devig.py:155 — n=2 closed form + discriminant clamp. Cheap, deterministic, removes solver cost on the bulk of 2-way markets.
3. **Dual timestamps + per-outcome change tracking** in `app/ingestion` (provider_ts + received_at_ns; generalize Betfair's `_seen_price`). Foundation for both steam detection and the drift VIEW; no schema-heavy work.
4. **Imminence-tiered cadence** in scheduler.py:436 (fast lane near kickoff, slow lane days-ahead). The only safe latency lever given no WebSocket.
5. **Dynamic λ via edge-shrinkage** in `app/risk/staking.py` (extend the already-built-but-off drawdown multiplier at staking.py:47), driven by the new fair-prob variance.
6. **CLV attribution cuts + drift VIEW** over existing `odds_snapshots` (no new table) + the good/bad-variance 2×2 label.
7. **Defer:** Busseti-Boyd portfolio RCK (only when ≥3 correlated picks are common) and any WebSocket/push-vendor adoption (costed ADR).

## Honest alpha assessment

On a **single-region OddsPortal scrape today**, the real, durable alpha is the **log-odds synthetic fair line (#1) and the dynamic-uncertainty Kelly (#5)** — both are math-boundary changes that extract more signal and size more honestly from the data you already have, independent of latency, and they directly counter the structural sharp-coverage gap. The Shin hardening (#2) and CLV attribution (#6) are correctness/measurement wins, not new edge. The **latency work (#3, #4) is mostly marginal in this configuration**: with one scraped request-response feed gated against a 300s freshness window, you cannot meaningfully "capture the lead sharp faster" — Betfair (the sharpest price) is on the slowest transport (Playwright DOM, up to 72s/page, betfair_exchange.py:437), and steam/RLM thresholds are unvalidated practitioner folklore that will mostly fire on stale-feed catch-up unless rigorously staleness-gated. Tiered cadence is worth doing as cheap insurance and as the substrate for steam, but treat sub-cycle latency edge as **folklore until measured on this project's own CLV log** — it only becomes real alpha if a delayed-key Betfair stream or a paid push vendor is added across multiple independent sharps, which is a separate cost/ToS decision, not a code optimization.