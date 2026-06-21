# Research brief — Cross-sport calibrator + pick-optimizer + leakage-safe betting-ML frameworks

Date: 2026-06-21 · Author: quant-sports-researcher · Status: feeds an ADR
(calibration method selection + optional portfolio-Kelly).

## Question

What single probability **calibrator** and what **EV/pick + fractional-Kelly
optimizer** would best slot into this project, given it already has:
log-loss/Brier/ignorance/ECE/MCE (`app/backtesting/calibration.py`), a
LightGBM value-filter meta-model with serialized isotonic/Platt calibrators
(`app/models/value_filter.py`), single-bet fractional-Kelly + a Busseti-Boyd
drawdown-constrained multiplier closed form (`app/risk/staking.py`), and a CLV
walk-forward validator (`app/backtesting/clv.py`)? And which canonical
betting-ML frameworks enforce walk-forward + CLV?

## Method

GitHub metadata via `gh api` / plugin MCP (stars, license, last push) + exa
semantic search for READMEs/source + arXiv for the calibration-method ranking.
Every repo judged against project doctrine (picks-only; CLV is the validated
edge; free-first; leakage-safe walk-forward; no autobet). Already-settled
repos (georgedouzas/sports-betting, ProphitBet, AlphaPy, kyleskom, NBA\_\*) were
NOT re-evaluated — see `.claude/memory/decisions.md` and
`docs/research/betting-repo-research.md`.

## Findings (cited)

### A. Calibration methods — established ranking

- **2026 benchmark, model-agnostic post-hoc calibration** (Manokhin &
  Grønhaug, arXiv:2601.19944, "Classifier Calibration at Scale", 21
  classifiers incl. LightGBM/XGBoost/CatBoost on TabArena-v0.1, 5-fold):
  **Venn-Abers gives the largest average log-loss reduction, Beta calibration
  second and improves log-loss most frequently; Platt scaling is weak/
  inconsistent; isotonic AND Platt can systematically DEGRADE proper scores
  for strong tree models.** No method dominates uniformly. This is directly
  relevant — our value-filter is LightGBM, currently calibrated with isotonic/
  Platt, i.e. the two methods the benchmark flags as risky for tree models.
- **Beta calibration** (Kull, Silva Filho, Flach, AISTATS 2017): parametric,
  3 params, order-preserving, behaves well on tails — the failure mode that
  matters for Kelly (tail mis-calibration → systematic over/under-staking).
- **Venn-Abers** (Vovk & Petej, UAI 2014): two isotonic fits (g0 on y=0
  subset, g1 on y=1), distribution-free validity under exchangeability;
  returns a [p0,p1] probability interval — a free calibration-uncertainty
  signal that maps cleanly onto a pick gate (widen/abstain when wide). IVAP
  (single calibration split) vs CVAP (cross-validated).
- **Temperature scaling** (Guo et al. 2017): one parameter, multiclass-native
  via softmax(logits/T). Best fit for 3-way 1X2 if we ever calibrate the
  Dixon-Coles joint, but for a binary value-filter it reduces to Platt.
- scikit-learn 1.9 now ships `method="temperature"` in `CalibratedClassifierCV`
  alongside sigmoid/isotonic (official docs) — but sklearn is NOT an inference
  dependency here by design (the project clean-room-replays calibrator params).

### B. Calibration libraries

| Lib         | URL                                             | Stars | License    | Last push      | Note                                                                                                             |
| ----------- | ----------------------------------------------- | ----- | ---------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| venn-abers  | github.com/ip200/venn-abers                     | 203   | MIT        | 2026-05-08     | sklearn-native binary+multiclass IVAP/CVAP; the reference impl; 100% sklearn Pipeline/CV compatible              |
| MAPIE       | github.com/scikit-learn-contrib/MAPIE           | 1558  | BSD-3      | 2026-06-19     | scikit-learn-contrib; conformal classification/regression + calibration utilities; references Venn-Abers         |
| crepes      | github.com/henrikbostrom/crepes                 | 576   | BSD-3      | 2026-06-12     | conformal classifiers/regressors + Mondrian (class-conditional) — conformal angle, not a drop-in calibrator      |
| netcal      | github.com/EFS-OpenSource/calibration-framework | 377   | Apache-2.0 | 2026-04-16     | TemperatureScaling/Beta/Platt/Isotonic/BBQ/ENIR + ECE/MCE; heavier, NN-oriented                                  |
| probmetrics | github.com/dholzmueller/probmetrics             | 65    | Apache-2.0 | active         | PyTorch hub: 15+ calibrators + L_p calibration metrics + CalArena benchmark; great for OFFLINE method bake-off   |
| betacal     | github.com/betacal/python                       | 31    | MIT        | 2021 (release) | official Beta-calibration `BetaCalibration` class; tiny, stable, ~500k downloads/mo on PyPI                      |
| calibcraft  | github.com/neeljshah/calibcraft                 | 0     | MIT        | 2026-04        | Platt/isotonic/beta + ECE/MCE extracted from a personal NBA system; honest README, but unproven — reference only |

### C. EV/pick + fractional-Kelly portfolio optimizers

- **Risk-Constrained Kelly Gambling** (Busseti, Ryu, Boyd 2016;
  arXiv:1603.06183; Stanford paper appendix B gives the exact CVXPY spec).
  This is the SIMULTANEOUS-bet generalization of the single-bet drawdown
  bound already in `app/risk/staking.py`. The convex program:
  `maximize π·log(rᵀb) s.t. 1ᵀb=1, b≥0, log_sum_exp(log π − λ·log(rᵀb)) ≤ 0`
  — the math is public-domain (paper); CVXPY/ECOS solve it directly.
  - Reference code `cvxgrp/kelly_code` (29★, **GPL-3.0**, last push **2020** —
    fails freshness AND license gates → idea/math source only, never import).
  - `cvxgrp/cvxportfolio` (1231★, **GPL-3.0**, active) — same group, mature,
    but GPL → cannot vendor; the formulation, not the package, is what we want.
  - The MO-book notebook (mobook.github.io) and CVXR docs give clean
    exponential-cone reformulations to cross-check any implementation.
- For BINARY/independent simultaneous bets (our case — picks across different
  matches are ~independent), the full RCK reduces to per-bet fractional Kelly
  with a shared λ; the portfolio coupling only bites with correlated legs
  (same-game, same-team multi-market). So a portfolio optimizer is a
  PHASE-LATER refinement, not a current gap.

### D. Betting-ML frameworks that enforce walk-forward / CLV

- None found that enforces BOTH walk-forward AND CLV better than what this
  project already has. The candidates and why they fall short:
  - **georgedouzas/sports-betting** (712★, MIT) — TimeSeriesSplit-guarded
    backtest, but settles at near-closing decision odds, flat stake, no
    signal-time prices, no CLV. ALREADY SETTLED as idea-only 2.5/10
    (decisions.md). Do not re-adopt.
  - **flumine** (betcode-org) — event-based trading framework with
    `market.cancel_order` / `replace_order` / live execution → **REJECT
    outright (autobet, violates HARD SAFETY RULE 1)**; not even idea-safe.
  - **Edge-Radar**, **Sports-Steve**, **OmegaSports**, **Sport-suite** —
    surfaced; all couple to live order execution and/or are unvetted
    single-author apps. Sport-suite documents expanding-window walk-forward +
    Platt + CLV honestly, but it's a sprawling NBA-prop monolith, not a
    liftable library. Reference-only / reject; none inspected file-by-file for
    adopt.

## Implications for THIS project

1. The value-filter currently uses isotonic/Platt — exactly the pair the 2026
   benchmark flags as risky for tree models. **Beta calibration is the
   lowest-risk upgrade**: 3-param, tail-safe, and trivially clean-room-
   replayable in inference (the project already serializes calibrator params
   and re-applies them with numpy — adding `kind=="beta"` to
   `value_filter.calibrate()` is ~10 lines). No new INFERENCE dependency;
   `betacal` only at TRAIN time in the trainer.
2. **Venn-Abers is the higher-upside but heavier option**: best log-loss in
   the benchmark AND yields a calibration-uncertainty interval that could feed
   an abstain gate. Cost: it is non-parametric (two isotonic fits) → the
   serialized form is the same piecewise-linear thresholds the project already
   stores for isotonic, so inference replay is feasible; but it needs careful
   train-side integration and a held-out CLV check before adopt.
3. CLV stays the arbiter regardless. Calibration is a SCREEN/diagnostic, not
   the validator (doctrine). Any calibrator swap must be proven by the same
   one-shot held-out incremental-CLV test the value-filter already passed
   (>2 SE), not by Brier/log-loss alone.
4. Portfolio Kelly (Busseti-Boyd simultaneous) is NOT a current gap: our picks
   are cross-match ~independent, so per-bet fractional Kelly with the existing
   drawdown multiplier is the correct closed form. Revisit only when
   correlated same-game multi-markets are added (phase 6+).

## Recommended decision

- **Calibrator to adopt:** add **Beta calibration** as a trainer-time option
  (`betacal`, MIT) with a 3-param numpy inference replay in
  `app/models/value_filter.calibrate()`; gate adoption on held-out CLV, not
  Brier alone. Carry **Venn-Abers** (`ip200/venn-abers`, MIT) as the
  reference candidate for a v3 calibrator bake-off (its [p0,p1] interval is
  the abstain-gate upside). Use **probmetrics** (Apache-2.0) OFFLINE only, as
  the bake-off harness to rank isotonic vs Platt vs beta vs Venn-Abers on the
  value dataset before committing.
- **Optimizer:** keep the current single-bet fractional Kelly + Busseti-Boyd
  drawdown multiplier. Record the RCK simultaneous formulation
  (arXiv:1603.06183, appendix B) as the math source for a FUTURE correlated-
  legs optimizer; do NOT vendor `cvxgrp/kelly_code`/`cvxportfolio` (GPL).
- **Frameworks:** adopt nothing as a dependency — the project's walk-forward +
  CLV stack is ahead of every surveyed framework. flumine is REJECTED on
  safety grounds.

## Open questions

- Does Venn-Abers' [p0,p1] interval width carry exploitable abstain signal on
  OUR pick-conditional (selection-biased) settled set, or is it dominated by
  the existing q\* operating point? (needs an offline study on the value set.)
- Beta vs Venn-Abers on the SPECIFIC 1X2/OU2.5 value distribution — the
  benchmark is general tabular; replicate the ranking on our data via
  probmetrics before the ADR locks a choice.
