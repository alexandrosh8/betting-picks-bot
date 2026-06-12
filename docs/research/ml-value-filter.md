# ML Value Filter — Meta-Labeling the Sharp-vs-Soft Value Signal

**Date:** 2026-06-12 · **Verdict: ADOPT (meta-model)** — wired as
`app/models/value_filter.py`, enforcement flag `VALUE_ML_FILTER` **default
OFF** · **Status:** annotation-only shadow phase until live CLV confirms.

> Doctrine guard (non-negotiable): the edge is sharp-vs-soft line shopping +
> CLV discipline. ML **winner-prediction from team stats backtested NEGATIVE**
> (docs/backtesting/findings.md) and is forbidden as a strategy. "ML" here is
> **meta-modeling the validated value signal** (Lopez de Prado meta-labeling):
> a deterministic PRIMARY rule generates candidates; a SECONDARY classifier
> learns which candidates deliver positive CLV. Nothing here predicts match
> outcomes, and nothing here places bets.

## 1. Question

Given the validated value candidates (best pre-match price beats
devig(Pinnacle pre-match) fair), can a secondary model select a **better
subset** than (a) the production edge thresholds and (b) an interpretable
per-cell threshold table — measured in **held-out incremental CLV and ROI
with bootstrap CIs, never accuracy**?

## 2. Data

- **Source:** football-data.co.uk mmz4281 CSVs (free, read-only GETs, cached
  under `data/ml/cache/`). 18 leagues (the `scripts/value_backtest.py`
  baseline universe: E0 E1 E2 E3 SC0 D1 D2 I1 I2 SP1 SP2 F1 F2 N1 B1 P1 T1
  G1), markets 1x2 + ou25, **maxavg era only** (Max\*/MaxC\* columns,
  2019/20+), production v4 odds floor **1.6**.
- **Dataset:** `scripts/ml/build_value_dataset.py` →
  `data/ml/value_candidates.parquet` (sha256
  `dc67f4a8de5dbe43df3d9a0af99695d1c3ed2b6120633fb1ea62c8dec9d304b0`).
  One row per (match, market, selection) with edge ≥ 0.005.
- **Leakage contract:** every column classified ID / SIGNAL / LABEL in the
  builder's `SCHEMA`; close-derived (PSC\*/MaxC\*/PC\*) and outcome-derived
  (FTR/FTHG/FTAG) columns feed **LABELs only**. Enforced by build-breaking
  gates `assert_no_label_leak()` + `assert_feature_hygiene()`, locked by
  `tests/test_ml_dataset.py` and `tests/test_ml_filter.py`. The independent
  leakage audit (this workflow, 2026-06-12) traced every SIGNAL feature to
  pre-match columns: **clean**.
- **Features (14):** edge, fair_prob, best_price, pinn_price, overround_pinn,
  overround_best, devig_spread, book_count, day_of_week, days_to_season_end
  (fixed June-30 convention — no schedule leak), is_argmax_edge, league,
  market, selection_type.
- **Label:** `clv_max > 0` — the bet beats the **vig-free Max-of-books
  close**, the strictest reference (strips the mechanical best-of-N premium).
- **Devig:** differential_margin_weighting on BOTH fill and close (ADR-0006).

## 3. Methodology (pre-registered, frozen before the holdout was touched)

- **Splits:** TRAIN seasons 1920–2324; HOLDOUT 2425+2526, touched only under
  `--final`. This was **consultation #4** of this holdout (v3, min-odds
  re-run, v4 came before) and was declared FINAL — ROI point estimates carry
  degraded credibility; the **binding gate metric is incremental CLV vs the
  Max close**.
- **CV:** season-blocked expanding walk-forward (fit seasons [0..k-2],
  calibrate on season k-1's chronological tail — isotonic if n≥1000 else
  Platt — predict season k). The summer break is the embargo; matches never
  split across folds.
- **Model selection:** pooled out-of-fold **log-loss** (Brier reported,
  never accuracy) over 6 specs (2 logreg, 4 shallow monotone-constrained
  LightGBM). Winner: `lgbm_mcs500_n200` (OOF log-loss 0.6518, Brier 0.2288,
  ECE 0.023).
- **Selection rule:** one bet per (match, market) — argmax score, price ≥
  1.6, score ≥ cutoff.
- **Operating point (train-only choice):** max TRAIN-OOF ROI s.t. n ≥ 300
  and incCLV_max − 2·bootstrap_SE > 0 → **q\* = 0.725** (train OOF: n=716,
  ROI +8.2%, incCLV_max +0.0257 ± 0.0044).
- **Adoption gates (all four required on the holdout):** C1 incCLV_max > 2SE
  vs thr=0 null; C2 ROI ≥ volume-baseline ROI; C3 incCLV_max beats the
  per-(league,market) threshold control (tuned on TRAIN only); C4 n ≥ 300.
  Bootstrap CIs cluster by **match** (correlated 1x2/ou25 outcomes).
- **Seed:** 20260612; B=2000 bootstrap on the holdout.

## 4. Held-out results — all policies side by side (seasons 2425+2526, one shot)

| Policy (held-out)                |    n | matches | hit % |    ROI | ROI 95% CI       | CLV_max | incCLV_max vs null (±2SE) |
| -------------------------------- | ---: | ------: | ----: | -----: | ---------------- | ------: | ------------------------- |
| null thr=0 (bet everything)      | 4833 |    4282 | 38.6% |  −1.4% | [−5.1%, +2.6%]   | +0.0008 | —                         |
| **keep-baseline** volume ≥0.015  |  360 |     350 | 41.7% |  +4.7% | [−8.9%, +19.0%]  | +0.0146 | +0.0138 ± 0.0118          |
| **keep-baseline** premium ≥0.03  |   61 |      59 | 47.5% | +21.1% | [−12.6%, +56.7%] | +0.0800 | +0.0792 ± 0.0420          |
| **adopt-control** per-cell table |  767 |     741 | 41.1% |  +1.4% | [−8.0%, +11.0%]  | +0.0090 | +0.0082 ± 0.0061          |
| **adopt-ml** META q≥0.725        |  396 |     384 | 40.9% | +12.0% | [−1.6%, +26.7%]  | +0.0365 | **+0.0357 ± 0.0075**      |

Holdout reliability: ECE 0.014 (calibrated scores remain calibrated out of
sample). Sources: `data/ml/value_filter_manifest.json` (this table) and
`data/ml/threshold_control_report.json` (the standalone league-tier control
variant: n=204, ROI +3.9% [−14.6%, +22.6%], incCLV_max +0.0200 ± 0.0175 —
also weaker than META, and its Pinnacle-fill haircut goes **negative**,
ROI −9.1%).

### Adoption gates (computed by the trainer, not by hand)

- **C1 PASS** — incCLV_max +0.0357, 2SE = 0.0075 → lower bound +0.0282 > 0.
- **C2 PASS** — ROI +12.0% ≥ volume baseline +4.7%.
- **C3 PASS** — +0.0357 > per-cell control +0.0082 (≈4× at comparable n).
- **C4 PASS** — n = 396 ≥ 300.

**VERDICT: ADOPT.**

## 5. Why ADOPT (and why not the others)

- The META selection delivers ~2.5× the incremental CLV of the volume
  baseline at similar scale (396 vs 360 bets) and ~4× the per-cell control
  at half its volume — with the tightest CI of any non-trivial policy. CLV
  is the evaluation currency that predicts long-run profit; ROI agrees in
  direction.
- The premium ≥0.03 baseline has the best point ROI (+21.1%) but on **n=61**
  with a CI spanning −12.6%…+56.7% — it stays the alerted tier; the
  meta-model does not replace it, it **filters** it.
- The per-cell threshold control — the interpretable "simpler model must be
  beaten first" gate — was decisively beaten (C3). Complexity is paid for.
- Honest caveats: (a) 4th holdout consultation — ROI points are softened,
  the CLV gate is what carried the verdict; (b) execution realism — settling
  the same picks at Pinnacle instead of Max prices erases part of the edge
  (the best-of-books fill assumes shopping every book at snapshot time);
  (c) the training snapshot is football-data's ~T−1 collection, not the live
  intraday OddsPortal snapshot — a distribution shift the shadow phase must
  measure.

## 6. Production wiring (this change)

- `app/models/value_filter.py` — loads `value_filter_manifest.json` +
  `value_filter_model.txt` from `VALUE_ML_MODEL_DIR` (default `data/ml`);
  refuses manifests whose verdict ≠ ADOPT; replays the manifest calibrator
  with numpy only. Lazy lightgbm/pandas imports — the `ml` extra stays
  optional; missing deps/artifacts = unfiltered pipeline, logged.
- `run_value_pipeline` scores candidates **after** the edge gate. Scope
  gates: 1x2/ou25 only, mapped league (18 trained codes), named sharp
  anchor, price ≥ 1.6. Out-of-scope candidates are **unscored, never
  vetoed**.
- `VALUE_ML_FILTER=false` (default): scores are annotated on picks (DB
  column `picks.value_filter_score`, dashboard shows "ML 0.78" on scored
  rows) — pure shadow evidence. `true`: premium candidates scoring below
  q\*=0.725 are **demoted to the volume tier** (no alert, no exposure) so
  their CLV evidence keeps accumulating.
- Artifacts stay **out of git** (`/data/` gitignored); the deployment copy
  step is documented in `docs/deployment/openclaw-ubuntu.md`.

## 7. Reproduction

```bash
uv sync --extra ml
uv run python scripts/ml/build_value_dataset.py          # cached GETs -> parquet
uv run python scripts/ml/train_value_filter.py           # train phase only
uv run python scripts/ml/train_value_filter.py --final   # ONE-SHOT holdout + artifacts
uv run python scripts/ml/optimize_thresholds.py          # league-tier control variant
uv run pytest tests/test_ml_dataset.py tests/test_ml_filter.py \
  tests/test_value_filter.py tests/test_value_filter_ml.py -q
```

Deterministic: seed 20260612, mergesort ordering, dataset sha256 pinned in
the manifest.

## 8. What would change the answer

- **Live evidence first:** the shadow phase (flag OFF, scores annotated)
  must show score-stratified live CLV consistent with the backtest before
  `VALUE_ML_FILTER=true` is justified. If high-score picks do NOT out-CLV
  low-score picks live, the filter stays off regardless of this holdout.
- **Intraday snapshots accumulating in `odds_snapshots`:** the live change-
  only odds history will eventually support retraining on the REAL pick-time
  distribution (true openers → `open_to_signal_drift` stops being null;
  line-movement features become possible). That dataset supersedes the
  football-data T−1 snapshot when it reaches comparable scale.
- **A fresh holdout:** season 2627 provides the first never-consulted test
  set. Any threshold/feature change must re-run the one-shot protocol there
  — this holdout (2425+2526) is spent.
- **Era drift:** the model is maxavg-era only. If football-data's best-price
  pool composition changes (books added/removed), `book_count`/`overround_
best` semantics shift → retrain, don't patch.
- **Scope expansion** (more leagues/markets) requires training data for the
  new cells; the league map in `app/models/value_filter.py` deliberately
  refuses to guess.

## 9. Decision-support only

Recommended picks, edges, EV, and scores are informational. This system
never places bets, and nothing in this research changes that (ADR-0002).
