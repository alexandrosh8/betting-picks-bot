# Premium Tier v2 — Value-Filter Retrain Round (SHADOW-CANDIDATE)

- **Date:** 2026-06-12
- **Verdict:** **stage-v2-shadow** — the v2 model ships as a
  **SHADOW-CANDIDATE** (manifest verdict `CANDIDATE`, annotation-only).
  It does NOT replace the v1 ADOPT filter and it can NEVER demote picks.
- **Artifacts (gitignored):** `data/ml/value_filter_manifest_v2.json`,
  `data/ml/value_filter_model_v2.txt`, sweep log
  `data/ml/value_filter_v2_sweep.csv`, dataset
  `data/ml/value_candidates_v2.parquet` (sha256 in the manifest).
- **Code:** `scripts/ml/build_value_dataset.py` (v2 features),
  `scripts/ml/train_value_filter_v2.py` (pre-registered protocol frozen in
  its docstring), `app/models/value_filter.py` (shadow loader),
  `app/edge/value_policy.py` (premium-tier knobs, all default-off).

## 0. Spent-holdout discipline statement (binding)

Seasons **2425+2526 are SPENT** as a holdout (consulted 4x —
`.claude/memory/decisions.md`, `docs/research/ml-value-filter.md`). This
round NEVER consulted them: the v2 trainer drops both seasons at load time
(asserted in `load_candidates_v2`), and **no 2425/2526 quantity appears in
the v2 manifest, the sweep log, or this document**. (The dataset parquet
contains the full universe including those seasons for future one-time use;
no computation in this round consumed them. The manifest's
`spent_seasons_never_loaded` field and the digit-substring coincidences
inside random-draw hyperparameter floats in the sweep CSV are the only grep
hits — neither is a quantity.) Every number below is either
**[TRAIN-OOF]** (nested season-blocked walk-forward within seasons
<= 2324 — the development sandbox) or **[FRESH-ONE-SHOT]** (a pre-registered
single look at never-before-loaded divisions). **The binding verdict for v2
is live shadow CLV + the one-shot fresh 2627 season.** Any 2425/2526 number
anywhere else must be labeled CONTAMINATED-REFERENCE and is never a
headline.

## 1. Research digest (what informed the round)

**Strategy.** The edge remains sharp-vs-soft line shopping + CLV; ML only
scores VALUE CANDIDATES (meta-labeling: Lopez de Prado, _Advances in
Financial Machine Learning_, 2018, ch. 3) — winner-prediction ML backtested
negative and stays forbidden (`docs/backtesting/findings.md`). Selection
currency is pooled OOF log-loss for model choice and CLV+ROI with
match-clustered bootstrap CIs for policy choice — never accuracy.
Learning-to-rank was evaluated and **rejected**: ranker outputs are
relevance scores, not the calibrated probabilities the pipeline consumes;
(match, market) qid groups hold 1–3 rows → near-zero effective pairs;
bucketing `clv_max` would be a fresh researcher degree of freedom.

**GBM findings** (docs-grounded; `docs/research/lightgbm-vs-xgboost.md`,
ADR-0009). LightGBM stays the default engine, XGBoost 3.2 the standing
challenger promoted only on Brier/ECE — implemented exactly so here.
LightGBM Parameters-Tuning guidance drove the sweep space
(`num_leaves < 2^max_depth`, `min_data_in_leaf` in the hundreds at our n,
`bagging_fraction` requires `bagging_freq>=1` — structurally OFF in v1,
fixed in v2 — `path_smooth`, `extra_trees`, `max_bin`); monotone
constraint on `edge` only, `monotone_constraints_method="advanced"`
(penalization method). DART/GOSS excluded (the early-stopping callback
hard-disables itself in dart mode; GOSS is a large-data speed lever).
XGBoost per its categorical + monotone tutorials: `tree_method="hist"`,
`enable_categorical=True`, `max_bin=512` to soften histogram/monotone
interaction. Early-stopping protocol pre-registered as "option A": eval set
= the fold's CALIBRATION season (double duty with isotonic — both freezing
operations on strictly pre-predict data; residual mild optimism disclosed
in the manifest as `calib_season_ece`).

**Data findings.** Dataset v2 keeps the v1 train universe byte-identical
(18 football-data.co.uk leagues, maxavg era, 1x2+ou25, min_odds 1.6, devig
`differential_margin_weighting` per ADR-0006; 18,786 labeled train rows)
and adds **37 new pre-match feature columns** (`v2_new_features` in the
manifest): devig-disagreement and price-shape features (`fair_mult_delta`,
`fair_shin_delta`, `fair_power_delta`, `price_ratio_best_pinn`,
`prob_gap_pinn_best`, `fair_prob_rank`), leakage-safe rolling form r5/r10
(goals, shots, SOT, corners, for/against — windows over strictly prior
completed fixtures), Understat xG r5/r10 + xG-overperformance via the
penaltyblog scrapers (big-5 top flights 2014/15+, nullable elsewhere;
versioned entity-resolution table in the builder), and an `odds_band`
bucket (favourite–longshot-bias literature: margin is loaded onto
longshots — which bands are soft must be learned, never asserted). The
finalize-round leakage audit traced **all new builder features to pre-match
data**: rolling form uses only completed prior fixtures, xG joins are
strictly pre-kickoff, and no closing-odds-derived column enters the feature
set (the close exists only inside the `clv_max` label).

## 2. What was built

1. **Dataset v2** (`build_value_dataset.py`): the 37 new columns above plus
   a `consulted_before` flag separating the 18 consulted leagues from the
   never-loaded fresh divisions (EC/SC1/SC2/SC3).
2. **Trainer v2** (`train_value_filter_v2.py`): pre-registered protocol
   frozen in code — 3 expanding season folds (fit → calib → predict),
   seeded random sweeps (100 LGBM / 80 XGB draws), four arms
   (`v1_grounding` reproduction, `v2_lgbm_v1feat` hyperparameter-only lift,
   `v2_lgbm` new-feature headline, `xgb_v2feat` challenger), v1's
   operating-point criterion unchanged, artifacts to `*_v2` paths only,
   manifest verdict hard-coded `CANDIDATE` (this script can never emit
   ADOPT). An honest amendment note in the docstring records that final
   feature-set selection was widened to the global OOF argmin after the
   train-OOF tables existed but strictly before the fresh slice was
   consulted.
3. **Shadow loader** (this finalize round): `ValueFilterModel.load` gains
   `manifest_filename` / `model_filename` / `allow_shadow`;
   `Settings.value_ml_manifest_filename` / `value_ml_model_filename` /
   `value_ml_manifest_allow_shadow` (env `VALUE_ML_MANIFEST_ALLOW_SHADOW`).
   A non-ADOPT manifest loads ONLY with allow_shadow, is marked
   `shadow=True`, and may only annotate: the pipeline demotion branch
   refuses shadow models and the composition root force-disables
   enforcement (with a loud error) if `VALUE_ML_FILTER=true` meets a shadow
   manifest. **Enforcement always requires a true ADOPT manifest.** The
   config DEFAULT still points at the v1 ADOPT artifacts — the v2 manifest
   does not pass the loader's ADOPT gate semantics, so no default changed.

## 3. Results (all labels explicit)

**Model selection [TRAIN-OOF], pooled calibrated OOF log-loss:**

| Arm                                    | Features | OOF log-loss | OOF ECE |
| -------------------------------------- | -------- | ------------ | ------- |
| v1_grounding (exact v1 winner)         | 14 v1    | 0.65175      | 0.0225  |
| **v2_lgbm_v1feat (selected, draw 81)** | 14 v1    | **0.64968**  | 0.0217  |
| v2_lgbm (new features)                 | v1+37    | 0.65230      | 0.0279  |
| xgb_v2feat (challenger)                | v1+37    | 0.65082      | 0.0179  |

- The grounding arm reproduces the v1 manifest number exactly (protocol
  parity verified).
- **Honest no-lift:** the 37 new features did NOT improve pooled OOF
  log-loss (`feature_lift_v1_to_v2_logloss = −0.0026` — the best
  v1-feature sweep beats the best v2-feature sweep). The selected final
  model keeps the **14 v1 features** with swept hyperparameters; the v2
  gain over v1 is hyperparameters only (0.65175 → 0.64968).
- XGBoost challenger refused by the pre-registered rule (OOF log-loss not
  strictly below the LGBM winner's).
- Final fit: 1920–2223, early-stop + isotonic on 2324; `book_count` gain 0
  in the final model (candidate for removal in v3).

**Operating point [TRAIN-OOF]** (criterion unchanged from v1: max train-OOF
ROI s.t. n>=300 and incCLV_max − 2SE > 0): q\* = 0.725, n=892,
ROI +14.3% [boot CI +6.6%, +22.8%], incCLV_max +0.0249 (boot SE 0.0019,
CI [+0.0211, +0.0282]), incCLV_pinn +0.0263. Development-sandbox numbers —
mildly optimistic by construction, never headline evidence.

**Fresh never-consulted divisions [FRESH-ONE-SHOT]** (EC/SC1/SC2/SC3,
seasons 1920–2324 — a pre-registered single look; transportability check,
NOT an adoption gate): scored rows 2,117; log-loss 0.7118, Brier 0.2482,
ECE 0.0591 (calibration degrades out of vocabulary — fresh league levels
are unseen categories routed through LightGBM's missing bin).

| Policy                | n     | ROI    | incCLV_max (vs thr=0 null)    |
| --------------------- | ----- | ------ | ----------------------------- |
| null (bet everything) | 3,243 | +1.1%  | —                             |
| volume edge>=0.015    | 547   | +13.7% | +0.0253                       |
| premium edge>=0.03    | 78    | +6.0%  | +0.0438                       |
| META q>=0.725         | 138   | +7.9%  | +0.0338 [CI +0.0246, +0.0438] |

Honest reading: the meta filter **transports** (clearly positive
incremental CLV vs the null on divisions it never saw) but does **not
separate from the plain premium threshold** in this slice (overlapping CIs,
premium n=78) — exactly the question live shadow + 2627 must answer.

## 4. Why SHADOW-CANDIDATE and not ADOPT

The v1 ADOPT verdict was earned on a one-shot holdout that is now spent.
v2 has no unconsulted held-out season left to earn the same verdict, and
train-OOF improvements are development-sandbox evidence by definition. An
honest manifest therefore says
`CANDIDATE (binding verdict: live shadow CLV + fresh 2627 season)` — and
the loader's ADOPT gate refuses it for enforcement by construction. The
trainer cannot emit ADOPT at all; flipping the verdict requires the
evidence in §5 attached to the manifest.

## 5. Pre-registered path to enforcement

1. **Live shadow stratification.** Optionally point live annotation at v2
   (`VALUE_ML_MANIFEST_FILENAME=value_filter_manifest_v2.json`,
   `VALUE_ML_MODEL_FILENAME=value_filter_model_v2.txt`,
   `VALUE_ML_MANIFEST_ALLOW_SHADOW=true`) — scores land in
   `picks.value_filter_score`, behavior unchanged (this swaps which model
   annotates; v1's live score stream pauses while v2 shadows — the v1
   evidence already booked is unaffected). Gate: with n >= 300 scored
   premium+volume picks with settled closes, score-stratified live CLV
   (>= q\* vs < q\*) must show positive incremental CLV vs the unfiltered
   premium tier at 2SE.
2. **One-shot fresh 2627 season.** When 2627 football-data closes exist:
   run the FROZEN v2 artifacts (no refit, no protocol change, one look)
   through the v1 adoption gates on that season (n>=300, incCLV_max − 2SE
   > 0, beats the volume baseline and the per-cell threshold control).
3. Only if (1) and (2) pass may the manifest verdict be edited to ADOPT
   (with the evidence embedded); only then can
   `VALUE_ML_FILTER=true` demote on v2. Failing either, v2 is retired and
   v1 remains the only enforcement-eligible artifact.

## 6. Free-data integration roadmap

| Source                                                       | Status                                                                                                                                           | Next step                                                                                                                                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Understat xG** (penaltyblog scrapers, big-5 2014/15+)      | In dataset v2; not in the selected feature set (no OOF lift)                                                                                     | Revisit in v3 with interaction-aware models or per-league models; live-side rolling-xG parity needed only if a future selected feature set uses it                                            |
| **FPL injuries** (official Fantasy Premier League API, free) | Research-flagged, not integrated                                                                                                                 | EPL-only availability/chance-of-playing flags as pre-match team-news features; must be snapshotted pre-kickoff (point-in-time store) to stay leakage-safe; evaluate on nested CV <= 2324 only |
| **sbrscrape / SBR** (free NBA odds)                          | Wave-4 verdict: idea worth keeping, needs its own evaluation (`docs/research/betting-repo-research.md`, `docs/research/nba-repo-evaluations.md`) | License/safety/coverage evaluation before any code lands; phase-5 NBA line-shopping spine candidate                                                                                           |

## 7. Reproduction

```bash
uv run --extra ml --extra football python scripts/ml/build_value_dataset.py
uv run --extra ml python scripts/ml/train_value_filter_v2.py
uv run pytest -q tests/test_train_value_filter_v2.py tests/test_value_filter.py tests/test_value_filter_ml.py
```

Seeds: model 20260612, sweep 20260712. The trainer is deterministic given
the dataset; the dataset sha256 is pinned in the manifest.

## 8. Decision-support only

Scores, edges, EV, and stakes remain informational for manual betting.
Nothing here places bets, and no number above is a promise of profit.
