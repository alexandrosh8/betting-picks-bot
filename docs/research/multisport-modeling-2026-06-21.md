# Multi-sport modeling research — basketball / NFL / tennis (2026-06-21)

6-agent sweep (GitHub MCP + web), judged against the CLV-validated, picks-only,
leakage-safe doctrine. penaltyblog is football-only; this asked what (if anything)
to adopt for basketball, NFL, tennis.

## Headline finding (the most important one)

**No repo or paper — across all five briefs — demonstrates a market-beating model
validated by held-out CLV for ANY sport.** Every "profitable" headline is one of:
accuracy/ROI on _soft_ books, in-play, or an AI-generated portfolio case-study
(e.g. `ianalloway/nba-edge`, `gmalbert` 91.9% win-rate). This _confirms the
project's doctrine_: a points/goals/Elo model is a **SCREEN**; the validated edge
stays **sharp-vs-soft line shopping proven by CLV (>2 SE)**. The research did not
find a magic model — it found the doctrine is correct.

## Per-sport verdicts

### Basketball (NBA) — ADOPT nothing new (already correct)

- Keep the settled stack (nba_api ingestion + LightGBM screen + isotonic calib).
  NBA is the **one validated CLV sport** here.
- **Reference patterns only** (do NOT depend): `genehuh39/sports-projection` (MIT —
  sorted walk-forward CV + `shift(1)` discipline + Platt), `conorwalsh99/ml-for-sports-betting`
  (MIT, archived — verified repro of **Walsh & Joshi 2024**, the calibration-over-
  accuracy paper: calibration-based selection +34.69% ROI vs −35.17% for accuracy).
- Edge concentrates in **moneylines**; spreads/totals are near-efficient (Wang 2026).
- EuroLeague/WNBA/NBL: **no free close → informational only.** REJECT
  `giasemidis/euroleague_api` (GPLv3, unliftable), `wehoop` (R), all winner-predictor
  "nba-betting" repos as strategies.

### NFL (+ NCAA/CFL/UFL) — data spine yes, model = clean-room screen

- **ADOPT** `nflverse/nflreadpy` (MIT) as the free NFL feature/line spine (GET-only,
  `.to_pandas()` at the boundary). `nfl_data_py` is deprecated → use nflreadpy.
- **Build clean-room** (no liftable model lib exists): regularized Elo + QB-Elo +
  opponent-adjusted rolling-EPA, then a **market-regression layer** blending toward
  the devigged line (the `greerreNFL/nfelo` method — no LICENSE, ideas only).
- Honest gate: the **NFL closing line beats strong models on Brier**; thin,
  high-confidence-only. Keep NFL **visibility-only/screen** until CLV > 2 SE on a
  multi-month held-out sample. REJECT overfit repos (gmalbert), in-game WP ports.

### Tennis — SKIP for staking (data, not model, is the blocker)

- **No free closing line** in this pipeline → can't CLV-validate → can only ever be
  an **informational screen**, never staked/alerted as +EV.
- If a dashboard-only screen is ever wanted: `mbhynes/skelo` (MIT, pip) is the **one
  genuinely bindable, leakage-safe Elo/Glicko2 lib** found; surface-blend + our own
  per-surface isotonic/beta calibration. License risk: JeffSackmann data is
  CC-BY-NC (noncommercial); tennis-crystal-ball splits Apache code / CC-BY-NC data.

## Cross-cutting: calibrator + pick-optimizer

- **Best calibrator to ADD — Beta calibration** (Kull/Silva-Filho/Flach, AISTATS
  2017; lib `betacal`, MIT, TRAIN-time only). `app/models/value_filter.py` exposes
  only isotonic + Platt; beta is the documented better middle ground for small,
  skewed pick-conditional sets.
- **v3 candidate (carry, don't swap)** — Venn-Abers (`ip200/venn-abers`, MIT): tops
  the 2026 log-loss benchmark on tree models + yields a [p0,p1] uncertainty interval.
- **Offline bake-off harness** — `dholzmueller/probmetrics` (Apache-2.0, OFFLINE
  only, never a runtime dep) to rank isotonic vs Platt vs beta vs Venn-Abers on OUR
  settled pick-conditional set before an ADR locks a calibrator.
- **Pick-optimizer — NO change.** `app/risk/staking.py` already has fractional Kelly +
  the Busseti-Boyd drawdown-constrained multiplier. Simultaneous/correlated-bet
  portfolio Kelly is the only open case (only if same-game legs are ever scoped).
- **REJECT `flumine`** (betcode-org) — contains live order execution → violates HARD
  SAFETY RULE 1.

## Honest gaps

- NBA is the ONLY league with a validated forward-CLV history; EuroLeague/WNBA/NBL
  have no free close.
- NFL CLV still accruing — no NFL edge validated at > 2 SE yet.
- Tennis has no free close (re-confirmed).
- The 2026 calibrator benchmark is GENERAL tabular — the beta/Venn-Abers advantage
  must be replicated on our own 1X2 settled set before adoption.

## Concrete next steps (doctrine-fit, in priority order)

1. **Beta calibration** in `value_filter.py` (`kind=='beta'`, ~10 lines numpy + the
   `betacal` dev dep) + an offline `probmetrics` bake-off recorded here. Highest
   value / lowest risk; CLV stays the sole arbiter.
2. **nflreadpy** as the NFL data spine; clean-room Elo/EPA + market-regression
   screen, visibility-only until CLV clears.
3. **skelo** tennis surface-Elo _informational_ screen only (clearly tagged), if
   wanted — never staked.
