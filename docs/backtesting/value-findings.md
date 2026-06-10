# Value-Strategy Findings — A Pick Finder That DOES Beat the Market

- **Date:** 2026-06-10
- **Strategy:** sharp-vs-soft line shopping (`app/edge/value.py`,
  `scripts/value_backtest.py`). Fair value = devig(Pinnacle pre-match). Bet the
  best available price (Max across all books) when it beats Pinnacle fair by
  ≥ threshold. Settle on the real result; CLV vs the Pinnacle **closing** line.
  **No goals model** — this prices off the sharpest market, not a prediction.

## Result: positive, conclusive CLV (11,667 matches, 6 leagues, 5 seasons)

| min edge  | bets     | hit%     | ROI%            | mean CLV (95% CI)    | beats close % |
| --------- | -------- | -------- | --------------- | -------------------- | ------------- |
| 0.000     | 9756     | 41.0     | +0.35           | +0.0120 ± 0.0016     | 61.8          |
| **0.010** | **1684** | **51.8** | **+6.75**       | **+0.0325 ± 0.0035** | **74.1**      |
| **0.015** | **615**  | **57.1** | **+9.25**       | **+0.0428 ± 0.0065** | **77.4**      |
| 0.020     | 227      | 58.6     | +5.78           | +0.0529 ± 0.0123     | 83.3          |
| 0.030     | 64       | 54.7     | −5.08 (small n) | +0.0892 ± 0.0331     | 92.2          |

**The CLV is positive and its 95% interval excludes zero by a wide margin at
every actionable threshold.** Positive CLV is the gold-standard leading
indicator of a real edge — and here the picks beat even Pinnacle's _closing_
line ~74–77% of the time. This is the opposite of the goals model
(`findings.md`: CLV −0.075, beats close 20%).

## Why this works and the goals model didn't

- The goals model tried to _out-predict_ a sharp market with a thin
  information set (goals only) → it can't, so its "edges" are model error.
- This strategy doesn't predict at all. It takes the sharpest available price
  (Pinnacle) as the best estimate of truth, then finds another book that is
  momentarily too generous on the same outcome. That gap is real
  inefficiency, and it persists to the close ~3/4 of the time.
- This is the foundation of professional value betting ("beating the closing
  line by line-shopping"), now validated on this data.

## Recommended operating point

**edge ≥ 0.015** is the sweet spot: 615 bets over 5 seasons (≈120/year/6
leagues), +9.25% backtested ROI, +0.043 CLV, 77% beat-close. Higher thresholds
have better CLV but tiny samples.

## Honest caveats (do not over-trust the headline ROI)

1. **Best-price assumption.** The backtest bets the Max line — the single best
   price across all tracked books. Capturing it live requires accounts at many
   books and prompt action. Realized prices are often a notch lower.
2. **Soft books limit winners.** Books that consistently offer beatable prices
   restrict or close winning accounts; sustained volume is harder than the
   backtest implies. CLV stays the honest metric — track it.
3. **No execution here.** This is manual decision support: the bot finds the
   value and names the book/price; the user reviews and places any bet. The
   system never places bets and nothing is a guarantee of profit.

## How it's wired

- `app/edge/value.py::find_value_bets` — pure, tested value detection.
- `scripts/value_backtest.py` — the validation above (re-runnable).
- `scripts/value_picks.py` — LIVE picks: scrape multi-book odds → find value →
  ranked list with the exact book and price to take.
