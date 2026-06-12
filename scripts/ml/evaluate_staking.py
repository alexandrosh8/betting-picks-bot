"""Track D — recommended-stake policy evaluation on the premium pick stream.

Question: does the drawdown-constrained fractional-Kelly variant
(app/risk/staking.py — STAKE_MAX_DRAWDOWN / STAKE_MAX_DRAWDOWN_PROBABILITY,
default OFF) beat the deployed default (0.25x Kelly + 2% per-bet cap) as a
RECOMMENDED-stake policy? Stakes are informational only — this platform never
places bets — and this script changes NO defaults; it prints the exact .env
lines a human could adopt if the pre-registered criterion passes.

Methodology (kelly-bankroll + walkforward-backtest skills):
- The premium pick stream is regenerated with the deployed selection config
  (differential-margin devig, edge >= 0.03, odds >= 1.60, markets 1x2 + ou25,
  one bet per (match, market) — same construction as scripts/value_backtest.py)
  on TRAIN seasons <= 2324 ONLY. Seasons 2425 + 2526 are a SPENT holdout
  (docs/research/ml-value-filter.md, docs/research/premium-tier-v2.md);
  staking choice is strategy-adjacent, so the spent slice is never consulted
  here — assert_train_only() hard-fails on any season newer than 2324.
- Every per-bet stake fraction routes through app.risk.staking
  .recommended_stake — the SAME code path live picks use (kelly-bankroll
  checklist: live and backtest must share the Kelly path).
- Circular block bootstrap of the chronologically-ordered stream (blocks of
  consecutive picks preserve streak/regime structure), >= 10k bankroll paths,
  plus a single chronological sanity pass in real order.
- The decision criterion below is PRE-REGISTERED: written in code before the
  first real run, per the project's pre-registration discipline.

Honest limitations: the daily-exposure ledger (5% cap) is not simulated —
premium picks are sparse enough that the 2% per-bet cap dominates; the
bootstrap treats blocks as exchangeable; Kelly sizes on the devigged Pinnacle
pre-match fair probability and settles on real outcomes, so any optimism in
p flows into sizing exactly as it would live.

Run (offline once data/ml/cache is populated; any fetch is a read-only GET):

    uv run python scripts/ml/evaluate_staking.py
    uv run python scripts/ml/evaluate_staking.py --paths 20000 --block-size 50

Decision-support only — nothing here places bets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx
import numpy as np

from app.ingestion.football_data import fetch_season_csv
from app.probabilities.devig import DevigMethod, devig
from app.risk.staking import (
    StakePolicy,
    effective_kelly_multiplier,
    recommended_stake,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DEFAULT = REPO_ROOT / "data" / "ml" / "cache"
OUT_DEFAULT = REPO_ROOT / "data" / "ml" / "staking_evaluation.csv"

# Deployed premium-tier selection config (app/config.py) — NOT tuned here.
PREMIUM_DEVIG = DevigMethod.DIFFERENTIAL_MARGIN
PREMIUM_EDGE = 0.03
PREMIUM_MIN_ODDS = 1.60
MARKET_KEYS: tuple[str, ...] = ("1x2", "ou25")
LEAGUES_18 = "E0,E1,E2,E3,SC0,D1,D2,I1,I2,SP1,SP2,F1,F2,N1,B1,P1,T1,G1"

# Spent-holdout ledger (binding): anything newer than 2324 is consumed.
LAST_TRAIN_SEASON = "2324"
TRAIN_SEASONS = "1920,2021,2122,2223,2324"

RUIN_DRAWDOWN = 0.50  # "ruin" = a 50% drawdown from the running peak

# --- PRE-REGISTERED DECISION CRITERION (Track D) -----------------------------
# Written in code 2026-06-12 BEFORE the first real simulation run (binding,
# per the project's pre-registration discipline). The comparison is the
# deployed default (0.25x Kelly + 2% cap) vs each drawdown-constrained
# variant, on the block-bootstrap distributions at the CLI defaults
# (paths=10000, block_size=20, seed=20260612). Flat staking is reported as
# context only and never drives the switch decision.
#
# Recommend switching the default ONLY if a variant:
#   (A) strictly improves median growth — median log-growth per bet greater
#       than the default's; OR
#   (B) materially cuts the bad tail — P95(max drawdown) reduced by at least
#       TAIL_DD_RELATIVE_CUT (20% relative) — while costing at most
#       GROWTH_COST_TOLERANCE (10%) of median growth (when default growth is
#       positive; when it is <= 0 the variant must simply grow no slower).
# Among passing variants: prefer (A) winners by highest median growth; if
# only (B) winners exist, take the (B) winner with the highest median growth.
# Anything else => keep the deployed default. No live default changes follow
# from this script under any outcome — output is the exact .env lines only.
GROWTH_COST_TOLERANCE = 0.10
TAIL_DD_RELATIVE_CUT = 0.20

# Drawdown-constrained grid under evaluation: (max_drawdown, max_probability).
# Effective multiplier = min(0.25, lambda*) — the variant can only ever
# TIGHTEN the deployed default, never loosen it (cap math in app/risk).
DD_GRID: tuple[tuple[float, float], ...] = (
    (0.5, 0.01),
    (0.5, 0.005),
    (0.5, 0.001),
    (0.3, 0.05),
    (0.3, 0.01),
    (0.2, 0.05),
    (0.2, 0.01),
)

DEFAULT_POLICY_NAME = "kelly 0.25x + 2% cap (deployed)"
FLAT_POLICY_NAME = "flat 1u of 100 (context only)"


def assert_train_only(seasons: Sequence[str]) -> None:
    """Hard-fail on any spent-holdout season (newer than 2324).

    Staking evaluation is a strategy-adjacent decision; the 2425+2526 holdout
    stays spent and is never consulted here (binding ledger).
    """
    for season in seasons:
        if len(season) != 4 or not season.isdigit():
            raise ValueError(f"season must be 4 digits like '2324', got {season!r}")
        if int(season[:2]) > int(LAST_TRAIN_SEASON[:2]):
            raise ValueError(
                f"season {season} is in the SPENT holdout (newer than {LAST_TRAIN_SEASON}); "
                "staking evaluation runs on TRAIN seasons only"
            )


# --- premium pick stream (same construction as scripts/value_backtest.py) ----
def _f(x: object) -> float | None:
    try:
        v = float(str(x))
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


def _won_1x2(r: Mapping[str, str], i: int) -> bool | None:
    ftr = r.get("FTR")
    if ftr not in ("H", "D", "A"):
        return None
    return ftr == ("H", "D", "A")[i]


def _won_ou25(r: Mapping[str, str], i: int) -> bool | None:
    try:
        goals = int(r["FTHG"]) + int(r["FTAG"])
    except (KeyError, TypeError, ValueError):
        return None
    return (goals >= 3) if i == 0 else (goals <= 2)


# market -> (pre-match Pinnacle cols, pre-match Max cols, settle fn).
# Column tuples are parity-locked against scripts/value_backtest.py by
# tests/test_evaluate_staking.py.
MARKETS: dict[str, tuple[tuple[str, ...], tuple[str, ...], Callable[..., bool | None]]] = {
    "1x2": (("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA"), _won_1x2),
    "ou25": (("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5"), _won_ou25),
}


def _parse_date(raw: str) -> date | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class Pick:
    """One settled premium pick: model probability, taken price, outcome."""

    match_date: date
    league: str
    home: str
    away: str
    market: str
    won: bool
    odds: float
    p_fair: float  # devigged sharp fair probability (the model p for Kelly)
    edge: float


def premium_picks(
    rows: Sequence[Mapping[str, str]],
    *,
    threshold: float = PREMIUM_EDGE,
    devig_method: DevigMethod = PREMIUM_DEVIG,
    markets: tuple[str, ...] = MARKET_KEYS,
    min_odds: float = PREMIUM_MIN_ODDS,
) -> list[Pick]:
    """One pick per (match, market): the highest-edge selection >= threshold,
    chronologically sorted (deterministic tie-break on league/teams/market)."""
    picks: list[Pick] = []
    for r in rows:
        match_date = _parse_date(r.get("Date") or "")
        if match_date is None:
            continue
        for market in markets:
            ps_cols, mx_cols, won_fn = MARKETS[market]
            ps_raw = [_f(r.get(c)) for c in ps_cols]
            mx_raw = [_f(r.get(c)) for c in mx_cols]
            if any(v is None for v in ps_raw) or any(v is None for v in mx_raw):
                continue
            ps = [v for v in ps_raw if v is not None]
            mx = [v for v in mx_raw if v is not None]
            if won_fn(r, 0) is None:
                continue
            fair = devig(ps, method=devig_method)
            best: tuple[float, int] | None = None
            for i, (price, p_fair) in enumerate(zip(mx, fair, strict=True)):
                if price < min_odds:
                    continue
                edge = p_fair - 1.0 / price
                if edge >= threshold and (best is None or edge > best[0]):
                    best = (edge, i)
            if best is None:
                continue
            edge, i = best
            won = won_fn(r, i)
            if won is None:
                continue
            picks.append(
                Pick(
                    match_date=match_date,
                    league=r.get("Div") or "",
                    home=(r.get("HomeTeam") or "").strip(),
                    away=(r.get("AwayTeam") or "").strip(),
                    market=market,
                    won=won,
                    odds=mx[i],
                    p_fair=fair[i],
                    edge=edge,
                )
            )
    picks.sort(key=lambda p: (p.match_date.isoformat(), p.league, p.home, p.away, p.market))
    return picks


# --- staking policies ---------------------------------------------------------
@dataclass(frozen=True)
class PolicySpec:
    """A staking policy under evaluation; stake_policy None means flat units."""

    name: str
    stake_policy: StakePolicy | None
    dd_params: tuple[float, float] | None = None  # (max_drawdown, max_probability)


def policy_grid() -> list[PolicySpec]:
    specs = [
        PolicySpec(FLAT_POLICY_NAME, None),
        PolicySpec(DEFAULT_POLICY_NAME, StakePolicy()),
    ]
    for max_dd, beta in DD_GRID:
        specs.append(
            PolicySpec(
                name=f"ddKelly d={max_dd:.2f} beta={beta:.3f}",
                stake_policy=StakePolicy(max_drawdown=max_dd, max_drawdown_probability=beta),
                dd_params=(max_dd, beta),
            )
        )
    return specs


def per_unit_returns(picks: Sequence[Pick]) -> np.ndarray:
    """Profit per 1 unit staked: odds-1 on a win, -1 on a loss."""
    return np.array([(p.odds - 1.0) if p.won else -1.0 for p in picks], dtype=np.float64)


def stake_fractions(picks: Sequence[Pick], policy: StakePolicy) -> np.ndarray:
    """Per-pick recommended bankroll fraction via the SAME path live picks use."""
    return np.array(
        [recommended_stake(p.p_fair, p.odds, policy).final for p in picks],
        dtype=np.float64,
    )


# --- bankroll-path simulation (vectorized) -------------------------------------
@dataclass(frozen=True)
class SimResult:
    terminal: np.ndarray  # terminal bankroll as a multiple of the start
    max_drawdown: np.ndarray  # worst peak-to-trough fraction per path


def bootstrap_indices(
    n_bets: int, n_paths: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular block bootstrap: each path concatenates blocks of consecutive
    picks (wrapping at the end), truncated to the original stream length."""
    if n_bets <= 0:
        raise ValueError("need at least one bet to bootstrap")
    block = max(1, min(block_size, n_bets))
    n_blocks = -(-n_bets // block)
    starts = rng.integers(0, n_bets, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_bets
    return idx.reshape(n_paths, n_blocks * block)[:, :n_bets]


def simulate_proportional(per_bet_log_return: np.ndarray, idx: np.ndarray) -> SimResult:
    """Bankroll paths for stake-a-fraction-of-current-bankroll policies."""
    log_wealth = np.cumsum(per_bet_log_return[idx], axis=1)
    log_peak = np.maximum.accumulate(np.maximum(log_wealth, 0.0), axis=1)
    drawdown = 1.0 - np.exp(log_wealth - log_peak)
    return SimResult(terminal=np.exp(log_wealth[:, -1]), max_drawdown=drawdown.max(axis=1))


def simulate_flat(
    per_unit_return: np.ndarray,
    idx: np.ndarray,
    stake_units: float = 1.0,
    initial_units: float = 100.0,
) -> SimResult:
    """Bankroll paths for a fixed absolute stake (does not stop at ruin)."""
    wealth = initial_units + stake_units * np.cumsum(per_unit_return[idx], axis=1)
    peak = np.maximum.accumulate(np.maximum(wealth, initial_units), axis=1)
    drawdown = np.minimum((peak - wealth) / peak, 1.0)
    terminal = np.maximum(wealth[:, -1], 0.0) / initial_units
    return SimResult(terminal=terminal, max_drawdown=drawdown.max(axis=1))


@dataclass(frozen=True)
class PolicyMetrics:
    """Bootstrap + chronological metrics for one staking policy."""

    name: str
    dd_params: tuple[float, float] | None
    kelly_multiplier: float | None  # None for flat staking
    n_bets: int
    mean_stake_fraction: float
    median_terminal: float
    p5_terminal: float
    p95_terminal: float
    median_log_growth_per_bet: float
    median_max_drawdown: float
    p95_max_drawdown: float
    p_ruin: float  # Pr(max drawdown >= RUIN_DRAWDOWN)
    chrono_terminal: float
    chrono_max_drawdown: float


def summarize(
    spec: PolicySpec,
    sim: SimResult,
    chrono: SimResult,
    n_bets: int,
    mean_stake_fraction: float,
) -> PolicyMetrics:
    safe_terminal = np.maximum(sim.terminal, 1e-12)
    multiplier = (
        None if spec.stake_policy is None else effective_kelly_multiplier(spec.stake_policy)
    )
    return PolicyMetrics(
        name=spec.name,
        dd_params=spec.dd_params,
        kelly_multiplier=multiplier,
        n_bets=n_bets,
        mean_stake_fraction=mean_stake_fraction,
        median_terminal=float(np.median(sim.terminal)),
        p5_terminal=float(np.percentile(sim.terminal, 5)),
        p95_terminal=float(np.percentile(sim.terminal, 95)),
        median_log_growth_per_bet=float(np.median(np.log(safe_terminal)) / n_bets),
        median_max_drawdown=float(np.median(sim.max_drawdown)),
        p95_max_drawdown=float(np.percentile(sim.max_drawdown, 95)),
        p_ruin=float(np.mean(sim.max_drawdown >= RUIN_DRAWDOWN)),
        chrono_terminal=float(chrono.terminal[0]),
        chrono_max_drawdown=float(chrono.max_drawdown[0]),
    )


def evaluate_policies(
    picks: Sequence[Pick],
    specs: Sequence[PolicySpec],
    *,
    n_paths: int = 10_000,
    block_size: int = 20,
    seed: int = 20260612,
    flat_stake_units: float = 1.0,
    flat_initial_units: float = 100.0,
    chunk_paths: int = 2_000,
) -> list[PolicyMetrics]:
    """Block-bootstrap every policy on the SAME resampled paths (paired
    comparison), chunked to bound memory; plus one chronological pass."""
    n_bets = len(picks)
    if n_bets == 0:
        raise ValueError("no picks to evaluate")
    unit_returns = per_unit_returns(picks)

    log_returns: dict[str, np.ndarray] = {}
    mean_stakes: dict[str, float] = {}
    for spec in specs:
        if spec.stake_policy is None:
            mean_stakes[spec.name] = flat_stake_units / flat_initial_units
        else:
            fractions = stake_fractions(picks, spec.stake_policy)
            log_returns[spec.name] = np.log1p(fractions * unit_returns)
            mean_stakes[spec.name] = float(fractions.mean())

    def run(spec: PolicySpec, idx: np.ndarray) -> SimResult:
        if spec.stake_policy is None:
            return simulate_flat(unit_returns, idx, flat_stake_units, flat_initial_units)
        return simulate_proportional(log_returns[spec.name], idx)

    rng = np.random.default_rng(seed)
    terminals: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    drawdowns: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    done = 0
    while done < n_paths:
        chunk = min(chunk_paths, n_paths - done)
        idx = bootstrap_indices(n_bets, chunk, block_size, rng)
        for spec in specs:
            sim = run(spec, idx)
            terminals[spec.name].append(sim.terminal)
            drawdowns[spec.name].append(sim.max_drawdown)
        done += chunk

    chrono_idx = np.arange(n_bets, dtype=np.int64)[None, :]
    metrics: list[PolicyMetrics] = []
    for spec in specs:
        sim = SimResult(
            terminal=np.concatenate(terminals[spec.name]),
            max_drawdown=np.concatenate(drawdowns[spec.name]),
        )
        metrics.append(summarize(spec, sim, run(spec, chrono_idx), n_bets, mean_stakes[spec.name]))
    return metrics


# --- pre-registered verdict logic ----------------------------------------------
def split_metrics(metrics: Sequence[PolicyMetrics]) -> tuple[PolicyMetrics, list[PolicyMetrics]]:
    """(deployed default, drawdown-constrained variants). Flat is context only."""
    default = next(m for m in metrics if m.name == DEFAULT_POLICY_NAME)
    variants = [m for m in metrics if m.dd_params is not None]
    return default, variants


def growth_cost_acceptable(
    default_growth: float, variant_growth: float, tolerance: float = GROWTH_COST_TOLERANCE
) -> bool:
    if default_growth > 0.0:
        return variant_growth >= (1.0 - tolerance) * default_growth
    return variant_growth >= default_growth


def variant_passes(default: PolicyMetrics, variant: PolicyMetrics) -> tuple[bool, str]:
    """Apply the pre-registered criterion (module docstring + constants above)."""
    if variant.median_log_growth_per_bet > default.median_log_growth_per_bet:
        return True, "A: improves median growth"
    if default.p95_max_drawdown > 0.0:
        cut = (default.p95_max_drawdown - variant.p95_max_drawdown) / default.p95_max_drawdown
        if cut >= TAIL_DD_RELATIVE_CUT and growth_cost_acceptable(
            default.median_log_growth_per_bet, variant.median_log_growth_per_bet
        ):
            return True, f"B: cuts P95 max-drawdown {cut:.0%} within the growth budget"
    return False, "fails the pre-registered criterion"


def select_recommendation(
    default: PolicyMetrics, variants: Sequence[PolicyMetrics]
) -> tuple[PolicyMetrics | None, str]:
    """Pre-declared selection rule: (A) winners by growth, else (B) winners by
    growth, else keep the deployed default."""
    growth_winners: list[PolicyMetrics] = []
    tail_winners: list[PolicyMetrics] = []
    for variant in variants:
        passed, reason = variant_passes(default, variant)
        if not passed:
            continue
        (growth_winners if reason.startswith("A") else tail_winners).append(variant)
    if growth_winners:
        best = max(growth_winners, key=lambda m: m.median_log_growth_per_bet)
        return best, "improves median growth over the deployed default"
    if tail_winners:
        best = max(tail_winners, key=lambda m: m.median_log_growth_per_bet)
        return best, "cuts tail drawdown within the pre-registered growth budget"
    return None, "no variant passes the pre-registered criterion — keep the deployed default"


# --- IO: cached read-only loading + report -------------------------------------
async def _fetch_missing(cache_dir: Path, pairs: Sequence[tuple[str, str]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        for league, season in pairs:
            text = await fetch_season_csv(client, league, season)
            (cache_dir / f"{season}_{league}.csv").write_text(text, encoding="utf-8")
            await asyncio.sleep(0.3)


def load_rows(
    leagues: Sequence[str], seasons: Sequence[str], cache_dir: Path
) -> list[dict[str, str]]:
    missing = [
        (league, season)
        for season in seasons
        for league in leagues
        if not (cache_dir / f"{season}_{league}.csv").exists()
    ]
    if missing:
        asyncio.run(_fetch_missing(cache_dir, missing))
    rows: list[dict[str, str]] = []
    for season in seasons:
        for league in leagues:
            text = (cache_dir / f"{season}_{league}.csv").read_text(
                encoding="utf-8", errors="replace"
            )
            for raw in csv.DictReader(io.StringIO(text)):
                if raw.get("Date") and raw.get("HomeTeam"):
                    raw.setdefault("Div", league)
                    rows.append(raw)
    return rows


def format_table(metrics: Sequence[PolicyMetrics]) -> str:
    header = (
        f"{'policy':<32} {'mult':>5} {'stk%':>5} | {'medT':>7} {'P5':>7} {'P95':>8} "
        f"{'g/bet':>9} | {'medDD':>6} {'P95DD':>6} {'Pruin':>6} | {'chrT':>7} {'chDD':>6}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        mult = f"{m.kelly_multiplier:.3f}" if m.kelly_multiplier is not None else "    -"
        lines.append(
            f"{m.name:<32} {mult:>5} {m.mean_stake_fraction * 100:>4.2f} | "
            f"{m.median_terminal:>7.3f} {m.p5_terminal:>7.3f} {m.p95_terminal:>8.3f} "
            f"{m.median_log_growth_per_bet:>+9.6f} | "
            f"{m.median_max_drawdown * 100:>5.1f}% {m.p95_max_drawdown * 100:>5.1f}% "
            f"{m.p_ruin * 100:>5.2f}% | "
            f"{m.chrono_terminal:>7.3f} {m.chrono_max_drawdown * 100:>5.1f}%"
        )
    return "\n".join(lines)


def write_csv(metrics: Sequence[PolicyMetrics], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "max_drawdown",
        "max_drawdown_probability",
        "kelly_multiplier",
        "n_bets",
        "mean_stake_fraction",
        "median_terminal",
        "p5_terminal",
        "p95_terminal",
        "median_log_growth_per_bet",
        "median_max_drawdown",
        "p95_max_drawdown",
        "p_ruin",
        "chrono_terminal",
        "chrono_max_drawdown",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for m in metrics:
            dd, beta = m.dd_params if m.dd_params is not None else (None, None)
            writer.writerow(
                [
                    m.name,
                    dd,
                    beta,
                    m.kelly_multiplier,
                    m.n_bets,
                    m.mean_stake_fraction,
                    m.median_terminal,
                    m.p5_terminal,
                    m.p95_terminal,
                    m.median_log_growth_per_bet,
                    m.median_max_drawdown,
                    m.p95_max_drawdown,
                    m.p_ruin,
                    m.chrono_terminal,
                    m.chrono_max_drawdown,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", default=LEAGUES_18)
    parser.add_argument("--seasons", default=TRAIN_SEASONS, help="TRAIN seasons only (<= 2324)")
    parser.add_argument("--edge-threshold", type=float, default=PREMIUM_EDGE)
    parser.add_argument("--min-odds", type=float, default=PREMIUM_MIN_ODDS)
    parser.add_argument("--markets", default=",".join(MARKET_KEYS))
    parser.add_argument("--devig", default=PREMIUM_DEVIG.value)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    markets = tuple(x.strip() for x in args.markets.split(",") if x.strip())
    assert_train_only(seasons)

    rows = load_rows(leagues, seasons, args.cache_dir)
    picks = premium_picks(
        rows,
        threshold=args.edge_threshold,
        devig_method=DevigMethod(args.devig),
        markets=markets,
        min_odds=args.min_odds,
    )
    if not picks:
        print("No premium picks in the requested slice — nothing to evaluate.")
        return

    unit = per_unit_returns(picks)
    print(
        f"\nTRACK D — STAKING EVALUATION (recommended-stake policy; informational only)\n"
        f"premium stream: n={len(picks)} picks | {picks[0].match_date} .. "
        f"{picks[-1].match_date} | seasons {seasons}\n"
        f"selection (deployed, not tuned here): devig={args.devig}, "
        f"edge>={args.edge_threshold}, odds>={args.min_odds}, markets={markets}\n"
        f"hit {sum(p.won for p in picks) / len(picks) * 100:.1f}% | "
        f"flat ROI {unit.mean() * 100:+.2f}%/bet | mean odds "
        f"{np.mean([p.odds for p in picks]):.2f} | mean edge "
        f"{np.mean([p.edge for p in picks]):.4f}\n"
        f"bootstrap: {args.paths} circular block paths, block={args.block_size}, "
        f"seed={args.seed} | ruin = {RUIN_DRAWDOWN:.0%} drawdown from peak\n"
    )

    metrics = evaluate_policies(
        picks,
        policy_grid(),
        n_paths=args.paths,
        block_size=args.block_size,
        seed=args.seed,
    )
    print(format_table(metrics))

    default, variants = split_metrics(metrics)
    print("\nPre-registered criterion per variant (vs deployed default):")
    for variant in variants:
        passed, reason = variant_passes(default, variant)
        print(f"  {'PASS' if passed else 'fail'}  {variant.name:<28} {reason}")

    best, why = select_recommendation(default, variants)
    print("\nVERDICT (computed from the pre-registered criterion):")
    if best is None:
        print(f"  KEEP the deployed default — {why}")
    else:
        assert best.dd_params is not None
        max_dd, beta = best.dd_params
        print(f"  {best.name} — {why}")
        print("  No defaults change. To adopt this recommended-stake policy, set in .env:")
        print(f"    STAKE_MAX_DRAWDOWN={max_dd:g}")
        print(f"    STAKE_MAX_DRAWDOWN_PROBABILITY={beta:g}")

    write_csv(metrics, args.out)
    print(f"\nfull table -> {args.out}")
    print(
        "\nRecommended stakes are informational only; this platform never places "
        "bets and nothing here is betting advice or a guarantee of profit."
    )


if __name__ == "__main__":
    main()
