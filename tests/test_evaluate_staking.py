"""Track D staking evaluation: synthetic streams, deterministic seeds.

Loads scripts/ml/evaluate_staking.py by path (scripts/ is not a package) and
exercises the pick-stream builder, the block-bootstrap bankroll simulation,
and the PRE-REGISTERED verdict criterion — no network, no real data. The
premium-stream construction is parity-locked against scripts/value_backtest.py.
"""

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.risk.staking import StakePolicy, drawdown_constrained_multiplier, recommended_stake

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


es: Any = _load(_SCRIPTS / "ml" / "evaluate_staking.py", "evaluate_staking")
vb: Any = _load(_SCRIPTS / "value_backtest.py", "value_backtest")

SEED = 20260612


def _pick(
    won: bool,
    odds: float = 2.0,
    p_fair: float = 0.55,
    match_date: date = date(2023, 1, 1),
) -> Any:
    return es.Pick(
        match_date=match_date,
        league="E0",
        home="H",
        away="A",
        market="1x2",
        won=won,
        odds=odds,
        p_fair=p_fair,
        edge=p_fair - 1.0 / odds,
    )


# ---------------------------------------------------------------------------
# Premium stream parity + construction
# ---------------------------------------------------------------------------
def test_markets_parity_locked_against_value_backtest() -> None:
    assert set(es.MARKETS) == set(vb.MARKETS)
    sample_rows = [
        {"FTR": "H", "FTHG": "2", "FTAG": "1"},
        {"FTR": "D", "FTHG": "1", "FTAG": "1"},
        {"FTR": "A", "FTHG": "0", "FTAG": "3"},
        {"FTR": "H", "FTHG": "3", "FTAG": "1"},
        {"FTHG": "", "FTAG": ""},
    ]
    for market, (ps_cols, mx_cols, won_fn) in es.MARKETS.items():
        vb_ps, vb_mx, _vb_psc, _vb_mxc, vb_won = vb.MARKETS[market]
        assert ps_cols == vb_ps  # same pre-match sharp columns
        assert mx_cols == vb_mx  # same best-price columns
        for row in sample_rows:
            for i in range(len(ps_cols)):
                assert won_fn(row, i) == vb_won(row, i)


def test_premium_picks_extracts_known_1x2_row() -> None:
    # multiplicative devig of (1.8, 3.6, 3.6): fair = (0.5, 0.25, 0.25);
    # home edge = 0.5 - 1/2.2 = 0.0454.. >= 0.03; draw/away edges negative.
    row = {
        "Div": "E0",
        "Date": "02/01/2023",
        "HomeTeam": "H",
        "AwayTeam": "A",
        "FTR": "H",
        "PSH": "1.8",
        "PSD": "3.6",
        "PSA": "3.6",
        "MaxH": "2.2",
        "MaxD": "3.6",
        "MaxA": "3.6",
    }
    from app.probabilities.devig import DevigMethod

    picks = es.premium_picks(
        [row],
        threshold=0.03,
        devig_method=DevigMethod.MULTIPLICATIVE,
        markets=("1x2",),
        min_odds=1.6,
    )
    assert len(picks) == 1  # one bet per (match, market) — highest edge only
    pick = picks[0]
    assert pick.won is True
    assert pick.odds == pytest.approx(2.2)
    assert pick.p_fair == pytest.approx(0.5)
    assert pick.edge == pytest.approx(0.5 - 1.0 / 2.2)
    assert pick.match_date == date(2023, 1, 2)


def test_premium_picks_settles_ou25_and_respects_min_odds() -> None:
    from app.probabilities.devig import DevigMethod

    row = {
        "Div": "D1",
        "Date": "05/03/2022",
        "HomeTeam": "H",
        "AwayTeam": "A",
        "FTR": "H",
        "FTHG": "3",
        "FTAG": "1",
        "P>2.5": "1.8",
        "P<2.5": "2.2",
        "Max>2.5": "2.0",
        "Max<2.5": "2.2",
    }
    picks = es.premium_picks(
        [row],
        threshold=0.03,
        devig_method=DevigMethod.MULTIPLICATIVE,
        markets=("ou25",),
        min_odds=1.6,
    )
    # fair(over) = (1/1.8)/(1/1.8 + 1/2.2) = 0.55; edge = 0.55 - 0.5 = 0.05
    assert len(picks) == 1
    assert picks[0].won is True  # 4 goals -> over 2.5 wins
    assert picks[0].p_fair == pytest.approx(0.55)
    assert picks[0].edge == pytest.approx(0.05)
    # odds floor excludes the same selection
    assert (
        es.premium_picks(
            [row],
            threshold=0.03,
            devig_method=DevigMethod.MULTIPLICATIVE,
            markets=("ou25",),
            min_odds=2.1,
        )
        == []
    )


def test_premium_picks_sorted_chronologically() -> None:
    from app.probabilities.devig import DevigMethod

    def row(d: str) -> dict[str, str]:
        return {
            "Div": "E0",
            "Date": d,
            "HomeTeam": "H",
            "AwayTeam": "A",
            "FTR": "H",
            "PSH": "1.8",
            "PSD": "3.6",
            "PSA": "3.6",
            "MaxH": "2.2",
            "MaxD": "3.6",
            "MaxA": "3.6",
        }

    picks = es.premium_picks(
        [row("10/02/2023"), row("01/02/2023")],
        threshold=0.03,
        devig_method=DevigMethod.MULTIPLICATIVE,
        markets=("1x2",),
        min_odds=1.6,
    )
    assert [p.match_date for p in picks] == [date(2023, 2, 1), date(2023, 2, 10)]


# ---------------------------------------------------------------------------
# Spent-holdout guard
# ---------------------------------------------------------------------------
def test_train_seasons_pass_the_guard() -> None:
    es.assert_train_only(["1920", "2021", "2122", "2223", "2324"])


@pytest.mark.parametrize("season", ["2425", "2526", "2627"])
def test_spent_or_future_seasons_hard_fail(season: str) -> None:
    with pytest.raises(ValueError, match="SPENT"):
        es.assert_train_only([season])


def test_malformed_season_hard_fails() -> None:
    with pytest.raises(ValueError, match="4 digits"):
        es.assert_train_only(["24/25"])


# ---------------------------------------------------------------------------
# Stakes route through app.risk.staking (the SAME path live picks use)
# ---------------------------------------------------------------------------
def test_stake_fractions_match_recommended_stake() -> None:
    picks = [_pick(won=True, odds=2.0, p_fair=0.55), _pick(won=False, odds=3.0, p_fair=0.40)]
    policy = StakePolicy()
    fractions = es.stake_fractions(picks, policy)
    expected = [recommended_stake(p.p_fair, p.odds, policy).final for p in picks]
    assert fractions.tolist() == pytest.approx(expected)
    assert fractions[0] == pytest.approx(0.02)  # quarter Kelly 0.025 capped at 2%


def test_grid_multipliers_only_tighten_the_default() -> None:
    from app.risk.staking import effective_kelly_multiplier

    for spec in es.policy_grid():
        if spec.dd_params is None:
            continue
        max_dd, beta = spec.dd_params
        expected = min(0.25, drawdown_constrained_multiplier(max_dd, beta))
        assert effective_kelly_multiplier(spec.stake_policy) == pytest.approx(expected)
        assert effective_kelly_multiplier(spec.stake_policy) <= 0.25


def test_per_unit_returns_exact() -> None:
    picks = [_pick(won=True, odds=2.5), _pick(won=False, odds=2.5)]
    assert es.per_unit_returns(picks).tolist() == pytest.approx([1.5, -1.0])


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_indices_shape_and_block_contiguity() -> None:
    n_bets, n_paths, block = 10, 7, 3
    idx = es.bootstrap_indices(n_bets, n_paths, block, np.random.default_rng(SEED))
    assert idx.shape == (n_paths, n_bets)
    assert idx.min() >= 0 and idx.max() < n_bets
    # within each block, successive indices are consecutive mod n (circular)
    steps = (idx[:, 1:] - idx[:, :-1]) % n_bets
    for j in range(n_bets - 1):
        if (j + 1) % block != 0:  # not a block boundary
            assert np.all(steps[:, j] == 1)


def test_bootstrap_indices_deterministic_per_seed() -> None:
    a = es.bootstrap_indices(50, 20, 10, np.random.default_rng(SEED))
    b = es.bootstrap_indices(50, 20, 10, np.random.default_rng(SEED))
    c = es.bootstrap_indices(50, 20, 10, np.random.default_rng(SEED + 1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Bankroll-path arithmetic (exact tiny streams)
# ---------------------------------------------------------------------------
def test_simulate_proportional_exact_two_bet_path() -> None:
    # f=0.02 at odds 2.0: win then lose -> 1.02 * 0.98 = 0.9996;
    # max drawdown = 1 - 0.9996/1.02 (peak after the win).
    log_r = np.log1p(0.02 * np.array([1.0, -1.0]))
    sim = es.simulate_proportional(log_r, np.array([[0, 1]]))
    assert sim.terminal[0] == pytest.approx(1.02 * 0.98)
    assert sim.max_drawdown[0] == pytest.approx(1.0 - (1.02 * 0.98) / 1.02)


def test_simulate_proportional_drawdown_measured_from_initial_peak() -> None:
    # straight losses: peak stays at the start, drawdown = 1 - 0.98^k
    log_r = np.full(40, np.log(0.98))
    sim = es.simulate_proportional(log_r, np.arange(40)[None, :])
    assert sim.max_drawdown[0] == pytest.approx(1.0 - 0.98**40)
    assert sim.max_drawdown[0] >= es.RUIN_DRAWDOWN  # 35+ straight 2% losses = ruin


def test_simulate_flat_exact_and_ruin() -> None:
    # 2 units start, 1u stakes, two straight losses -> wealth 1 then 0.
    sim = es.simulate_flat(
        np.array([-1.0, -1.0]), np.array([[0, 1]]), stake_units=1.0, initial_units=2.0
    )
    assert sim.terminal[0] == pytest.approx(0.0)
    assert sim.max_drawdown[0] == pytest.approx(1.0)
    assert sim.max_drawdown[0] >= es.RUIN_DRAWDOWN


def test_simulate_flat_win_then_loss_drawdown_from_peak() -> None:
    # 100u start, 1u stake at odds 3.0: +2 then -1 -> peak 102, trough 101.
    sim = es.simulate_flat(
        np.array([2.0, -1.0]), np.array([[0, 1]]), stake_units=1.0, initial_units=100.0
    )
    assert sim.terminal[0] == pytest.approx(1.01)
    assert sim.max_drawdown[0] == pytest.approx(1.0 / 102.0)


# ---------------------------------------------------------------------------
# End-to-end evaluation: determinism + structure
# ---------------------------------------------------------------------------
def _synthetic_stream(n: int = 60) -> list[Any]:
    # deterministic alternating stream, dated chronologically
    return [
        _pick(
            won=(i % 3 != 0),  # 2/3 hit rate at odds 2.0 -> positive stream
            odds=2.0,
            p_fair=0.55,
            match_date=date(2023, 1, 1 + i % 28),
        )
        for i in range(n)
    ]


def test_evaluate_policies_deterministic_for_seed() -> None:
    picks = _synthetic_stream()
    kwargs = {"n_paths": 200, "block_size": 5, "seed": SEED, "chunk_paths": 64}
    first = es.evaluate_policies(picks, es.policy_grid(), **kwargs)
    second = es.evaluate_policies(picks, es.policy_grid(), **kwargs)
    assert first == second  # frozen dataclasses of floats -> exact equality


def test_evaluate_policies_emits_all_policies_with_chrono_sanity() -> None:
    picks = _synthetic_stream()
    metrics = es.evaluate_policies(picks, es.policy_grid(), n_paths=100, block_size=5, seed=SEED)
    assert len(metrics) == 2 + len(es.DD_GRID)
    default, variants = es.split_metrics(metrics)
    assert default.name == es.DEFAULT_POLICY_NAME
    assert len(variants) == len(es.DD_GRID)
    flat = next(m for m in metrics if m.name == es.FLAT_POLICY_NAME)
    assert flat not in variants  # flat staking never drives the switch decision
    for m in metrics:
        assert m.n_bets == len(picks)
        assert 0.0 <= m.p_ruin <= 1.0
        assert m.chrono_terminal > 0.0
        assert 0.0 <= m.chrono_max_drawdown <= 1.0


def test_tighter_multiplier_means_smaller_drawdown_on_same_paths() -> None:
    # paired paths: a strictly smaller Kelly multiplier can never produce a
    # LARGER max drawdown than the default on the same resampled stream.
    picks = _synthetic_stream()
    metrics = es.evaluate_policies(picks, es.policy_grid(), n_paths=200, block_size=5, seed=SEED)
    default, variants = es.split_metrics(metrics)
    for variant in variants:
        assert variant.kelly_multiplier <= default.kelly_multiplier
        assert variant.median_max_drawdown <= default.median_max_drawdown + 1e-12
        assert variant.p95_max_drawdown <= default.p95_max_drawdown + 1e-12


# ---------------------------------------------------------------------------
# PRE-REGISTERED verdict criterion — every branch
# ---------------------------------------------------------------------------
def _metrics(name: str, growth: float, p95_dd: float, dd_params: Any = None) -> Any:
    return es.PolicyMetrics(
        name=name,
        dd_params=dd_params,
        kelly_multiplier=0.25,
        n_bets=100,
        mean_stake_fraction=0.015,
        median_terminal=float(np.exp(growth * 100)),
        p5_terminal=0.8,
        p95_terminal=2.0,
        median_log_growth_per_bet=growth,
        median_max_drawdown=p95_dd / 2,
        p95_max_drawdown=p95_dd,
        p_ruin=0.0,
        chrono_terminal=1.0,
        chrono_max_drawdown=p95_dd / 2,
    )


DEFAULT_M = _metrics(es.DEFAULT_POLICY_NAME, growth=0.0010, p95_dd=0.30)


def test_criterion_a_growth_improvement_passes() -> None:
    variant = _metrics("v", growth=0.0011, p95_dd=0.30, dd_params=(0.5, 0.01))
    passed, reason = es.variant_passes(DEFAULT_M, variant)
    assert passed and reason.startswith("A")


def test_criterion_b_tail_cut_within_growth_budget_passes() -> None:
    # 33% P95-drawdown cut at a 5% growth cost (budget: >=20% cut, <=10% cost)
    variant = _metrics("v", growth=0.00095, p95_dd=0.20, dd_params=(0.3, 0.05))
    passed, reason = es.variant_passes(DEFAULT_M, variant)
    assert passed and reason.startswith("B")


def test_criterion_rejects_tail_cut_costing_more_than_ten_percent_growth() -> None:
    variant = _metrics("v", growth=0.00085, p95_dd=0.20, dd_params=(0.2, 0.01))
    passed, _ = es.variant_passes(DEFAULT_M, variant)
    assert not passed


def test_criterion_rejects_immaterial_tail_cut() -> None:
    variant = _metrics("v", growth=0.00095, p95_dd=0.27, dd_params=(0.5, 0.005))
    passed, _ = es.variant_passes(DEFAULT_M, variant)  # only a 10% cut
    assert not passed


def test_criterion_with_nonpositive_default_growth_requires_no_worse_growth() -> None:
    shrinking_default = _metrics(es.DEFAULT_POLICY_NAME, growth=-0.0010, p95_dd=0.30)
    same_growth = _metrics("v", growth=-0.0010, p95_dd=0.20, dd_params=(0.3, 0.05))
    worse_growth = _metrics("w", growth=-0.0011, p95_dd=0.20, dd_params=(0.2, 0.05))
    assert es.variant_passes(shrinking_default, same_growth)[0]
    assert not es.variant_passes(shrinking_default, worse_growth)[0]


def test_selection_prefers_growth_winner_over_tail_winner() -> None:
    growth_winner = _metrics("g", growth=0.0012, p95_dd=0.30, dd_params=(0.5, 0.01))
    tail_winner = _metrics("t", growth=0.00095, p95_dd=0.20, dd_params=(0.3, 0.05))
    best, why = es.select_recommendation(DEFAULT_M, [tail_winner, growth_winner])
    assert best is growth_winner
    assert "median growth" in why


def test_selection_keeps_default_when_nothing_passes() -> None:
    losers = [
        _metrics("x", growth=0.0008, p95_dd=0.29, dd_params=(0.5, 0.005)),
        _metrics("y", growth=0.0007, p95_dd=0.28, dd_params=(0.2, 0.01)),
    ]
    best, why = es.select_recommendation(DEFAULT_M, losers)
    assert best is None
    assert "keep the deployed default" in why
