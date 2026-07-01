"""CLV statistical-significance pure helpers (app/backtesting/clv.py).

A positive point estimate of CLV is worthless without a significance test: the
whole strategy's proof rests on CLV being statistically > 0, not small-sample
noise. These helpers compute a one-sample t-test of mean clv_log vs 0 (with a
t-based 95% CI) and a Wilson 95% CI on the beat-close rate. Pure math: numpy /
scipy / stdlib only, no env/DB/HTTP.
"""

import math

import pytest

from app.backtesting.clv import mean_significance, wilson_interval


def test_clearly_positive_large_n_is_significant_ci_excludes_zero() -> None:
    # A tight, clearly-positive large-n CLV series: the mean is ~0.10 with small
    # spread, so the 95% CI sits entirely above 0 and the flag reads significant.
    rng = [0.10 + (0.02 if i % 2 else -0.02) for i in range(200)]  # mean 0.10, sd ~0.02
    sig = mean_significance(rng)
    assert sig is not None
    assert sig.n == 200
    assert sig.mean == pytest.approx(0.10, abs=1e-9)
    assert sig.tstat > 0
    assert sig.ci_low > 0  # CI excludes zero on the positive side
    assert sig.ci_high > sig.ci_low
    assert sig.significant is True


def test_tiny_n_noisy_series_is_not_significant() -> None:
    # n=3 with real spread: the t-CI is wide enough to straddle 0 -> NOT
    # significant. This is the honest live-state output (trusted sharp n ~3).
    sig = mean_significance([0.30, -0.10, 0.05])
    assert sig is not None
    assert sig.n == 3
    assert sig.ci_low < 0 < sig.ci_high  # CI straddles zero
    assert sig.significant is False


def test_single_observation_returns_none() -> None:
    # n=1 has no sample variance -> a t-test is undefined; return None, not a crash.
    assert mean_significance([0.05]) is None


def test_empty_series_returns_none() -> None:
    assert mean_significance([]) is None


def test_zero_variance_positive_series_collapses_ci_to_mean() -> None:
    # Degenerate: every clv_log identical and > 0. SE is 0, so the CI collapses to
    # the mean with no divide-by-zero crash — but a zero-variance sample carries NO
    # dispersion evidence, so it is NOT significant (a stratum of identical values,
    # e.g. n small, must not read as a proven edge).
    sig = mean_significance([0.05, 0.05, 0.05, 0.05])
    assert sig is not None
    assert sig.mean == pytest.approx(0.05)
    assert sig.ci_low == pytest.approx(0.05)
    assert sig.ci_high == pytest.approx(0.05)
    assert math.isinf(sig.tstat) and sig.tstat > 0
    assert sig.significant is False


def test_zero_variance_negative_series_not_significant() -> None:
    sig = mean_significance([-0.05, -0.05, -0.05])
    assert sig is not None
    assert sig.significant is False
    assert math.isinf(sig.tstat) and sig.tstat < 0


def test_alpha_widens_ci_when_smaller() -> None:
    # A smaller alpha (99% CI) is wider than the default 95% CI on the same series.
    series = [0.04 + (0.03 if i % 2 else -0.03) for i in range(40)]
    ci95 = mean_significance(series, alpha=0.05)
    ci99 = mean_significance(series, alpha=0.01)
    assert ci95 is not None and ci99 is not None
    assert ci99.ci_low < ci95.ci_low
    assert ci99.ci_high > ci95.ci_high
    assert ci95.alpha == 0.05
    assert ci99.alpha == 0.01


def test_wilson_interval_bounds_in_unit_and_ordered() -> None:
    ci = wilson_interval(30, 50)  # 60% beat-close
    assert ci is not None
    low, high = ci
    assert 0.0 <= low <= high <= 1.0
    assert low < 0.60 < high  # Wilson interval brackets the point estimate


def test_wilson_lower_bound_above_half_for_strong_large_sample() -> None:
    # 70/100 beat-close: lower Wilson bound clears 0.5 -> beating the close more
    # than half the time is statistically real at this sample size.
    ci = wilson_interval(70, 100)
    assert ci is not None
    low, high = ci
    assert low > 0.5
    assert high <= 1.0


def test_wilson_lower_bound_below_half_for_tiny_sample() -> None:
    # 2/3 beat-close looks like 67% but the Wilson lower bound is well under 0.5.
    ci = wilson_interval(2, 3)
    assert ci is not None
    assert ci[0] < 0.5


def test_wilson_extremes_clamped_to_unit_interval() -> None:
    ci0 = wilson_interval(0, 10)
    assert ci0 is not None
    low0, high0 = ci0
    assert low0 == 0.0  # clamped, never negative
    assert 0.0 < high0 < 1.0
    ci_all = wilson_interval(10, 10)
    assert ci_all is not None
    low_all, high_all = ci_all
    assert high_all == 1.0  # clamped, never above 1
    assert 0.0 < low_all < 1.0


def test_wilson_zero_n_returns_none() -> None:
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize("succ,n", [(-1, 5), (6, 5)])
def test_wilson_rejects_out_of_range_successes(succ: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(succ, n)
