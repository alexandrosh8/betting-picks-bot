"""Stratified live-evidence report (app/backtesting/live_evidence.py).

Pure-module tests on synthetic settled-pick rows: score buckets around q*,
tier split, feature-detected anchor dimension, and the honesty contract —
every stratum carries n, sub-min_n strata are flagged insufficient.
"""

import math

import pytest

from app.backtesting.live_evidence import (
    MIN_STRATUM_N,
    SettledPickRow,
    live_evidence_report,
)


def row(
    tier: str = "premium",
    score: float | None = None,
    clv: float | None = 0.02,
    beat: bool | None = True,
    stake: float = 10.0,
    pnl: float | None = 1.0,
    anchor: str | None = None,
) -> SettledPickRow:
    return SettledPickRow(
        tier=tier,
        value_filter_score=score,
        clv_log=clv,
        beat_close=beat,
        stake=stake,
        pnl=pnl,
        anchor_type=anchor,
    )


def test_score_buckets_split_on_q_star_inclusive() -> None:
    rows = [
        row(score=0.80),  # >= q*
        row(score=0.725),  # exactly q* -> >= bucket (gate parity: >= keeps)
        row(score=0.70),  # < q*
        row(score=None),  # unscored
    ]
    report = live_evidence_report(rows, ml_threshold=0.725)
    assert report["q_star"] == 0.725
    assert report["by_score"]["score_ge_q"]["n"] == 2
    assert report["by_score"]["score_lt_q"]["n"] == 1
    assert report["by_score"]["unscored"]["n"] == 1


def test_no_threshold_means_one_scored_bucket() -> None:
    report = live_evidence_report([row(score=0.9), row(score=0.1)], ml_threshold=None)
    assert report["q_star"] is None
    assert set(report["by_score"]) == {"scored"}
    assert report["by_score"]["scored"]["n"] == 2


def test_tier_split_and_clv_roi_math() -> None:
    rows = [
        row(tier="premium", clv=0.10, beat=True, stake=10.0, pnl=5.0),
        row(tier="premium", clv=-0.02, beat=False, stake=30.0, pnl=-10.0),
        row(tier="volume", clv=0.01, beat=True, stake=5.0, pnl=0.5),
    ]
    report = live_evidence_report(rows, ml_threshold=0.725, min_n=1)
    premium = report["by_tier"]["premium"]
    assert premium["n"] == 2
    assert premium["n_clv"] == 2
    assert premium["mean_clv_log"] == pytest.approx((0.10 - 0.02) / 2)
    # stake-weighted: (10*0.10 + 30*-0.02) / 40
    assert premium["stake_weighted_clv_log"] == pytest.approx(0.4 / 40.0)
    assert premium["beat_close_rate"] == pytest.approx(0.5)
    assert premium["roi"] == pytest.approx(-5.0 / 40.0)
    assert premium["sufficient"] is True
    assert report["by_tier"]["volume"]["n"] == 1


def test_unrevalidated_rows_stay_in_n_but_out_of_estimates() -> None:
    rows = [
        row(clv=None, beat=None, pnl=None),  # settled but never revalidated
        row(clv=0.03, beat=True, pnl=2.0),
    ]
    report = live_evidence_report(rows, ml_threshold=None, min_n=1)
    stats = report["by_tier"]["premium"]
    assert stats["n"] == 2  # honest n: every settled row counts
    assert stats["n_clv"] == 1  # ...but only CLV rows enter CLV estimates
    assert stats["n_roi"] == 1
    assert stats["mean_clv_log"] == pytest.approx(0.03)


def test_insufficient_stratum_is_flagged_below_min_n() -> None:
    # 49 CLV rows < default 50 -> insufficient; the 50th flips it.
    rows = [row(clv=0.01) for _ in range(MIN_STRATUM_N - 1)]
    report = live_evidence_report(rows, ml_threshold=None)
    assert report["min_n"] == MIN_STRATUM_N
    assert report["by_tier"]["premium"]["sufficient"] is False
    report = live_evidence_report(rows + [row(clv=0.01)], ml_threshold=None)
    assert report["by_tier"]["premium"]["sufficient"] is True


def test_insufficient_stratum_nulls_estimates_at_source() -> None:
    """Validator-confirmed hardening: an insufficient stratum must carry NO
    point estimates in the payload — the dashboard honors the flag, but any
    other consumer of GET /performance would otherwise read noise-level
    numbers. Denominators and the flag survive; estimates are nulled."""
    rows = [row(clv=0.05, beat=True, pnl=3.0) for _ in range(MIN_STRATUM_N - 1)]
    stats = live_evidence_report(rows, ml_threshold=None)["by_tier"]["premium"]
    assert stats["sufficient"] is False
    assert stats["mean_clv_log"] is None
    assert stats["stake_weighted_clv_log"] is None
    assert stats["beat_close_rate"] is None
    assert stats["roi"] is None
    # honest denominators stay visible for the insufficient-state render
    assert stats["n"] == MIN_STRATUM_N - 1
    assert stats["n_clv"] == MIN_STRATUM_N - 1
    assert stats["n_roi"] == MIN_STRATUM_N - 1
    # ...and the same rows above the floor keep their estimates
    full = live_evidence_report(rows + [row(clv=0.05, pnl=3.0)], ml_threshold=None)
    assert full["by_tier"]["premium"]["sufficient"] is True
    assert full["by_tier"]["premium"]["mean_clv_log"] == pytest.approx(0.05)


def test_anchor_dimension_is_feature_detected() -> None:
    # No row carries anchor_type (column not landed) -> dimension is None,
    # distinguishable from an empty grouping; once values exist, it appears.
    without = live_evidence_report([row(), row()], ml_threshold=None)
    assert without["by_anchor"] is None
    with_anchor = live_evidence_report(
        [row(anchor="sharp"), row(anchor="consensus"), row()], ml_threshold=None, min_n=1
    )
    assert with_anchor["by_anchor"] is not None
    assert with_anchor["by_anchor"]["sharp"]["n"] == 1
    assert with_anchor["by_anchor"]["consensus"]["n"] == 1


def test_empty_rows_yield_empty_but_valid_report() -> None:
    report = live_evidence_report([], ml_threshold=0.725)
    assert report["n_settled"] == 0
    assert report["by_score"] == {}
    assert report["by_tier"] == {}
    assert report["by_anchor"] is None


def test_non_finite_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        live_evidence_report([], ml_threshold=math.nan)
