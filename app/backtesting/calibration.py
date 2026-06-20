"""Calibration of devigged fair-anchor probabilities vs realized outcomes.

Pure module (math/stdlib only) — DB reads live in app/storage/repositories.py
or a report script's composition root, never here.

DOCTRINE NOTE — this is a DIAGNOSTIC, not the live validator. CLV stays the
evaluation currency for the line-shopping edge (app/backtesting/clv.py); this
only checks the secondary question "when the anchor says X%, do those bets land
~X%?". It is computed over SETTLED PICKS, so it is pick-CONDITIONAL
(selection-biased toward selections where we found value): it answers "are the
fair probs on the bets I actually make well-calibrated?", NOT "is the anchor
calibrated unconditionally". A consumer must read it as a sanity check on the
fair-prob anchor, never as a profit signal.

Honesty (mirrors app/backtesting/live_evidence.py): below MIN_CALIBRATION_N
binary observations the report is `insufficient` and every point estimate is
nulled at the source — no consumer can read noise-level calibration numbers.

Metrics (all on a binary outcome y in {0,1} and predicted P(win) p):
  - log_loss  = -mean( y·ln p + (1-y)·ln(1-p) )   [strictly proper, local;
                the penaltyblog author's preferred score — pena.lt/y blog]
  - ignorance =  log_loss / ln(2)  — the same score in BITS (Good 1952), the
                units penaltyblog.metrics.ignorance reports
  - brier     =  mean( (p - y)^2 )
  - ece       =  sum_bins (n_b/N)·|mean_pred_b - observed_b|   (n-weighted)
  - mce       =  max_bins |mean_pred_b - observed_b|
  - reliability bins: equal-width [0,1] partition; each carries its own n so
    a consumer can see which probability ranges are thin.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Below this many binary observations a (sub)report is "insufficient" — point
#: estimates are nulled so noise can never read as a calibration verdict. Same
#: floor as the live-evidence CLV strata.
MIN_CALIBRATION_N = 50

#: Clamp predicted probabilities into (eps, 1-eps) before log-loss so a 0/1
#: prediction can never produce an infinite score.
_EPS = 1e-12


@dataclass(frozen=True)
class CalibrationObservation:
    """One settled, binary-outcome pick reduced to plain floats at the DB
    boundary. Non-binary settlements (push/void/quarter-line half_*) are
    excluded upstream — they have no win/lose label to calibrate against."""

    fair_prob: float  # devigged anchor P(selection wins), expected in (0, 1)
    won: bool  # True = selection won, False = lost


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_pred: float | None  # mean predicted prob in the bin (None if empty)
    observed: float | None  # observed win frequency in the bin (None if empty)


@dataclass(frozen=True)
class CalibrationReport:
    n: int  # binary observations scored
    insufficient: bool  # n < min_n -> all estimates below are None
    log_loss: float | None
    # ignorance score (Good 1952) = log_loss in BITS = log_loss / ln(2). The same
    # strictly-proper local score the penaltyblog author prefers, in the units his
    # penaltyblog.metrics.ignorance reports — exposed beside Brier so the report
    # carries both proper scores (nats + bits) and the calibration diagnostics.
    ignorance: float | None
    brier: float | None
    ece: float | None
    mce: float | None
    base_rate: float | None  # observed overall win frequency
    mean_pred: float | None  # mean predicted probability
    bins: tuple[ReliabilityBin, ...]


def _clamp01(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def calibration_report(
    observations: Sequence[CalibrationObservation],
    *,
    n_bins: int = 10,
    min_n: int = MIN_CALIBRATION_N,
) -> CalibrationReport:
    """Score a set of (fair_prob, won) observations. Empty/insufficient input
    returns a report with `insufficient=True` and nulled estimates (n is always
    reported). `n_bins` equal-width reliability bins span [0, 1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    n = len(observations)
    if n < min_n:
        return CalibrationReport(
            n=n,
            insufficient=True,
            log_loss=None,
            ignorance=None,
            brier=None,
            ece=None,
            mce=None,
            base_rate=None,
            mean_pred=None,
            bins=(),
        )

    ys = [1.0 if o.won else 0.0 for o in observations]
    ps = [_clamp01(o.fair_prob) for o in observations]

    log_loss = (
        -sum(y * math.log(p) + (1.0 - y) * math.log(1.0 - p) for y, p in zip(ys, ps, strict=True))
        / n
    )
    ignorance = log_loss / math.log(2)  # log_loss in bits — Good's ignorance score
    brier = sum((p - y) ** 2 for y, p in zip(ys, ps, strict=True)) / n
    base_rate = sum(ys) / n
    mean_pred = sum(ps) / n

    # Equal-width bins over [0, 1]; the top edge is inclusive so p == 1.0 lands
    # in the last bin rather than overflowing. Interior left edges are subject
    # to float representation (e.g. int(0.3/0.1) == 2 because 0.3/0.1 ==
    # 2.9999...), so a prob exactly on a boundary can land one bin low — benign:
    # every obs is still counted once, and ECE/MCE use each bin's OWN mean_pred
    # (not the nominal edge), so aggregate metrics are unaffected.
    width = 1.0 / n_bins
    bin_pred_sum = [0.0] * n_bins
    bin_obs_sum = [0.0] * n_bins
    bin_count = [0] * n_bins
    for y, p in zip(ys, ps, strict=True):
        idx = min(n_bins - 1, int(p / width))
        bin_pred_sum[idx] += p
        bin_obs_sum[idx] += y
        bin_count[idx] += 1

    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo = i * width
        hi = (i + 1) * width
        cnt = bin_count[i]
        if cnt == 0:
            bins.append(ReliabilityBin(lo=lo, hi=hi, n=0, mean_pred=None, observed=None))
            continue
        mp = bin_pred_sum[i] / cnt
        ob = bin_obs_sum[i] / cnt
        gap = abs(mp - ob)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lo=lo, hi=hi, n=cnt, mean_pred=mp, observed=ob))

    return CalibrationReport(
        n=n,
        insufficient=False,
        log_loss=log_loss,
        ignorance=ignorance,
        brier=brier,
        ece=ece,
        mce=mce,
        base_rate=base_rate,
        mean_pred=mean_pred,
        bins=tuple(bins),
    )
