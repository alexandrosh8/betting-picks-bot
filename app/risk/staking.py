"""Fractional-Kelly recommended staking with a transparent decomposition.

Pure module. Stakes are RECOMMENDATIONS ONLY — this platform never places
bets. Each step of the sizing path stays visible (raw Kelly -> fractional ->
cap) so picks can expose the full breakdown (risk-kelly-engineer mandate).
"""

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly bankroll fraction, clipped to >= 0 (never recommends laying).

    kelly = ((d - 1) * p - (1 - p)) / (d - 1)
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {decimal_odds}")
    b = decimal_odds - 1.0
    edge_per_unit = (b * probability - (1.0 - probability)) / b
    return max(edge_per_unit, 0.0)


@dataclass(frozen=True)
class StakePolicy:
    """Sizing policy; constructed from Settings at the composition root."""

    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.02
    # OPTIONAL drawdown-constrained Kelly (default OFF — both None keeps the
    # plain 0.25x/2% path bit-for-bit). When BOTH are set, the Kelly
    # multiplier becomes min(fractional_kelly, lambda*) where lambda* is the
    # largest multiplier satisfying Pr(drawdown > max_drawdown) <=
    # max_drawdown_probability under the continuous-Kelly approximation —
    # the single-bet closed form of the Busseti-Boyd 2016 risk constraint
    # (arXiv:1603.06183; kelly-bankroll skill). Staking can NEVER raise
    # per-bet yield — this knob shapes growth/drawdown only. Enabling it is
    # a phase-6 bankroll decision; evidence requirements live in
    # docs/backtesting/value-findings.md and the next ADR.
    max_drawdown: float | None = None
    max_drawdown_probability: float | None = None


def drawdown_constrained_multiplier(max_drawdown: float, max_probability: float) -> float:
    """Largest Kelly multiplier lambda with Pr(drawdown > d) <= beta.

    Continuous-approximation drawdown law for betting lambda * f_full
    (Thorp; MacLean-Thorp-Ziemba "Good and Bad Properties"):

        Pr(wealth ever dips below (1 - d) of start) = (1 - d)^(2/lambda - 1)

    Solving (1-d)^(2/lambda - 1) <= beta for the largest lambda gives
    lambda* = 2 / (1 + ln(beta)/ln(1-d)), capped at 1.0 (full Kelly).
    Sanity anchors: d=0.5, beta=0.5 -> 1.0 (full Kelly has ~50% chance of a
    50% drawdown); d=0.5, beta=0.5**7 -> 0.25 (quarter Kelly).
    """
    if not 0.0 < max_drawdown < 1.0:
        raise ValueError(f"max_drawdown must be in (0, 1), got {max_drawdown}")
    if not 0.0 < max_probability < 1.0:
        raise ValueError(f"max_drawdown_probability must be in (0, 1), got {max_probability}")
    floor = 1.0 - max_drawdown
    lam = 2.0 / (1.0 + math.log(max_probability) / math.log(floor))
    return min(lam, 1.0)


def effective_kelly_multiplier(policy: StakePolicy) -> float:
    """The Kelly multiplier actually applied: the configured fraction, tightened
    by the drawdown constraint when (and only when) both knobs are set."""
    if policy.max_drawdown is None or policy.max_drawdown_probability is None:
        return policy.fractional_kelly
    return min(
        policy.fractional_kelly,
        drawdown_constrained_multiplier(policy.max_drawdown, policy.max_drawdown_probability),
    )


@dataclass(frozen=True)
class StakeBreakdown:
    """Transparent sizing path: raw -> fractional -> capped final fraction."""

    raw_kelly: float
    fractional: float
    capped: bool
    final: float


def recommended_stake(
    probability: float,
    decimal_odds: float,
    policy: StakePolicy,
) -> StakeBreakdown:
    raw = kelly_fraction(probability, decimal_odds)
    # effective_kelly_multiplier == policy.fractional_kelly unless the
    # optional drawdown constraint (OFF by default) tightens it.
    fractional = raw * effective_kelly_multiplier(policy)
    capped = fractional > policy.max_stake_fraction
    final = min(fractional, policy.max_stake_fraction)
    return StakeBreakdown(raw_kelly=raw, fractional=fractional, capped=capped, final=final)


def stake_amount(fraction: float, bankroll: Decimal) -> Decimal:
    """Money amount for a bankroll fraction, quantized to cents."""
    if fraction < 0.0:
        raise ValueError(f"stake fraction must be >= 0, got {fraction}")
    return (Decimal(str(fraction)) * bankroll).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
