"""Line-movement / steam-awareness gate for the sharp-vs-soft value finder.

The value finder (app/edge/value.py) prices a candidate from a SINGLE snapshot:
the best soft price beats the sharp anchor's fair value by `min_edge`. That is a
point-in-time read of a moving market, and the dominant soft-book false positive
is a movement artefact, not a real edge:

  * soft-toward-anchor — the soft book has ALREADY moved toward the anchor (the
    gap is closing, the edge is correcting/evaporating). The residual edge on the
    latest snapshot is stale; betting it is chasing a price that is leaving.
  * stale-anchor — the sharp anchor's last observation is older than a freshness
    window (a feed gap / an archived close re-injected as "current"). The edge is
    measured against a price that may no longer reflect the market — a phantom.
  * soft-steamed-away (informational) — the soft price is moving sharply AWAY
    from the anchor. It may be reacting to real information (lineup/injury); a
    flag, not a demotion, by default.

This module is PURE (stdlib only — the pure-math boundary, CLAUDE.md): no env, no
DB, no HTTP, no logging side effects. All policy arrives as a frozen ``SteamPolicy``
built from Settings at the composition root. Every signal works in IMPLIED-PROB
space (devig is monotonic, so raw quoted prices are a faithful proxy for the
convergence question, and movement is conventionally measured on quoted prices).

Pick-time discipline / NO LEAKAGE: only observations with ``captured_at <= now``
inside the lookback window are ever consulted — a future-stamped row can never
influence a verdict. The caller (app/pipeline.py) supplies trajectories assembled
from ``odds_snapshots`` history (change-only per-book time series) plus the current
cycle's snapshots; the gate reads them, it never fetches.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

# (captured_at, decimal_odds) — one observation of one book's price.
TrajectoryPoint = tuple[datetime, float]


@dataclass(frozen=True)
class SteamPolicy:
    """Movement-gate thresholds. Immutable; built from Settings at the root.

    The default ``enabled=False`` runs the gate in SHADOW: the verdict is
    computed and surfaced but the pick tier is unchanged, so its effect on real
    picks can be measured before it ever enforces. ``enabled=True`` makes a
    tripped verdict DEMOTE a premium candidate to the volume (shadow) tier —
    persisted + CLV-tracked, never alerted — exactly like the other built-but-off
    premium gates (it never silently drops a pick).
    """

    enabled: bool = False
    # Trajectory window: observations older than this (or newer than now) are
    # ignored. Bounds the per-book history the gate consults.
    lookback_seconds: float = 21600.0  # 6h
    # Minimum in-window observations of the FILL book required before any
    # movement judgement — below this the gate cannot judge and stays inert.
    min_points: int = 2
    # Convergence trip: the fraction of the ORIGINAL fill-vs-anchor gap that must
    # already have closed for the edge to be treated as correcting/evaporating.
    soft_toward_anchor_close_frac: float = 0.5
    # Below this opening gap (prob units) there was no meaningful edge to close,
    # so the convergence signal is suppressed (avoids dividing tiny gaps).
    min_initial_gap: float = 0.01
    # Stale-anchor trip: the anchor's most-recent in-window observation must be
    # no older than this, else the edge is measured against an unverifiable price.
    anchor_staleness_seconds: float = 7200.0  # 2h
    # Soft-steamed-away FLAG: the fill implied prob dropping (odds lengthening)
    # by at least this much over the window is a sharp move away from the anchor.
    soft_steam_away_delta: float = 0.04
    # Whether a soft-steamed-away flag also TRIPS the gate (default: flag only).
    demote_on_soft_steam: bool = False


@dataclass(frozen=True)
class SteamVerdict:
    """Per-candidate movement verdict. ``tripped`` is the gate decision; the
    component flags and the numeric detail fields are surfaced for logging /
    shadow measurement."""

    soft_toward_anchor: bool
    stale_anchor: bool
    soft_steamed: bool
    tripped: bool
    reasons: tuple[str, ...]
    closed_fraction: float | None
    anchor_age_seconds: float | None
    soft_move: float | None


def implied_prob(odds: float) -> float:
    """Implied (vig-inclusive) probability of a decimal price.

    Raises ``ValueError`` on a degenerate price (<= 1.0) — garbage must never
    propagate into a movement judgement (boundary discipline)."""
    if odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {odds}")
    return 1.0 / odds


def _window(
    trajectory: Sequence[TrajectoryPoint], now: datetime, lookback_seconds: float
) -> list[TrajectoryPoint]:
    """In-window, in-the-past, sane-price points sorted oldest-first.

    The ``captured_at <= now`` filter is the no-leakage guard; the lower bound is
    the lookback window; ``odds > 1.0`` drops degenerate quotes defensively."""
    lo = now - timedelta(seconds=lookback_seconds)
    pts = [(t, o) for (t, o) in trajectory if lo <= t <= now and o > 1.0]
    pts.sort(key=lambda p: p[0])
    return pts


def evaluate_steam(
    *,
    fill_trajectory: Sequence[TrajectoryPoint],
    anchor_trajectory: Sequence[TrajectoryPoint],
    now: datetime,
    policy: SteamPolicy,
) -> SteamVerdict:
    """Movement verdict for one candidate from the fill-book and anchor-book
    price trajectories. Pure; consults only ``captured_at <= now`` points."""
    fill = _window(fill_trajectory, now, policy.lookback_seconds)
    anchor = _window(anchor_trajectory, now, policy.lookback_seconds)

    reasons: list[str] = []

    # --- stale anchor -------------------------------------------------------
    anchor_age: float | None = None
    stale = False
    if not anchor:
        # No anchor observation in the window: freshness cannot be confirmed, so
        # the edge sits against an unverifiable price -> conservative stale.
        stale = True
        reasons.append("stale_anchor")
    else:
        anchor_age = (now - anchor[-1][0]).total_seconds()
        if anchor_age > policy.anchor_staleness_seconds:
            stale = True
            reasons.append("stale_anchor")

    # --- soft movement: convergence + steam-away ----------------------------
    closed_fraction: float | None = None
    soft_move: float | None = None
    toward = False
    steamed = False
    if len(fill) >= policy.min_points:
        p_fill_open = implied_prob(fill[0][1])
        p_fill_now = implied_prob(fill[-1][1])
        soft_move = p_fill_now - p_fill_open
        # Soft odds lengthening (implied prob dropping) sharply = steaming away.
        if -soft_move >= policy.soft_steam_away_delta:
            steamed = True
            reasons.append("soft_steamed_away")
        # Convergence is defined on the GAP between fill and anchor, using each
        # series' own endpoints (the edge evaporates whether the soft corrects
        # up or the anchor drifts down). Needs an anchor to compare against.
        if anchor:
            p_anchor_open = implied_prob(anchor[0][1])
            p_anchor_now = implied_prob(anchor[-1][1])
            gap_open = p_anchor_open - p_fill_open
            gap_now = p_anchor_now - p_fill_now
            if gap_open >= policy.min_initial_gap:
                closed_fraction = (gap_open - gap_now) / gap_open
                if gap_now < gap_open and closed_fraction >= policy.soft_toward_anchor_close_frac:
                    toward = True
                    reasons.append("soft_toward_anchor")

    tripped = toward or stale or (steamed and policy.demote_on_soft_steam)
    return SteamVerdict(
        soft_toward_anchor=toward,
        stale_anchor=stale,
        soft_steamed=steamed,
        tripped=tripped,
        reasons=tuple(reasons),
        closed_fraction=closed_fraction,
        anchor_age_seconds=anchor_age,
        soft_move=soft_move,
    )


class _SnapshotLike(Protocol):
    """Structural input to ``build_trajectories`` — an OddsSnapshotIn satisfies it.

    Members are read-only properties so the protocol matches COVARIANTLY: a
    concrete ``market: Market`` (enum) satisfies ``market`` typed as ``object``."""

    @property
    def event_id(self) -> str: ...
    @property
    def bookmaker(self) -> str: ...
    @property
    def market(self) -> object: ...
    @property
    def selection(self) -> str: ...
    @property
    def decimal_odds(self) -> float: ...
    @property
    def market_detail(self) -> str | None: ...
    @property
    def captured_at(self) -> datetime: ...


# Grouping key shared with the value pipeline: (event_id, market, market_detail).
TrajectoryKey = tuple[str, object, str | None]
# Per (selection, normalized-book) ordered price history within one market.
MarketTrajectories = dict[tuple[str, str], list[TrajectoryPoint]]


def build_trajectories(
    snapshots: Sequence[_SnapshotLike],
    now: datetime,
    lookback_seconds: float,
) -> dict[TrajectoryKey, MarketTrajectories]:
    """Group odds snapshots into per-(market) per-(selection, book) trajectories.

    Returns ``{(event_id, market, market_detail): {(selection, book_norm):
    [(captured_at, odds), ...]}}`` with each series sorted oldest-first and
    filtered to the lookback window AND ``captured_at <= now`` (no future
    leakage). Book names are normalized (strip+lower) so a candidate's fill/
    anchor book — looked up the same way — matches regardless of source casing.

    Pure helper: the caller assembles ``snapshots`` from ``odds_snapshots``
    history plus the current cycle; this function never does IO.
    """
    lo = now - timedelta(seconds=lookback_seconds)
    out: dict[TrajectoryKey, MarketTrajectories] = {}
    for s in snapshots:
        captured = s.captured_at
        if not (lo <= captured <= now):
            continue
        odds = float(s.decimal_odds)
        if odds <= 1.0:
            continue
        key: TrajectoryKey = (s.event_id, s.market, s.market_detail)
        sel_book = (s.selection, s.bookmaker.strip().lower())
        out.setdefault(key, {}).setdefault(sel_book, []).append((captured, odds))
    for market in out.values():
        for series in market.values():
            series.sort(key=lambda p: p[0])
    return out


def lookup_trajectory(
    market_trajectories: Mapping[tuple[str, str], list[TrajectoryPoint]],
    selection: str,
    book: str,
) -> list[TrajectoryPoint]:
    """The ordered price series for one (selection, book), or [] when absent.
    Book is normalized to match ``build_trajectories`` keys."""
    return list(market_trajectories.get((selection, book.strip().lower()), []))
