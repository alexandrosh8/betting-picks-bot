"""Sharp-vs-soft value finder — the strategy with backtested positive CLV.

Instead of predicting outcomes with a goals model (which does NOT beat the
market — see docs/backtesting/findings.md), this prices "fair" from the
SHARPEST book available (Pinnacle by preference, else the lowest-overround
book / consensus) and flags selections where the BEST price across the other
books exceeds that fair value. Line-shopping the soft-vs-sharp gap is what
produces positive CLV (docs/backtesting/value-findings.md).

Pure module: no IO. Input is per-bookmaker decimal odds for one market.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.probabilities.devig import DevigMethod, devig

# Books treated as "sharp" for fair-value estimation, in priority order.
SHARP_BOOKS = ("pinnacle", "pinnacle sports", "betfair exchange", "smarkets")


@dataclass(frozen=True)
class ValueBet:
    selection: str
    best_book: str
    best_odds: float
    sharp_book: str
    sharp_fair_prob: float
    implied_prob: float
    edge: float  # sharp_fair_prob - implied_prob(best price)
    ev: float  # per unit stake, using sharp fair as the truth


def _norm(s: str) -> str:
    return s.strip().lower()


def _overround(odds: Sequence[float]) -> float:
    return sum(1.0 / o for o in odds) - 1.0


def find_value_bets(
    prices: Mapping[str, Mapping[str, float]],
    *,
    min_edge: float = 0.01,
    devig_method: DevigMethod = DevigMethod.POWER,
    sharp_books: Sequence[str] = SHARP_BOOKS,
) -> list[ValueBet]:
    """Find value selections for one market.

    `prices` maps selection -> {bookmaker: decimal_odds}. Selections are the
    market's mutually-exclusive outcomes (e.g. home/Draw/away). Returns value
    bets sorted by edge (desc).
    """
    selections = list(prices.keys())
    if len(selections) < 2:
        return []

    sharp_book, sharp_odds = _select_sharp_book(prices, selections, sharp_books)
    if sharp_book is None or sharp_odds is None:
        return []
    sharp_fair = devig(sharp_odds, method=devig_method)
    fair_by_sel = dict(zip(selections, sharp_fair, strict=True))

    out: list[ValueBet] = []
    for sel in selections:
        best_book, best_odds = _best_other_book(prices[sel], sharp_book)
        if best_book is None or best_odds is None:
            continue
        implied = 1.0 / best_odds
        fair_p = fair_by_sel[sel]
        edge = fair_p - implied
        if edge < min_edge:
            continue
        ev = fair_p * (best_odds - 1.0) - (1.0 - fair_p)
        out.append(
            ValueBet(
                selection=sel,
                best_book=best_book,
                best_odds=best_odds,
                sharp_book=sharp_book,
                sharp_fair_prob=fair_p,
                implied_prob=implied,
                edge=edge,
                ev=ev,
            )
        )
    out.sort(key=lambda v: v.edge, reverse=True)
    return out


def _select_sharp_book(
    prices: Mapping[str, Mapping[str, float]],
    selections: Sequence[str],
    sharp_books: Sequence[str],
) -> tuple[str | None, list[float] | None]:
    """Pick the fair-value source: a named sharp book that prices every
    selection, else the book with the lowest overround that does."""
    books = set.intersection(*[{_norm(b) for b in prices[s]} for s in selections])
    if not books:
        return None, None

    raw_by_norm: dict[str, str] = {}
    for s in selections:
        for b in prices[s]:
            raw_by_norm.setdefault(_norm(b), b)

    for pref in sharp_books:
        if pref in books:
            odds = [_lookup(prices[s], pref) for s in selections]
            if all(o is not None for o in odds):
                return raw_by_norm[pref], [float(o) for o in odds]  # type: ignore[arg-type]

    best_book, best_or, best_odds = None, float("inf"), None
    for b in books:
        odds = [_lookup(prices[s], b) for s in selections]
        if any(o is None for o in odds):
            continue
        ovr = _overround([float(o) for o in odds])  # type: ignore[arg-type]
        if ovr < best_or:
            best_book = raw_by_norm[b]
            best_or = ovr
            best_odds = [float(o) for o in odds]  # type: ignore[arg-type]
    return best_book, best_odds


def _best_other_book(
    book_odds: Mapping[str, float], sharp_book: str
) -> tuple[str | None, float | None]:
    """Best (highest) decimal odds among books other than the sharp source."""
    best_book, best_odds = None, 0.0
    sharp_norm = _norm(sharp_book)
    for book, odds in book_odds.items():
        if _norm(book) == sharp_norm:
            continue
        if odds > best_odds and odds > 1.0:
            best_book, best_odds = book, odds
    return (best_book, best_odds) if best_book is not None else (None, None)


def _lookup(book_odds: Mapping[str, float], norm_book: str) -> float | None:
    for book, odds in book_odds.items():
        if _norm(book) == norm_book:
            return odds
    return None
