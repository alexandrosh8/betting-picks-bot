"""Free OddsPortal odds via OddsHarvester, used directly (ADR-0012).

OddsHarvester (MIT, jordantete/OddsHarvester) scrapes oddsportal.com — an
odds AGGREGATOR, not a bookmaker: read-only data collection, no betting
surface. Its `run_scraper` coroutine is consumed as-is; this module only
adapts its match dicts into our `OddsSnapshotIn` stream and registers team
context in the EventDirectory for the model layer.

The oddsharvester import is lazy so the default (extras-free) install and CI
profile keep working; install with `uv sync --extra backfill`.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from app.ingestion.base import EventDirectory, EventTeams
from app.schemas.base import Market
from app.schemas.odds import OddsSnapshotIn

logger = logging.getLogger(__name__)

ScrapeFn = Callable[..., Awaitable[Any]]

# OddsHarvester market keys we can devig SOUNDLY (mutually-exclusive,
# full-coverage outcomes). Asian/European handicap keys are deliberately
# rejected: each line is a separate submarket (integer/quarter AH lines
# carry push outcomes), so naive devig is invalid — pricing those needs the
# penaltyblog grid bridge (goal_expectancy_extended + create_dixon_coles_grid,
# docs/research/value-platform-repo-research.md).
_EXACT_MARKETS: dict[str, Market] = {
    "1x2": Market.H2H,  # football/basketball 3-way
    "home_away": Market.H2H,  # basketball moneyline (OddsHarvester has no "moneyline" key)
    "btts": Market.BTTS,
    "dnb": Market.DNB,
    "double_chance": Market.DOUBLE_CHANCE,
}


def _market_for_key(key: str) -> Market | None:
    if key in _EXACT_MARKETS:
        return _EXACT_MARKETS[key]
    if key.startswith("over_under_"):
        return Market.TOTALS
    return None


def _totals_line(market_key: str) -> str:
    """'over_under_2_5' -> '2.5'; 'over_under_games_215_5' -> '215.5'."""
    raw = market_key.removeprefix("over_under_").removeprefix("games_")
    return raw.replace("_", ".")


def _validate_markets(markets: Sequence[str]) -> None:
    unknown = [m for m in markets if _market_for_key(m) is None]
    if unknown:
        hint = ""
        if any("handicap" in m for m in unknown):
            hint = (
                " (handicap submarkets cannot be naively devigged — they need "
                "the derived-pricing bridge, see app/edge/value.py docstring)"
            )
        raise ValueError(f"unsupported oddsportal markets: {unknown}{hint}")
    totals = [m for m in markets if m.startswith("over_under_")]
    if len(totals) > 1:
        # multiple totals lines would collapse into ONE Market.TOTALS group
        # per event and corrupt devig (selections across lines are not
        # mutually exclusive).
        raise ValueError(f"configure at most one over/under line, got: {totals}")


async def _default_scrape(**kwargs: Any) -> Any:
    """Call OddsHarvester's run_scraper as-is (lazy import)."""
    from oddsharvester.core.scraper_app import run_scraper
    from oddsharvester.utils.command_enum import CommandEnum

    return await run_scraper(command=CommandEnum.UPCOMING_MATCHES, **kwargs)


class OddsPortalLoader:
    """OddsLoader over OddsHarvester's upcoming-matches scraper."""

    def __init__(
        self,
        directory: EventDirectory,
        leagues_by_sport_key: dict[str, tuple[str, list[str]]],
        markets: Sequence[str] = ("1x2", "over_under_2_5"),
        scrape_fn: ScrapeFn | None = None,
        headless: bool = True,
        max_pages: int = 1,
        date: str | None = None,
        markets_by_sport_key: dict[str, Sequence[str]] | None = None,
    ) -> None:
        """`leagues_by_sport_key` maps our sport key (e.g. "soccer") to
        (oddsharvester sport, [oddsportal league slugs]). `markets_by_sport_key`
        overrides `markets` per sport key (football and basketball use
        different OddsHarvester market keys). `date` is an optional YYYYMMDD
        filter; None (default) scrapes the general upcoming page, which is
        what carries live pre-match odds."""
        self._markets_by_sport = {k: tuple(v) for k, v in (markets_by_sport_key or {}).items()}
        for market_list in (tuple(markets), *self._markets_by_sport.values()):
            _validate_markets(market_list)
        self._directory = directory
        self._config = dict(leagues_by_sport_key)
        self._markets = tuple(markets)
        self._scrape = scrape_fn or _default_scrape
        self._headless = headless
        self._max_pages = max_pages
        self._date = date

    def _markets_for(self, sport_key: str) -> tuple[str, ...]:
        return self._markets_by_sport.get(sport_key, self._markets)

    async def fetch_odds(self, sport_key: str) -> list[OddsSnapshotIn]:
        if sport_key not in self._config:
            logger.warning("no oddsportal league config for sport key %s", sport_key)
            return []
        sport, leagues = self._config[sport_key]
        now = datetime.now(tz=UTC)
        result = await self._scrape(
            sport=sport,
            date=self._date,
            leagues=leagues,
            markets=list(self._markets_for(sport_key)),
            headless=self._headless,
            max_pages=self._max_pages,
        )
        matches = getattr(result, "success", None) or []
        snapshots: list[OddsSnapshotIn] = []
        for match in matches:
            snapshots.extend(self._convert_match(match, now, self._markets_for(sport_key)))
        logger.info(
            "oddsportal %s: %d matches -> %d snapshots", sport_key, len(matches), len(snapshots)
        )
        return snapshots

    def _convert_match(
        self, match: dict[str, Any], now: datetime, markets: Sequence[str]
    ) -> list[OddsSnapshotIn]:
        home = str(match.get("home_team") or "").strip()
        away = str(match.get("away_team") or "").strip()
        if not home or not away:
            return []
        event_id = str(match.get("match_link") or f"{home}|{away}|{match.get('match_date', '')}")
        league = str(match.get("league_name") or "")
        self._directory.register(
            event_id,
            EventTeams(
                home=home,
                away=away,
                league=league,
                starts_at=_parse_ts(match.get("match_date")),
            ),
        )
        captured_at = _parse_ts(match.get("scraped_date")) or now

        snapshots: list[OddsSnapshotIn] = []
        for market_key in markets:
            entries = match.get(f"{market_key}_market") or []
            market = _market_for_key(market_key)
            if market is None:  # pragma: no cover — blocked by _validate_markets
                continue
            seen_books: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                bookmaker = str(entry.get("bookmaker_name") or "unknown")
                # OddsHarvester can emit duplicate bookmaker rows (e.g. odds
                # history); duplicates would corrupt devig (6-leg "markets").
                if bookmaker in seen_books:
                    continue
                seen_books.add(bookmaker)
                for label, selection in _selections(market_key, home, away):
                    odds = _parse_odds(entry.get(label))
                    if odds is None:
                        continue
                    snapshots.append(
                        OddsSnapshotIn(
                            event_id=event_id,
                            bookmaker=bookmaker,
                            market=market,
                            selection=selection,
                            decimal_odds=odds,
                            captured_at=captured_at,
                            ingested_at=now,
                        )
                    )
        return snapshots


def _selections(market_key: str, home: str, away: str) -> list[tuple[str, str]]:
    """OddsHarvester odds-label -> our selection name (must match what the
    model layer emits so the pipeline join works). Label names verified
    against OddsHarvester's SportMarketRegistrar (2026-06-10 research)."""
    if market_key == "1x2":
        return [("1", home), ("X", "Draw"), ("2", away)]
    if market_key == "home_away":  # basketball moneyline
        return [("1", home), ("2", away)]
    if market_key == "btts":
        return [("btts_yes", "BTTS Yes"), ("btts_no", "BTTS No")]
    if market_key == "dnb":
        return [("dnb_team1", home), ("dnb_team2", away)]
    if market_key == "double_chance":
        return [
            ("1X", f"{home} or Draw"),
            ("12", f"{home} or {away}"),
            ("X2", f"Draw or {away}"),
        ]
    if market_key.startswith("over_under_"):
        line = _totals_line(market_key)
        return [("odds_over", f"Over {line}"), ("odds_under", f"Under {line}")]
    return []


def _parse_odds(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if value > 1.0 else None


def _parse_ts(raw: Any) -> datetime | None:
    """Parse OddsHarvester timestamps. The installed version emits
    'YYYY-MM-DD HH:MM:SS UTC' (not ISO) — handle both, plus epoch floats."""
    if not raw:
        return None
    text = str(raw).strip()
    # epoch seconds (odds-history rows)
    try:
        return datetime.fromtimestamp(float(text), tz=UTC)
    except (ValueError, OverflowError, OSError):
        pass
    cleaned = text.replace("Z", "+00:00")
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4]
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
