"""The Odds API client (read-only, GET-only) with multi-key rotation.

Secret hygiene: API keys travel only in query params of outbound requests;
they are NEVER included in exceptions, logs, or any stringified output.
Keys advance on 401/429 (invalid/exhausted); transport errors retry with
exponential backoff; other 4xx fail without retry (they would burn credits).
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.schemas.base import Market
from app.schemas.odds import OddsSnapshotIn

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
# Provider market keys -> internal Market enum
_MARKET_MAP = {
    "h2h": Market.H2H,
    "spreads": Market.SPREADS,
    "totals": Market.TOTALS,
}


class OddsApiError(Exception):
    """Raised when no key can fetch odds. Message never contains a key."""


class OddsApiClient:
    """Read-only Odds API client. Keys rotate on 401/429."""

    def __init__(
        self,
        api_keys: Sequence[str],
        client: httpx.AsyncClient,
        base_url: str = DEFAULT_BASE_URL,
        regions: str = "eu",
        markets: str = "h2h,totals,spreads",
    ) -> None:
        if not api_keys:
            raise ValueError("at least one Odds API key is required")
        self._keys = tuple(api_keys)
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._regions = regions
        self._markets = markets

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=8.0),
        reraise=True,
    )
    async def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        return await self._client.get(url, params=params, timeout=20.0)

    async def fetch_odds(self, sport_key: str) -> list[OddsSnapshotIn]:
        url = f"{self._base_url}/sports/{sport_key}/odds"
        rotated = 0
        for key in self._keys:
            response = await self._get(
                url,
                params={
                    "apiKey": key,
                    "regions": self._regions,
                    "markets": self._markets,
                    "oddsFormat": "decimal",
                },
            )
            if response.status_code in (401, 429):
                rotated += 1
                logger.warning(
                    "odds api key #%d rejected (status %d); rotating",
                    rotated,
                    response.status_code,
                )
                continue
            if response.status_code != 200:
                # Never include response.url (carries apiKey) in the error.
                raise OddsApiError(
                    f"odds api returned status {response.status_code} for sport={sport_key}"
                )
            return self._parse(response.json())
        raise OddsApiError(
            f"all {len(self._keys)} odds api keys exhausted (401/429) for sport={sport_key}"
        )

    def _parse(self, payload: list[dict[str, Any]]) -> list[OddsSnapshotIn]:
        now = datetime.now(tz=UTC)
        snapshots: list[OddsSnapshotIn] = []
        for event in payload:
            event_id = str(event.get("id", ""))
            if not event_id:
                continue
            bookmakers = event.get("bookmakers") or []
            for bookmaker in bookmakers:
                book_key = str(bookmaker.get("key", "unknown"))
                last_update = _parse_ts(str(bookmaker.get("last_update", ""))) or now
                for market in bookmaker.get("markets") or []:
                    market_key = str(market.get("key", ""))
                    mapped = _MARKET_MAP.get(market_key)
                    if mapped is None:
                        continue
                    for outcome in market.get("outcomes") or []:
                        price = outcome.get("price")
                        name = str(outcome.get("name", ""))
                        point = outcome.get("point")
                        if not isinstance(price, int | float) or price <= 1.0:
                            continue
                        selection = f"{name} {point}" if point is not None else name
                        snapshots.append(
                            OddsSnapshotIn(
                                event_id=event_id,
                                bookmaker=book_key,
                                market=mapped,
                                selection=selection,
                                decimal_odds=float(price),
                                captured_at=last_update,
                                ingested_at=now,
                            )
                        )
        return snapshots


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
