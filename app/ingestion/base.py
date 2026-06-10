"""Ingestion contracts. ALL loaders are READ-ONLY (GET) by design — no code
in this package may write to any bookmaker, exchange, or odds provider."""

from collections.abc import Sequence
from typing import Protocol

from app.schemas.odds import OddsSnapshotIn


class OddsLoader(Protocol):
    """A source of odds snapshots for one sport key."""

    async def fetch_odds(self, sport_key: str) -> Sequence[OddsSnapshotIn]: ...
