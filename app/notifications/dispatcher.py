"""Alert dispatcher: idempotency-gate then fan out to all configured sinks."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.notifications.base import Alert, AlertSink
from app.notifications.dedupe import IdempotencyStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchResult:
    alert: Alert
    skipped_duplicate: bool
    sink_results: tuple[tuple[str, bool], ...]  # (sink name, delivered)


class AlertDispatcher:
    def __init__(self, sinks: Sequence[AlertSink], store: IdempotencyStore) -> None:
        self._sinks = tuple(sinks)
        self._store = store

    async def dispatch(self, alert: Alert) -> DispatchResult:
        if not await self._store.claim(alert.dedupe_key):
            logger.info("duplicate alert suppressed for pick %s", alert.pick_id)
            return DispatchResult(alert=alert, skipped_duplicate=True, sink_results=())

        results: list[tuple[str, bool]] = []
        for sink in self._sinks:
            try:
                delivered = await sink.send(alert)
            except Exception as exc:  # sinks shouldn't raise; belt and braces
                logger.error("sink %s raised %s", sink.name, type(exc).__name__)
                delivered = False
            results.append((sink.name, delivered))
        return DispatchResult(alert=alert, skipped_duplicate=False, sink_results=tuple(results))
