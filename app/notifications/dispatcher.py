"""Alert dispatcher: idempotency-gate then fan out to all configured sinks.

Claim/release contract: the dedupe key is claimed BEFORE sending, but the
claim only sticks if the alert actually REACHED a channel. If every
configured sink fails (or the fan-out itself raises), the claim is released
so the pipeline's next-cycle re-dispatch of the same market state retries —
claim-then-crash must never consume an alert forever.
"""

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
        any_configured = False
        any_delivered = False
        try:
            for sink in self._sinks:
                # Unconfigured sinks (no token/url) skip by design — their
                # False is not a delivery FAILURE, so it must not trigger the
                # release below (that would re-dispatch every cycle forever
                # on a no-channel deployment). Duck-typed: sinks without the
                # attribute count as configured.
                any_configured = any_configured or bool(getattr(sink, "configured", True))
                try:
                    delivered = await sink.send(alert)
                except Exception as exc:  # sinks shouldn't raise; belt and braces
                    logger.error("sink %s raised %s", sink.name, type(exc).__name__)
                    delivered = False
                results.append((sink.name, delivered))
                any_delivered = any_delivered or delivered
        except Exception:
            # claimed but never fully fanned out — hand the key back, then
            # surface the failure to the caller (the poll loop logs it).
            await self._store.release(alert.dedupe_key)
            raise
        if any_configured and not any_delivered:
            # Claimed but delivered nowhere: release so the next cycle's
            # re-dispatch retries. Partial delivery keeps the claim —
            # retrying would duplicate the alert to the healthy channel.
            await self._store.release(alert.dedupe_key)
            logger.warning(
                "alert for pick %s reached no sink; claim released for retry next cycle",
                alert.pick_id,
            )
        return DispatchResult(alert=alert, skipped_duplicate=False, sink_results=tuple(results))
