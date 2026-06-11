"""Secret hygiene: httpx logs the FULL request URL at INFO (Telegram bot
tokens ride in the URL path, Odds API keys in query strings) — the app must
pin the 'httpx'/'httpcore' loggers to WARNING so no URL ever reaches a log
record, at ANY configured LOG_LEVEL.
"""

import logging

import httpx
import pytest

from app.main import _silence_url_logging
from app.notifications.base import Alert
from app.notifications.telegram import TelegramSink

FAKE_TOKEN = "1234567890:FAKE-TEST-TOKEN-abcdef"  # synthetic — not a real secret


def test_silence_url_logging_pins_http_client_loggers_to_warning() -> None:
    # Even with root at DEBUG (the regression: explicit INFO pins used to
    # OVERRIDE a stricter LOG_LEVEL=WARNING and still log URLs at INFO).
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.DEBUG)
        _silence_url_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
    finally:
        logging.getLogger().setLevel(logging.WARNING)


async def test_telegram_dispatch_never_logs_url_or_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    async with httpx.AsyncClient(transport=transport) as client:
        sink = TelegramSink(FAKE_TOKEN, "chat-1", client)
        with caplog.at_level(logging.DEBUG):
            _silence_url_logging()
            ok = await sink.send(Alert(pick_id="p1", title="t", body="b", dedupe_key="k"))
    assert ok is True
    for record in caplog.records:
        message = record.getMessage()
        assert "api.telegram.org" not in message
        assert FAKE_TOKEN not in message
        assert "FAKE-TEST-TOKEN" not in message
