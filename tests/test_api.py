"""API surface: health endpoint and payload validation (no DB required)."""

import re
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_session
from app.api.routes import router


async def _no_session() -> AsyncIterator[None]:
    yield None


def make_app() -> FastAPI:
    # Router only — lifespan (DB/scheduler) intentionally not started; the
    # session dependency is stubbed so validation paths can be exercised.
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = _no_session
    return app


def test_health_reports_picks_only_mode() -> None:
    client = TestClient(make_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "picks-only"


def test_health_exposes_poll_liveness_payload() -> None:
    # The dashboard renders a degraded state (selector break / anti-bot wall:
    # matches listed, zero odds parsed) straight from the polls payload —
    # per-market counts, listing count and the explicit flag must pass through.
    from app.pipeline import LAST_POLL

    LAST_POLL["soccer"] = {
        "finished_at": "2026-06-11T00:00:00+00:00",
        "snapshots": 0,
        "picks": 0,
        "matches_found": 7,
        "per_market": {},
        "degraded": True,
    }
    try:
        body = TestClient(make_app()).get("/health").json()
        poll = body["polls"]["soccer"]
        assert poll["degraded"] is True
        assert poll["matches_found"] == 7
        assert poll["per_market"] == {}
    finally:
        LAST_POLL.pop("soccer", None)


def test_health_exposes_poll_interval_seconds() -> None:
    # The dashboard's "verified within" window must track the configured poll
    # cadence (max(45min, 3 * poll_interval)) instead of hardcoding 45 min —
    # so the cadence has to ride in the health payload.
    body = TestClient(make_app()).get("/health").json()
    assert isinstance(body["poll_interval_seconds"], int)
    assert body["poll_interval_seconds"] >= 30  # Settings enforces the floor


def test_dashboard_served_at_root() -> None:
    client = TestClient(make_app())
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # safety reminder must be visible on the dashboard
    assert "not</b> place bets" in response.text
    assert 'id="picks-table"' in response.text
    # untrusted scrape strings must never go through innerHTML
    assert "innerHTML" not in response.text


def test_dashboard_fetches_are_timeout_guarded() -> None:
    # Regression (browser QA, 2026-06-11): with postgres paused, /picks and
    # /performance hang ~80s before failing while /health answers instantly.
    # Each 60s tick's load() then failed ~20s AFTER the next tick had already
    # started, so the offline banner never rendered. The dashboard must
    # (a) abort every fetch after 15s, (b) never start a tick while a load()
    # is in flight, and (c) render a distinct ENGINE UNRESPONSIVE state for
    # the aborted/timed-out case (process up, not answering) — different from
    # OFFLINE (connection refused) and SERVER ERROR (HTTP 5xx).
    text = TestClient(make_app()).get("/").text
    assert "function fetchWithTimeout" in text
    # every data fetch must go through the timeout helper — a bare fetch(
    # would reintroduce the indefinite hang
    assert 'fetchWithTimeout("/picks' in text
    assert 'fetchWithTimeout("/performance' in text
    assert 'fetchWithTimeout("/health' in text
    # settle POST too (formatter may wrap the URL onto the next line)
    assert re.search(r'fetchWithTimeout\(\s*"/events/', text)
    # the raw fetch( primitive appears exactly once: inside the helper
    assert text.count("fetch(") == 1
    # in-flight guard: a new tick must not pile onto a hung load()
    assert "LOAD_IN_FLIGHT" in text
    # the distinct third banner state
    assert "ENGINE UNRESPONSIVE" in text
    assert "UNRESPONSIVE" in text


def test_result_payload_validation_rejects_bad_outcome() -> None:
    client = TestClient(make_app())
    response = client.post(
        "/picks/1/result",
        json={
            "pick_id": "1",
            "outcome": "smashed_it",  # not a valid Outcome
            "settled_at": "2026-06-10T12:00:00Z",
        },
    )
    assert response.status_code == 422


def test_result_payload_validation_rejects_naive_datetime() -> None:
    client = TestClient(make_app())
    response = client.post(
        "/picks/1/result",
        json={
            "pick_id": "1",
            "outcome": "won",
            "settled_at": "2026-06-10T12:00:00",  # naive
        },
    )
    assert response.status_code == 422


def test_event_result_rejects_negative_and_missing_scores() -> None:
    client = TestClient(make_app())
    assert (
        client.post("/events/1/result", json={"home_score": -1, "away_score": 0}).status_code == 422
    )
    assert client.post("/events/1/result", json={"home_score": 2}).status_code == 422
    assert (
        client.post("/events/1/result", json={"home_score": 2, "away_score": "x"}).status_code
        == 422
    )
