"""Poll-skip log-noise filter (app/scheduler.py).

The 60s poll interval + max_instances=1 + coalesce is the documented
continuous-polling design: a 20-40 min scrape cycle makes apscheduler skip
every overlapping slot, which is expected — not WARNING-worthy. Skips of any
OTHER job must stay visible.
"""

import logging

from app.scheduler import _PollSkipNoiseFilter

_SKIP_MSG = (
    'Execution of job "build_scheduler.<locals>.poll_odds '
    '(trigger: interval[0:01:00], next run at: 2026-06-11 15:32:29 EEST)" '
    "skipped: maximum number of running instances reached (1)"
)


def _rec(msg: str) -> logging.LogRecord:
    return logging.LogRecord("apscheduler.scheduler", logging.WARNING, __file__, 0, msg, None, None)


def test_drops_poll_odds_max_instances_skip() -> None:
    assert not _PollSkipNoiseFilter().filter(_rec(_SKIP_MSG))


def test_keeps_other_job_skips_and_other_warnings() -> None:
    f = _PollSkipNoiseFilter()
    assert f.filter(
        _rec(
            'Execution of job "settle_results" skipped: '
            "maximum number of running instances reached (1)"
        )
    )
    assert f.filter(_rec("Run time of job poll_odds was missed by 0:00:05"))
