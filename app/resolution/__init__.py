"""Pure, deterministic cross-source event resolution.

Links the same real-world fixture across odds sources (OddsPortal events vs the
Pinnacle arcadia archive) so a sharp Pinnacle close can be attached to a pick
for incremental CLV. STRICT matching only — no fuzzy joins (see matching.py).
"""

from app.resolution.matching import (
    AliasTable,
    EventCandidate,
    default_aliases,
    match_event,
    normalize_name,
)

__all__ = [
    "AliasTable",
    "EventCandidate",
    "default_aliases",
    "match_event",
    "normalize_name",
]
