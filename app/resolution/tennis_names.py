"""Canonicalize tennis player names across the two cross-source name formats.

OddsPortal tennis writes "Surname I." (surname first, then a first-initial); the
Pinnacle ARCADIA archive writes "Firstname Surname". A full first name is NOT
recoverable from "Surname I.", so the only ground both share is

    SURNAME + FIRST-INITIAL

e.g. both ``"Tiafoe F."`` and ``"Frances Tiafoe"`` collapse to ``"tiafoe f"``.
This lets the strict cross-source matcher pair tennis fixtures that previously
went unmatched (the surname-initial vs first-last gap).

Conservative by design: matching on surname + first-initial is looser than the
full-name match used for team sports, but tennis is a TWO-player fixture and the
matcher still requires BOTH players to agree within the day window, so a
surname+initial collision cannot mis-pair a real fixture in practice (you would
need two different players sharing surname AND first-initial facing two more such
players on the same day). Known limitation: a multi-token surname in the
"Firstname Surname" form (e.g. "Pablo Carreno Busta") keeps only its last token,
so it will MISS rather than mis-match its "Carreno Busta P." counterpart — a safe
miss, never a wrong close. Pure stdlib, no IO.
"""

from __future__ import annotations

import unicodedata


def _norm_token(token: str) -> str:
    """NFKD accent-strip -> ASCII -> casefold -> alphanumerics only."""
    decomposed = unicodedata.normalize("NFKD", token)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_only.casefold() if ch.isalnum())


def _is_initial(token: str) -> bool:
    """A single alphabetic letter, optionally with a trailing period: "F" / "F."."""
    stripped = token.rstrip(".")
    return len(stripped) == 1 and stripped.isalpha()


def canonical_tennis_name(name: str) -> str:
    """Map a tennis player name to ``"surname firstinitial"`` (normalized), or ""
    if it can't be parsed into a surname + initial.

    "Surname ... I." -> leading tokens are the surname, the trailing single
    letter is the first-initial. "Firstname ... Surname" -> the FIRST token gives
    the first-initial and the LAST token is the surname (best effort for compound
    surnames). A bare single token returns just its normalized form.
    """
    raw_tokens = [t for t in name.replace(",", " ").split() if t.strip()]
    if not raw_tokens:
        return ""
    if len(raw_tokens) == 1:
        return _norm_token(raw_tokens[0])
    if _is_initial(raw_tokens[-1]):  # "Surname ... I."
        initial = _norm_token(raw_tokens[-1])
        surname = "".join(_norm_token(t) for t in raw_tokens[:-1])
    elif _is_initial(raw_tokens[0]):  # "I. Surname ..."
        initial = _norm_token(raw_tokens[0])
        surname = "".join(_norm_token(t) for t in raw_tokens[1:])
    else:  # "Firstname ... Surname"
        initial = _norm_token(raw_tokens[0])
        surname = _norm_token(raw_tokens[-1])
    if not surname or not initial:
        return ""
    return f"{surname} {initial[:1]}"
