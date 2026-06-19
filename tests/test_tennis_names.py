"""canonical_tennis_name reconciles the two tennis name formats (OddsPortal
"Surname I." vs Pinnacle "Firstname Surname") onto surname + first-initial, so
the strict matcher can pair fixtures that were going unmatched. The example
pairs are the REAL unmatched fixtures the readiness probe surfaced.
"""

from app.resolution.tennis_names import canonical_tennis_name

# Real same-fixture name pairs that previously failed to match (OddsPortal,
# Pinnacle ARCADIA) — each must collapse to one canonical surname+initial.
RECONCILED_PAIRS = [
    ("Tiafoe F.", "Frances Tiafoe", "tiafoe f"),
    ("Pegula J.", "Jessica Pegula", "pegula j"),
    ("Keys M.", "Madison Keys", "keys m"),
    ("Hurkacz H.", "Hubert Hurkacz", "hurkacz h"),
    ("Altmaier D.", "Daniel Altmaier", "altmaier d"),
    ("Auger-Aliassime F.", "Felix Auger-Aliassime", "augeraliassime f"),  # hyphenated
]


def test_cross_format_pairs_canonicalize_equal() -> None:
    for odds_name, pinnacle_name, expected in RECONCILED_PAIRS:
        a = canonical_tennis_name(odds_name)
        b = canonical_tennis_name(pinnacle_name)
        assert a == b == expected, (odds_name, pinnacle_name, a, b)


def test_accents_are_stripped_consistently() -> None:
    assert (
        canonical_tennis_name("Müller A.")
        == canonical_tennis_name("Alexander Müller")
        == "muller a"
    )


def test_leading_initial_form() -> None:
    assert canonical_tennis_name("F. Tiafoe") == "tiafoe f"


def test_distinct_first_initials_do_not_collide() -> None:
    # Alexander vs Mischa Zverev share a surname but NOT a first-initial.
    assert canonical_tennis_name("Zverev A.") != canonical_tennis_name("Zverev M.")
    assert canonical_tennis_name("Zverev A.") == "zverev a"
    assert canonical_tennis_name("Mischa Zverev") == "zverev m"


def test_single_token_returns_normalized() -> None:
    assert canonical_tennis_name("Nadal") == "nadal"


def test_empty_is_empty() -> None:
    assert canonical_tennis_name("") == ""
    assert canonical_tennis_name("   ") == ""


def test_compound_surname_in_first_last_form_misses_not_mismatches() -> None:
    # Documented limitation: "Firstname Surname1 Surname2" keeps only the last
    # token, so it MISSES its "Surname1 Surname2 I." counterpart rather than
    # producing a wrong match. A safe miss is the doctrine-correct failure mode.
    first_last = canonical_tennis_name("Pablo Carreno Busta")  # -> "busta p"
    surname_initial = canonical_tennis_name("Carreno Busta P.")  # -> "carrenobusta p"
    assert first_last != surname_initial
