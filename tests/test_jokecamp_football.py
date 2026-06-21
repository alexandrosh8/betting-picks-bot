"""jokecamp/FootballData read-only loader (MIT). Pure-parser tests — no network.

The source all.csv is HEADERLESS, 18 positional columns; 'None' = missing. Only
~86k of 157k rows carry Pinnacle 1X2 (cols 11-13). A NAMED Pinnacle close for 22
countries beyond football-data.co.uk — used to validate the SHARP ANCHOR's
calibration on an independent sample (no soft best-price column, so NOT a
sharp-vs-soft ROI source). Frozen 2004-2016 backfill.
"""

from datetime import date

from app.ingestion.jokecamp_football import JokecampMatch, parse_jokecamp_rows


def _row(**kw: str) -> list[str]:
    return [
        kw.get("id", "29"),
        kw.get("country", "finland"),
        kw.get("league", "veikkausliiga-2014"),
        kw.get("home", "MyPa"),
        kw.get("away", "Honka"),
        kw.get("round", "17.0"),
        kw.get("url", "http://betexplorer/x"),
        kw.get("awa", "None"),
        kw.get("hs", "4.0"),
        kw.get("as_", "3.0"),
        kw.get("date", "2014-04-06"),
        kw.get("ph", "2.19"),
        kw.get("pd", "3.31"),
        kw.get("pa", "3.78"),
        kw.get("ah1", "2.19"),
        kw.get("ah2", "1.77"),
        kw.get("ah3", "1.53"),
        kw.get("ah4", "2.64"),
    ]


def test_parse_jokecamp_keeps_odds_bearing_rows() -> None:
    matches = parse_jokecamp_rows([_row()])
    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, JokecampMatch)
    assert m.country == "finland"
    assert m.home_team == "MyPa" and m.away_team == "Honka"
    assert m.home_score == 4 and m.away_score == 3
    assert m.match_date == date(2014, 4, 6)
    assert m.pinnacle_home == 2.19
    assert m.pinnacle_draw == 3.31
    assert m.pinnacle_away == 3.78


def test_parse_jokecamp_skips_rows_without_pinnacle_1x2() -> None:
    # 1999 rows predate Pinnacle coverage -> cols 11-13 are "None" -> skipped.
    assert parse_jokecamp_rows([_row(ph="None", pd="None", pa="None")]) == []


def test_parse_jokecamp_skips_rows_without_score() -> None:
    assert parse_jokecamp_rows([_row(hs="None", as_="None")]) == []
