# Pinnacle Odds Sources — Free, Read-Only Coverage Sweep (2026-06-19)

**Question:** Is there a FREE, read-only, no-auth Pinnacle odds source that would
meaningfully widen our sharp-close archive beyond the arcadia guest API we already
use in `app/ingestion/pinnacle_arcadia.py` — e.g. more leagues, more sports, or a
more reliable endpoint? If so, which exact endpoint/pattern to adopt.

**Driver:** picks cross-source match-rate is 32% (35/108). Of the 73 unmatched,
**34 are a COVERAGE gap** — the fixture has NO Pinnacle archive event at all. We
want a broader/more-reliable FREE Pinnacle source to close that 34.

**Method:** GitHub repo search + code search (via `gh api search/code` — the plugin
MCP `search_code` requires auth the server lacks; repo/file reads went through the
plugin MCP `mcp__plugin_everything-claude-code_github__*`). File-by-file inspection,
one real function quoted per core file. No clones, no install scripts run (read-only).

## TL;DR verdict

**No repo improves on the arcadia guest API.** Every FREE Pinnacle source on GitHub
(101+ files match `guest.api.arcadia.pinnacle.com` in code search) hits the **exact
same** `guest.api.arcadia.pinnacle.com/0.1` base we already use. There is no
secret "broader" free endpoint. Repos that expose _more_ (parlays, alt-lines,
V4 team-totals, balance) are the **PAID, authenticated** `api.pinnacle.com` /
`api.ps3838.com` API — auth-gated, account-required, and several are
bet-placement libraries → **doctrine-REJECTED for free coverage.**

**BUT** inspection surfaced **three doctrine-safe, additive API FACTS** (facts, not
code) that would widen OUR coverage within the same free guest API. These explain
the 34-fixture gap as a _sport/league-enumeration_ gap on our side, not a
missing-endpoint gap:

1. **Dynamic sport discovery** — `GET /0.1/sports` returns every sport with a live
   `matchupCount` and `isHidden` flag. We hardcode only 4 sport ids
   (`soccer=29, tennis=33, basketball=4, american_football=15`). The guest API
   carries **more** (verified ids in the wild: `MLB baseball=3, hockey=19,
rugby=27, handball=18`). Iterating active sports instead of a fixed 4 is the
   single biggest coverage lever.
2. **League-scoped endpoints** — `GET /0.1/leagues/{league_id}/matchups` and
   `GET /0.1/leagues/{league_id}/markets/straight` (same guest key, `?brandId=0`).
   Same data, league-scoped instead of sport-scoped. Pairs with
   `GET /0.1/sports/{sport_id}/leagues?all=false` to enumerate league ids. Lets us
   target/guarantee specific competitions rather than relying on the sport-wide
   matchups dump.
3. **Public key/base discovery** — `GET https://www.pinnacle.com/config/app.json`
   returns `api.haywire.apiKey` (the guest x-api-key), `...routes.curacao.guestRoot`
   (the base URL), and `apiVersion`. A pure public GET — **no Selenium, no login** —
   so the guest key can be refreshed if Pinnacle ever rotates it, hardening
   reliability vs. our current empty/optional key. (Bonus precision fact: the
   period-0 `cutoffAt` timestamp is the true betting cutoff; we use `startTime`.)

**Recommendation:** `adopt-pattern` (facts only) — extend `pinnacle_arcadia.py`
with (a) dynamic `/sports` enumeration and (b) optional `/config/app.json` key
discovery. Do **not** clone or copy any repo. No license is touched because we copy
no code. See "Adoption plan" below. Net: this is the only meaningful free-coverage
win available, and it lives entirely inside the source we already use.

---

## Scoring table (standard 11 columns)

| Repository                                                                                                                | Category                                                         | Stars / activity              | Core function                                                                    | Code quality                                                                         | Maintenance               | Directly reusable   | Best file to adapt                                                                                                            | Security concern                                                              | Auto-betting risk                               | Final decision                                                          |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| **eltonaguiar/findtorontoevents…archive**                                                                                 | Free guest-API moneyline scraper (stdlib, read-only)             | low / pushed 2026-05          | GET arcadia `/leagues/{id}/matchups` + `/markets/straight`, devig, snapshot JSON | High — stdlib only, GET-only, explicit "can't wager there, decision-support" framing | archive snapshot (frozen) | **partial — FACTS** | `tools/pinnacle_anchor_scrape.py` (`fetch_league()` quotes `cutoffAt`, league ids, sport_id=3 MLB)                            | none (no key committed; UA only)                                              | **none** (no placement, no auth)                | **adopt-pattern (facts: league endpoint, `cutoffAt`, MLB/NHL ids)**     |
| **theoam060-web/Pinnacle-Odds-Tracker**                                                                                   | Free guest-API coverage prober (TS)                              | low / pushed 2026             | enumerate ALL sports/matchups/markets; coverage stats                            | High — clean fetch, gzip, rate-limit, `primaryOnly=false`                            | active                    | **partial — FACTS** | `scripts/src/pinnacle-market-coverage.ts` (`main()` reads `/config/app.json` → key+guestRoot, then `/sports`)                 | key read from public config (not committed)                                   | **none**                                        | **adopt-pattern (facts: `/config/app.json`, dynamic `/sports`)**        |
| **pretrehr/Sports-betting**                                                                                               | Free guest-API multi-book scraper (Py)                           | ~mid / maintained             | arcadia `/leagues/{id}/matchups` + `/sports/{id}/leagues?all=false`; 6 sports    | Medium — works, but Selenium token harvest                                           | maintained                | **partial — FACTS** | `sportsbetting/bookmakers/pinnacle.py` (`parse_sport_pinnacle()` quotes `/sports/{id}/leagues?all=false`, sport ids 27/19/18) | Selenium opens pinnacle.com to grab X-API-KEY (browser, not login)            | **none** (read-only; no placement)              | **reference-only (league-enum fact; do NOT adopt Selenium token path)** |
| **pdmuinck/positive-value-trading**                                                                                       | Free guest-API bash scraper                                      | low / old                     | curl `/leagues/{id}/matchups`, league-id map                                     | Low — bash, **hardcoded guest key in file**                                          | stale                     | **partial — FACTS** | `backend/scrapers/events/pinnacle` (`get_events()` quotes league ids 1842/2196/1980/2436/2036/1817)                           | **commits a guest x-api-key in plaintext** — take pattern only, never the key | **none**                                        | **reference-only (league-id pattern; never copy the committed key)**    |
| **NDGopher/UnifiedBetting2**                                                                                              | Free guest-API event probe (Py)                                  | low / pushed 2026             | GET `/0.1/events/{event_id}?brandId=0`                                           | Low — debug script                                                                   | active                    | no                  | `backend/check_pinnacle_odds.py` (`check_pinnacle_odds()` quotes `/events/{id}` shape)                                        | none                                                                          | **none**                                        | **reference-only (per-event endpoint shape; not coverage-widening)**    |
| **iliyasone/ps3838api**                                                                                                   | **PAID** PS3838/Pinnacle API + **bet placement** (Py)            | ~mid / pushed 2026-03         | `place_straight_bet`, `get_client_balance`, login/password                       | Medium (typed)                                                                       | active                    | **NO**              | `README.md` (`place_straight_bet(...)`, `PinnacleClient(login=…, password=…)`)                                                | **stored betting credentials, login flow**                                    | **HIGH — places bets**                          | **REJECT (bet placement + auth; violates every constraint)**            |
| **ZAABY/Betting-Project-Using-Arb-Finder**                                                                                | Arb finder, **auto-places bets** (Py)                            | low / 2024                    | PS3838 API + Selenium, "places bets automatically"                               | n/a                                                                                  | stale                     | **NO**              | (desc) "places bets automatically"                                                                                            | login + automation                                                            | **HIGH — auto-bet**                             | **REJECT (auto-betting)**                                               |
| **benrules2/losemoneybot**                                                                                                | Pinnacle API **bet placement** demo (Py)                         | low / 2016                    | "interaction with Pinnacle Sports API to place a bet"                            | n/a                                                                                  | dead                      | **NO**              | (desc) place a bet                                                                                                            | paid account + placement                                                      | **HIGH**                                        | **REJECT (bet placement)**                                              |
| **rozzac90/pinnacle**                                                                                                     | Paid `api.pinnacle.com` v1 wrapper (Py)                          | ~mid / repo 2017, pushed 2022 | full v1 API incl. place-bet endpoints, HTTP Basic auth                           | Medium                                                                               | stale                     | **NO**              | `pinnacle/endpoints/baseendpoint.py` (`request()` uses `auth=self.client.auth`)                                               | **HTTP Basic = paid account creds**; placebet endpoints exist                 | **HIGH (lib exposes placement)**                | **REJECT (auth-gated paid API; not free)**                              |
| dpitkevics / wataruoguchi / chris-moreton / JulienSansot / zeroX-tj / gentoku / zhoudaqing / armandsar/pineapple / imbolc | Paid v1 `api.pinnacle.com` wrappers (PHP/Node/Java/Rust/Laravel) | low / 2015–2017, dead         | thin wrappers over the **paid** authenticated v1 API                             | n/a                                                                                  | dead                      | **NO**              | (not inspected individually — same class)                                                                                     | paid account creds                                                            | placement endpoints exposed                     | **REJECT (paid/auth; surfaced-not-inspected, never recommended)**       |
| **pinnapi/pinnapi**, **Sharp-API/SharpAPI-TS**, RapidAPI-Pinnacle, BetNet, football-odds-api                              | 3rd-party **paid** Pinnacle-odds resellers (own API key)         | low–mid / 2026                | resell Pinnacle odds via their paid API + key                                    | n/a                                                                                  | active                    | **NO**              | (vendor SDKs)                                                                                                                 | their API key, paid tiers                                                     | none (read), but **not free / not first-party** | **REJECT (paid 3rd-party; defeats free-first policy)**                  |

### Inspect-worthy gate notes

- Inspected in full (quoted ≥1 function): eltonaguiar archive, theoam060 tracker,
  pretrehr, pdmuinck, NDGopher, iliyasone (README), rozzac90.
- **Surfaced but not individually inspected** (same paid-v1-wrapper class as
  rozzac90, all dead 2015–2017, all auth-gated → ineligible for free coverage
  regardless): dpitkevics, wataruoguchi, chris-moreton, JulienSansot, zeroX-tj,
  gentoku, zhoudaqing, armandsar/pinnacle-pineapple, imbolc. Listed, never
  recommended.

---

## Files opened (with one-line descriptions)

- `app/ingestion/pinnacle_arcadia.py` (ours) — current capture: 4 hardcoded sport
  ids, sport-scoped `/sports/{id}/matchups` + `/markets/straight`, period-0 main
  lines only; `startTime` for kickoff.
- `pretrehr/Sports-betting:sportsbetting/bookmakers/pinnacle.py` — `parse_sport_pinnacle()`
  proves `GET /0.1/sports/{id}/leagues?all=false` (league enumeration) + per-league
  endpoints; extra sport ids rugby=27, hockey=19, handball=18.
- `steveo1287/SharkEdge:docs/FREE_ODDS_STRATEGY.md` — community audit naming the
  Pinnacle guest API "free, no auth required" and the standard free Pinnacle
  source; coverage gap framed as limited sport/league enumeration, not endpoint.
- `theoam060-web/Pinnacle-Odds-Tracker:scripts/src/pinnacle-market-coverage.ts` —
  `main()` reads `https://www.pinnacle.com/config/app.json` for `apiKey`+`guestRoot`,
  then enumerates ALL active `/sports` (filters `isHidden`, `matchupCount>0`).
- `pdmuinck/positive-value-trading:backend/scrapers/events/pinnacle` — `get_events()`
  curls `/0.1/leagues/{league_id}/matchups`; concrete EPL/La Liga/etc league ids;
  (anti-pattern: commits a guest key in plaintext).
- `eltonaguiar/…archive:tools/pinnacle_anchor_scrape.py` — `fetch_league()` uses
  league-scoped endpoints with `?brandId=0`, reads kickoff from period-0 `cutoffAt`,
  carries sport ids NBA=4/NHL=19/MLB=3/NFL=15 and league ids 487/1456/246/889.
- `NDGopher/UnifiedBetting2:backend/check_pinnacle_odds.py` — `check_pinnacle_odds()`
  shows the per-event endpoint `GET /0.1/events/{event_id}?brandId=0`.
- `iliyasone/ps3838api:README.md` — `place_straight_bet()` / `PinnacleClient(login,
password)` against `api.ps3838.com` (paid, auth, placement → rejected).
- `rozzac90/pinnacle:pinnacle/endpoints/baseendpoint.py` — `request()` with
  `auth=self.client.auth` (paid v1 API, HTTP Basic → rejected for free coverage).

---

## Why the 34-fixture COVERAGE gap is OURS, not the endpoint's

The arcadia guest API is the **ceiling** of free Pinnacle data — no inspected repo
reaches further on free. Our gap is structural on the consumer side:

1. **Only 4 sports captured.** `SPORT_IDS` is `{soccer, tennis, basketball,
american_football}`. Any pick on **MLB (sport_id=3), NHL (19), rugby (27),
   handball (18)** has zero chance of a Pinnacle archive event. Dynamic `/sports`
   enumeration removes the hardcoded ceiling.
2. **Sport-scoped matchups can omit lower-tier leagues** that the sport-wide dump
   de-prioritizes; the **league-scoped** endpoint guarantees a target competition is
   pulled even when it's not in the top sport feed.
3. **Brittle key handling.** We rely on an optional/empty guest key; if Pinnacle
   tightens the unauthenticated path, capture silently degrades to the documented
   "non-JSON body → scrape gap." `/config/app.json` gives a public, no-login refresh
   path for the current guest key, hardening reliability.

Closing (1) alone — capturing whatever sports our picks actually span — is the
direct lever on the 34. (2) and (3) are reliability hardening.

---

## Adoption plan (facts only — no code copied, license untouched)

All changes are **pure additive, read-only GET** extensions of the existing
`PinnacleArcadiaClient` / `PinnacleArcadiaCapture`. They place no bet, store no
credential, drive no browser, and bypass no anti-bot control — doctrine-clean.

1. **Dynamic sport enumeration (primary coverage win).**
   - Add `fetch_sports()` → `GET /0.1/sports`; keep only `isHidden == false` and
     `matchupCount > 0`. Drive capture off the live sport list (still namespaced
     `pinnacle_<sport>`), or at minimum extend `SPORT_IDS` to include the sports our
     picks actually span (MLB=3, hockey=19, rugby=27, handball=18) gated by config.
   - Per-sport isolation in `capture_once` already protects against one sport's gap
     aborting the rest — this scales cleanly.

2. **Optional public key discovery (reliability).**
   - Add a best-effort `GET https://www.pinnacle.com/config/app.json`; if it returns
     `api.haywire.apiKey` / `...guestRoot` / `apiVersion`, use them to populate the
     guest key + base URL at composition root. Treat as optional: on failure, fall
     back to the current empty key + `DEFAULT_BASE_URL` (today's behaviour). The key
     is a public web-client constant (authenticates no user) — same status the
     module already documents — and must still **never be logged or committed**
     (existing secret-hygiene rules apply unchanged).

3. **League-scoped fallback (optional, targeted).**
   - If specific competitions remain thin after (1), add `fetch_league_matchups(
league_id)` / `fetch_league_markets(league_id)` against `/0.1/leagues/{id}/…`
     plus `/0.1/sports/{id}/leagues?all=false` to resolve ids. Lower priority — only
     if dynamic-sport capture doesn't close the gap.

4. **Precision nicety (optional).** Prefer period-0 `cutoffAt` over `startTime` for
   the kickoff timestamp when present (truer betting cutoff → tighter "latest
   pre-kickoff row IS the close" selection in `closing_odds_from_snapshots`).

**Not adopted:** Selenium token harvest (pretrehr), any committed guest key
(pdmuinck), and every paid/auth/placement repo. No code is copied from any repo, so
no third-party license is engaged.

## Doctrine compliance check (all adopted facts)

- READ-ONLY GET only — yes (all are GET).
- No bet placement / order submission — yes (none of the adopted facts touch
  placement; all placement repos REJECTED).
- No bookmaker login / stored betting creds/cookies/session — yes (guest key is a
  public constant, no account, no login; Selenium token path explicitly NOT adopted).
- No anti-bot/Cloudflare/captcha bypass — yes (plain GET to a public JSON API and a
  public config file).
- Facts/patterns only, never committed secrets — yes (the one repo that commits a
  key is flagged; we never copy it).
- Pure additive read-only ingestion — yes.
