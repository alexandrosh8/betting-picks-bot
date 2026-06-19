# DESIGN.md — PICKS TERMINAL

A design system + redesign brief for the single-file web dashboard at
`app/api/dashboard.html`. Feed this file **together with the current
`dashboard.html`** to a design tool (Claude's design/artifact mode, or this
repo's working session) to create/edit/optimize the page.

> **Product, in one line:** a _picks-only, manual-betting +EV decision-support_
> dashboard for football/soccer + NBA (tennis/NFL visibility-only). It surfaces
> model-vs-market edges, fractional-Kelly stakes (informational), a 1–5★
> confidence rating, and CLV results. **It never places bets, never logs into a
> bookmaker, never stores credentials.** The user reads picks here and bets
> personally.

---

## 0. How to read this file

This spec has two kinds of rules. Respect the boundary:

- **🔒 LOCKED** — must not change. The data contract, the safety/compliance copy,
  the required content, the single-file/vanilla-JS/offline constraints, and the
  discipline rules. Breaking any of these breaks the app or its compliance
  posture. These are non-negotiable.
- **🎨 FREE** — the design tool's playground. The _entire visual language and
  the page layout/IA_. The token system, type, palette, density, motion, and
  especially **the arrangement of the page are yours to reinvent.** The concrete
  tokens in §3–§8 are a strong, reproducible **default** — adopt them, refine
  them, or replace them, as long as you honor the §9 discipline rules and stay
  inside everything LOCKED.

**The brief:** make this look and feel like a best-in-class, professional
sports-analytics / quant-edge web app — credible to someone who reads odds and
P&L for a living — on **desktop and mobile, both first-class**. Do **not**
reproduce today's flat `header → KPI cards → tabs → table → panels → legend`
stack. **Reimagine the layout.** A left nav rail + right proof rail, a split
pane, a hero-CLV-led page, a command-bar terminal — all welcome. Surprise me,
within the rails.

---

## 1. 🔒 LOCKED — the immutable data contract

The page is a static HTML file **read from disk** by FastAPI (`GET /` →
`Cache-Control: no-store`; it is **not** served via StaticFiles). It must stay a
plain standalone document. A 60-second timer re-fetches **data only** and must
never reload the HTML shell.

### 1.1 Endpoints (keep byte-for-byte compatible — same paths, same query params)

| Call                                                              | Purpose                                               |
| ----------------------------------------------------------------- | ----------------------------------------------------- |
| `GET /picks?limit=200&tier=premium`                               | premium picks (fetched separately)                    |
| `GET /picks?limit=200&tier=volume`                                | shadow/volume picks (merged + sorted client-side)     |
| `GET /games?limit=1000`                                           | Available games feed (visibility-only; no edge gates) |
| `GET /performance`                                                | KPI aggregates + `live_evidence` strata               |
| `GET /health`                                                     | engine status + `poll_interval_seconds`               |
| `GET /resolution/match-rate?days=180`                             | Pinnacle archive coverage (lazy)                      |
| `POST /events/{event_id}/result` · `POST /picks/{pick_id}/result` | manual settle                                         |
| `POST /logout` · `GET /login`                                     | session auth                                          |

Tabs are derived **client-side** from `status` + `starts_at` + `revalidated_at`

- `outcome` — there are **no** tab/status query params. The "verified within"
  window = `max(45min, 3 × poll_interval_seconds)` from `/health` — **never
  hardcode a cadence**.

### 1.2 `/picks` row fields (raw repo dicts — every number is a STRING on the wire)

Parse numerics from strings. The id wire field is `id` (int) and is the path
param for `POST /picks/{id}/result`. Render exactly these (do not drop any the
UI uses today):

`id`, `event_id`, `event` (`"Home vs Away"`), `league`, `starts_at` (ISO or
null → `TBD`), `market`, `selection`, `bookmaker`, `decimal_odds`,
`model_probability`, `fair_probability`, `edge`, `ev`, `confidence`,
`recommended_stake_fraction`, `recommended_stake_amount` (tooltip only),
`reason_summary`, `status` (`alerted`/`settled`/`superseded`),
`tier` (`premium`/`volume`), `value_filter_score` (or null),
`anchor_type` (`pinnacle`/`sharp`/`consensus`/null), `created_at`,
`clv_log` (or null), `beat_close` (bool/null), `current_odds` (or null),
`current_edge` (or null — **Edge column shows `current_edge ?? edge`**;
`current_edge <= 0` pre-kickoff triggers the "value gone" fade; null = not
faded), `closing_odds` (or null → `close X.XX`), `current_bookmaker`,
`revalidated_at` (or null — drives **LIVE vs UNVERIFIED**),
`min_acceptable_odds` (or null → `still +EV down to X.XX`), `outcome` (or null),
`pnl` (or null), and the route-attached
`confidence_rating { level: 1–5, label, reasons[] }`.

### 1.3 Required content/IA (must all exist; you may place/restyle them anywhere)

- **Four tabs** with live counts, e.g. `LIVE (3)`:
  - **LIVE** (default) — open, re-priced within the verified window, pre-kickoff. Bettable now.
  - **UNVERIFIED** — open, pre-kickoff, last re-price older than the window. Awaiting re-confirm.
  - **CLOSED** — open, kicked off, line closed, CLV locked. Per-row **Settle** action.
  - **SETTLED** — has a result. Table swaps to an **8-col RESULTS** layout (outcome, P&L, final CLV — no execution columns, no stars, no settle button).
- **Picks table (LIVE/UNVERIFIED/CLOSED, 11 cols):** `Kickoff · Match · Market · Pick · Book · Odds · Fair / EV · Edge · Confidence · CLV · Status`. **Sortable headers** (caret, `aria-sort`, keyboard, tooltip, persisted to localStorage; default sort = confidence ★ then effective edge).
- **Engine badge** with pulsing dot: `POLLING` / `OFFLINE` / `SERVER ERROR` / `UNRESPONSIVE` (red on failure). **Log out** control.
- **Status chips:** poll-age, verified-within window, per-sport fetched counts (`football N · nba N · tennis N`), per-sport ingestion (snapshots/matches, a `degraded` chip when matches>0 but 0 odds parsed). Variants ok/warn/err/info.
- **Offline banner** (hidden by default; only when `/picks` itself fails; distinguishes SERVER ERROR/UNRESPONSIVE/OFFLINE; shows last-good-data time).
- **KPI set (7, premium-scoped):** Open picks · Avg live edge · Beat close (CLV) · Settled · Record (W-L-P/V) · P&L / ROI · Stake-weighted CLV. Each can go `.stale` with a `STALE`/`NO DATA` flag.
- **Live evidence panel** (`/performance.live_evidence`): strata table `stratum · n · n CLV · sw CLV · beat close · ROI`; strata under n=50 CLV show an explicit insufficient-data state.
- **Available games** (collapsed default): sport filter + search; 5-col table `Kickoff · Match · League · Sport · Coverage` (Match carries an external OddsPortal link in href/title only; `UNVALIDATED` badge; Coverage = `N markets priced`).
- **Pinnacle archive coverage** (collapsed, lazy): purpose line + `Sport · Matched · Rate` (rate colored by 0.5 / 0 thresholds).
- **CLV scorecard** (CLOSED tab only): a headline over closed picks — `CLV ledger — N closed · X% beat the close · mean CLV +Y%.`
- **Status + CLV legend** — defines every status word and the CLV/★ meaning, and is the single home for the picks-only safety framing.

### 1.4 🔒 Safety / compliance copy — keep VERBATIM (do not soften, reword, or remove)

- "Recommended stakes, edges, and EV are informational only — never present betting as guaranteed profit." Stake figures must be labeled **"Informational — never a profit guarantee."**
- **"Consistently positive CLV is the only proof of real edge — not a profit guarantee."**
- CLV framing, exact: **"pick price vs the de-vigged closing line (the vig-free best-of-books closing line). Informational."**
- The picks-only / manual-betting identity: the user reviews picks and bets personally; the system never places bets.
- **"LIVE = still provisional until kickoff."**
- Keep the **UNVALIDATED** (games) and **UNVERIFIED** (picks) badges.

### 1.5 🔒 Hard technical constraints

- **One self-contained `dashboard.html`.** All CSS in one `<style>`, all JS in **one** inline `<script>` with `"use strict"`. Must pass `node --check`.
- **No CDN / no network assets.** Runs fully offline on an Ubuntu VPS. Favicon is an inline `data:` URI. **No web fonts fetched over the network** — local font stacks only.
- **Vanilla JS only.** No React/Vue/build/bundler/npm.
- **XSS rule:** every untrusted string renders via `textContent` / text nodes — **never `innerHTML`** for data. No template-string HTML where API/user data flows.
- **Time:** UTC-aware throughout; **display in Cyprus time (Asia/Nicosia), 24h.**
- **A11y:** WCAG-AA contrast on all text + semantic colors; visible focus; `prefers-reduced-motion` honored; sortable headers keyboard-operable.

---

## 2. 🎨 Core principle (the one signature move)

> **Color carries the data; the chrome stays neutral. Positive value is the
> brightest thing on the page.**

Everything structural is layered near-black slate with hairline borders. The
single saturated accent is reserved for **+EV / positive CLV / beat-the-close** —
the whole product exists to prove edge, so make that the hero signal. A page
where the _numbers_ glow and the _frame_ recedes reads as a serious instrument,
not a marketing site. This is LOCKED in spirit even if you restyle the tokens.

---

## 3. 🎨 Color — default system (reinterpretable; ramps + roles are the pattern)

Exact hex, grouped by **role**. The _meanings_ in "Semantic" are 🔒 LOCKED (they
map to data); the surface/ink ramp and the signature hue are FREE to refine.

### Surfaces (canvas → raised)

| Token         | Hex       | Use                                    |
| ------------- | --------- | -------------------------------------- |
| `--bg`        | `#0a0c10` | page canvas                            |
| `--surface-1` | `#0f1216` | primary panels, table body             |
| `--surface-2` | `#151921` | cards, raised rows, header             |
| `--surface-3` | `#1b212b` | hover, popovers, active tab            |
| `--hairline`  | `#232a35` | 1px borders, dividers, table gridlines |

### Ink (text ramp)

| Token          | Hex       | Use                             |
| -------------- | --------- | ------------------------------- |
| `--ink`        | `#e8edf4` | primary text, key figures       |
| `--ink-muted`  | `#aeb8c6` | secondary text, labels          |
| `--ink-subtle` | `#7c8696` | meta, captions, table sub-lines |
| `--ink-faint`  | `#565f6d` | disabled, empty ★, placeholder  |

### Semantic — 🔒 one meaning each, never decorative

| Token          | Hex       | Meaning (LOCKED)                                                                                                              |
| -------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--pos`        | `#34d399` | +EV, positive edge/CLV, **beat-close**, won. _Doubles as the signature accent — make +EV the most saturated thing on screen._ |
| `--pos-strong` | `#10b981` | emphasis fill / hero CLV                                                                                                      |
| `--neg`        | `#f4525f` | negative edge, value-gone, loss, error                                                                                        |
| `--warn`       | `#f0b429` | stale, thin liquidity, UNVERIFIED, degraded                                                                                   |
| `--info`       | `#38bdf8` | interactive: focus ring, selection, sort caret, links. **Never a value color.**                                               |

**Colorblind safety:** pair every won/beat-close color with a polarity glyph
(`✓` / `✗`) — never rely on red/green alone.

---

## 4. 🎨 Typography

Local stacks only (no network fonts). Distinctive, **not** Inter/Roboto. A
characterful display/grotesk for labels + headings + brand; a **true monospace
for every figure** (odds, %, money, timestamps) so columns align.

```
--font-display: "SF Pro Display","Söhne","Geist","Helvetica Neue",system-ui,sans-serif;
--font-mono:    ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
```

Apply `font-variant-numeric: tabular-nums` **globally**; `font-feature-settings:
"tnum" 1, "cv01" 1` on figure cells. Every odds/%/money/time value uses
`--font-mono`.

### Type scale (token | px | weight | tracking)

| Token     | Size | Weight | Letter-spacing       | Use                                     |
| --------- | ---- | ------ | -------------------- | --------------------------------------- |
| `display` | 32   | 600    | -0.5px               | hero CLV number                         |
| `h1`      | 22   | 600    | -0.3px               | section titles, brand                   |
| `h2`      | 16   | 600    | -0.2px               | panel headers                           |
| `label`   | 11   | 600    | +0.8px **UPPERCASE** | column heads, chip labels, KPI captions |
| `body`    | 14   | 400    | -0.1px               | descriptions, legend prose              |
| `mono`    | 13   | 500    | 0                    | table figures (odds/edge/CLV)           |
| `mono-sm` | 12   | 500    | 0                    | sub-lines (ML score, anchor)            |
| `micro`   | 10   | 600    | +0.6px UPPERCASE     | badges, status pills                    |

---

## 5. 🎨 Spacing, radius, elevation, motion

**Spacing scale (closed vocabulary, px):** `2 · 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64`. No free-form values.

**Radius:**
| Token | Value | Use |
| --- | --- | --- |
| `xs` | 3px | chips, badges, ★ |
| `sm` | 6px | buttons, inputs, table cells emphasis |
| `md` | 10px | cards, KPI tiles, panels |
| `pill` | 9999px | tabs, status dots, toggles |

**Elevation** (dark-aware shadows + 1px hairline; keep it restrained):
| Level | Shadow | Use |
| --- | --- | --- |
| L0 | none + `1px var(--hairline)` | table rows, inline panels |
| L1 | `0 1px 2px rgba(0,0,0,.4)` | cards, KPI tiles |
| L2 | `0 6px 20px rgba(0,0,0,.5)` | tooltips, popovers, sticky bars |
| L3 | `0 16px 48px rgba(0,0,0,.6)` | modal / confirm dialogs |

**Motion:** durations `120ms` (hover/press), `200ms` (state/tab), `320ms`
(value updates on refresh). Easing `cubic-bezier(.2,0,0,1)`. Animate only
`background-color, border-color, color, box-shadow, opacity, transform`. The
engine dot pulses (~2s). Loading uses a skeleton shimmer. **`@media
(prefers-reduced-motion: reduce)` → disable all non-essential animation.**

---

## 6. 🎨 Component anatomy (states are mandatory; values are the default)

Give every interactive component all states. Examples (restyle freely, keep the
state coverage):

**Tab (segmented control)** — `default`: `--ink-muted` on `--surface-1`;
`hover`: `--ink` ; `active`: `--ink` on `--surface-3`, 2px `--pos` underline or
left-mark; count badge in `--ink-subtle`. Keyboard: arrow-navigable, `aria-selected`.

**Pick row / card** — Odds, Edge, CLV in `--font-mono`, right-aligned,
`tabular-nums`. Edge cell carries a thin progress bar (width ∝ edge, fill
`--pos`/`--neg` by sign). Confidence = 5 stars (`★` filled `--pos`, `☆` empty
`--ink-faint`); hover surfaces `confidence_rating.label` + `reasons[]`. Anchor
chip `PIN`/`SHARP`/`CONS` and an ML-score sub-line sit under the Pick.
`hover`: row bg → `--surface-3`. **Value-gone** (`current_edge <= 0`
pre-kickoff): row at `opacity:.5` + strike affordance, restored on hover.

**Status pill** — LIVE = `--pos` dot + label; UNVERIFIED = `--warn`; CLOSED =
`--ink-subtle` (locked); SETTLED won = `--pos ✓` / lost = `--neg ✗`. **Must stay
on ONE line** (see mobile fix #3).

**KPI tile** — `label` caption, `display`/`mono` value, left accent bar tinted by
sign. `.stale` → dim to `--ink-subtle` + corner `STALE`/`NO DATA` flag.

**CLV scorecard** — the hero. Big `display` CLV %, the beat-close % and N
closed, ending in the verbatim CLV line (§1.4). Treat positive CLV as the
brightest element on the CLOSED tab.

**Button** — primary (settle/confirm): `--pos-strong` fill, `--bg` text, radius
`sm`, `pressed` darkens ~12%, `disabled` → `--surface-2`/`--ink-faint`. Secondary:
hairline outline, `--ink` text. Min target 36px desktop / **44px mobile**.

**Chip** — `micro` label, radius `xs`, tinted bg at ~12% alpha of its semantic
color, text at full semantic color.

---

## 7. 🎨 Layout — FREE, but both viewports are first-class

**Reimagine the IA.** Do not ship today's flat stack. Strong options (pick or
invent): a persistent **left nav rail** (LIVE/CLOSED/SETTLED/Evidence/Games) +
main picks pane + **right proof rail** (CLV hero + KPIs); or a **hero-CLV-led**
single column; or a **split pane**. Whatever you choose, the CLV proof and the
LIVE picks are the two things a user should see first.

The page reorders its DOM on load (`picksFirst()`) so picks sit up top — keep
that intent (picks + controls prominent, secondary panels below, legend last,
**no traditional footer**) regardless of the new layout.

### Responsive breakpoints

| Name       | Width      | Layout                                                    |
| ---------- | ---------- | --------------------------------------------------------- |
| Wide       | ≥1440px    | full multi-pane; widest table; rails expanded             |
| Desktop    | 1024–1439  | multi-pane; rails may narrow to icons                     |
| Tablet     | 768–1023   | rails collapse to a top bar; table scrolls or condenses   |
| **Mobile** | **≤720px** | **table → card grid** (the JS relies on the 720px switch) |

**Mobile (≤720px) — card grid, not a squished table.** Each pick is a card:
match + kickoff header, then a stats strip with **labeled** `FAIR/EV`, `EDGE`,
`CONFIDENCE`, then status + actions. Full-width controls/chips, **≥44px** tap
targets, KPI tiles 2-up, CLV scorecard + legend in card surrounds. Make the card
field-mapping robust (data-driven, not brittle `nth-child`) so it survives
column reordering.

**🔒 Fix these three known mobile defects explicitly:** (1) the Confidence value
lost its label on mobile — **always label it**; (2) the Fair/EV cell had an
oversized gap — **close it**; (3) the LIVE status badge wrapped to two rows —
**keep status on one line**.

---

## 8. 🎨 Micro-interactions

Tasteful only: row hover, engine-dot pulse, skeleton shimmer while loading,
smooth `320ms` value transitions on the 60s refresh (no layout thrash —
re-render in place), value-gone restore-on-hover. No bounce, no gratuitous
motion. Respect reduced-motion.

---

## 9. Discipline rules — Do / Don't (checkable)

**Do**

- Reserve `--pos` for +EV / positive CLV / beat-close / won — make it the page's brightest signal.
- Render every figure in `--font-mono` with `tabular-nums`; right-align numerics.
- Keep semantic colors one-meaning-each; pair won/lost with `✓`/`✗`.
- Give every interactive element hover/active/focus/disabled states and a visible focus ring (`--info`).
- Design desktop **and** mobile deliberately; test the ≤720px card grid.
- Keep all safety copy (§1.4) verbatim and the data bindings (§1.2) intact.

**Don't**

- Don't use Inter/Roboto/system-default as the display face, or purple-gradient-on-white slop.
- Don't color the chrome with semantic hues, or use `--info` for a value.
- Don't render money/odds without `tnum`, or in a proportional font.
- Don't reproduce today's flat header→KPI→table→legend arrangement.
- Don't introduce `innerHTML` for API/user data, web-font fetches, CDNs, or any build step.
- Don't invent or drop a `/picks` field, endpoint, tab, or safety line to make the design cleaner.

---

## 10. Agent / iteration guide

- **Inputs:** this file + the current `app/api/dashboard.html`. Edit the real
  file; don't guess its internals.
- **Output:** one self-contained `dashboard.html` (inline `<style>` + single
  `<script>`, `"use strict"`, passes `node --check`) that drops in 1:1 — same
  endpoints, fields, tabs, regions, safety copy — plus a short note on the type
  pairing, palette/accent, density approach, the LIVE/UNVERIFIED/SETTLED state
  treatment, and the three mobile fixes.
- **Iterate:** to request another pass, re-feed this file + the latest
  `dashboard.html` and name what to push further (e.g. "denser tables",
  "bolder hero", "calmer palette"). The LOCKED sections stay fixed across passes.

## 11. Known gaps (unspecified — flag, don't invent)

- Exact display **face** depends on what's installed on the host; the stack
  degrades gracefully. If a specific face is wanted, it must already be on the
  VPS (no web-font fetch).
- The signature **hue** is open — `--pos` green is the default because green =
  up reads naturally for value; a different accent is fine **if** +EV remains
  the brightest signal and stays distinct from `--warn`/`--neg`.
- Chart/sparkline styling for CLV-over-time is **not** specified (no such
  component exists yet) — if you add one, keep it monochrome with a single
  `--pos`/`--neg` series and document it here.

---

## Reference

Structure modeled on VoltAgent/awesome-design-md conventions (token tables,
role-grouped hex ramps, per-state components, breakpoint table). Closest
proven dark/data-dense exemplars if you want to study one: **Kraken**, **Sentry**
(deepest token coverage), **Linear**, **Binance** (trading-floor density).
This spec is bespoke to PICKS TERMINAL — the §1 contract and §1.4 copy are
project law, not style.
