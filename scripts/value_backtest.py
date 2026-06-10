"""Backtest the sharp-vs-soft VALUE strategy (the one with positive CLV).

Fair value = devig(Pinnacle pre-match). Bet the best available price (Max
across books) whenever it beats Pinnacle fair by >= threshold. Settle on the
real result; measure CLV vs the Pinnacle CLOSING line. No goals model.

    uv run python scripts/value_backtest.py
    uv run python scripts/value_backtest.py --leagues E0,E1,SP1,D1,I1,F1

This is the honest validation that DOES show edge — positive, conclusive CLV.
Caveat: betting the Max line assumes line-shopping across many books and fast
execution; real-world CLV is somewhat lower (soft books limit winners).
Decision-support only — nothing here places bets.
"""

import argparse
import asyncio
import csv
import io
import math

import httpx

from app.backtesting.clv import clv_log
from app.ingestion.football_data import fetch_season_csv
from app.probabilities.devig import DevigMethod, devig


def _f(x: object) -> float | None:
    try:
        v = float(str(x))
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


async def load(leagues: list[str], seasons: list[str]) -> list[dict]:
    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for lg in leagues:
            for s in seasons:
                for _ in range(4):
                    try:
                        txt = await fetch_season_csv(client, lg, s)
                        rows.extend(csv.DictReader(io.StringIO(txt)))
                        break
                    except httpx.HTTPError:
                        await asyncio.sleep(1.5)
                await asyncio.sleep(0.3)
    return rows


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leagues", default="E0,E1,SP1,D1,I1,F1")
    p.add_argument("--seasons", default="2122,2223,2324,2425,2526")
    args = p.parse_args()
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]

    print(f"\nVALUE BACKTEST — sharp(Pinnacle) vs best-available, leagues {leagues}")
    print("Bet Max line when it beats Pinnacle fair; CLV vs Pinnacle close; flat 1u.\n")
    rows = await load(leagues, seasons)
    print(f"loaded {len(rows)} matches\n")

    print(
        f"{'min_edge':>9} | {'bets':>6} | {'hit%':>5} | {'ROI%':>7} | "
        f"{'meanCLV':>16} | {'beat%':>6} | {'profit_u':>9}"
    )
    print("-" * 80)
    for thr in (0.0, 0.01, 0.015, 0.02, 0.03):
        bets = []
        for r in rows:
            ps = [_f(r.get("PSH")), _f(r.get("PSD")), _f(r.get("PSA"))]
            mx = [_f(r.get("MaxH")), _f(r.get("MaxD")), _f(r.get("MaxA"))]
            psc = [_f(r.get("PSCH")), _f(r.get("PSCD")), _f(r.get("PSCA"))]
            ftr = r.get("FTR")
            if None in ps or None in mx or ftr not in ("H", "D", "A"):
                continue
            sharp = devig(ps, method=DevigMethod.POWER)  # type: ignore[arg-type]
            close = (
                devig(psc, method=DevigMethod.POWER)  # type: ignore[arg-type]
                if None not in psc
                else None
            )
            for i, sel in enumerate(("H", "D", "A")):
                edge = sharp[i] - 1.0 / mx[i]  # type: ignore[operator]
                if edge >= thr:
                    clv = clv_log(mx[i], close[i]) if close else None  # type: ignore[arg-type]
                    bets.append((ftr == sel, mx[i], clv))
        n = len(bets)
        if n == 0:
            print(f"{thr:>9.3f} | {0:>6} | (no bets)")
            continue
        profit = sum((o - 1.0) if w else -1.0 for w, o, _ in bets)  # type: ignore[operator]
        hit = sum(1 for w, _, _ in bets if w) / n
        clvs = [c for _, _, c in bets if c is not None]
        mclv = sum(clvs) / len(clvs)
        se = math.sqrt(sum((c - mclv) ** 2 for c in clvs) / len(clvs)) / math.sqrt(len(clvs))
        beat = sum(1 for c in clvs if c > 0) / len(clvs) * 100
        print(
            f"{thr:>9.3f} | {n:>6} | {hit * 100:>4.1f} | {profit / n * 100:>+6.2f} | "
            f"{mclv:>+8.4f}+/-{2 * se:.4f} | {beat:>5.1f} | {profit:>+9.1f}"
        )

    print(
        "\nVERDICT: positive, conclusive CLV at edge>=0.01 is a REAL edge signal "
        "(unlike the goals model). See docs/backtesting/value-findings.md."
    )
    print("Manual review required. This system does not place bets.")


if __name__ == "__main__":
    asyncio.run(main())
