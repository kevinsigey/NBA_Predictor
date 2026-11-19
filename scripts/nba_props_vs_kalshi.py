"""NBA props vs Kalshi prop prices, run exactly as frozen in
docs/research/2026-11-18-nba-props-vs-kalshi-prereg.md.

    PYTHONPATH=src python scripts/nba_props_vs_kalshi.py            # fetch + score
    PYTHONPATH=src python scripts/nba_props_vs_kalshi.py --selfcheck
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import NormalDist, fmean

import requests

from nba_predictor.pipeline.refresh_kalshi import MATCH_HOURS, _hours_apart

SITE = "https://nba.40-160-91-131.sslip.io"  # the API is served at the root
SERIES = {"KXNBAPTS": "points", "KXNBAREB": "rebounds", "KXNBAAST": "assists", "KXNBA3PT": "threes"}
SPLIT_AT = "2026-11-18T02:44:51+00:00"  # NBA#173 merge (f9b7226)
EDGE_C, SLIP, FEE_C, HAIRCUT_C, Z80 = 4.0, 0.03, 100, 1.0, 1.2816
ITERS, SEED = 4000, 20261017
CACHE = Path("data/cache/prop_grading")
OUT = Path("docs/research/2026-11-18-nba-props-vs-kalshi-rows.csv")


def fee_c(price_c: float) -> float:
    """Per-contract taker fee in cents at C=100, as predictor-hub paper.fees.kalshi_fee_cents."""
    p = max(0, min(100, int(math.floor(price_c + 0.5))))
    return -(-7 * FEE_C * p * (100 - p) // 10000) / FEE_C


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def p_yes(pred: float, lo: float, hi: float, line: float) -> float:
    half = (hi - lo) / 2
    if lo == 0 and pred < half:
        half = hi - pred
    return 1 - NormalDist(pred, half / Z80).cdf(line)


def decide(p: float, yes_ask: float, no_ask: float, mid: float):
    """(side, edge_c) for the side clearing the 4c net edge under the 3c slippage cap, else (None, best edge)."""
    cands = []
    for side, q, ask in (("yes", p, yes_ask), ("no", 1 - p, no_ask)):
        side_mid = mid if side == "yes" else 1 - mid
        if ask - side_mid <= SLIP + 1e-9:
            cands.append((100 * q - 100 * ask - fee_c(100 * ask), side))
    if not cands:
        return None, None
    edge, side = max(cands)
    return (side if edge >= EDGE_C else None), edge


def ll(p: float, y: int) -> float:
    p = min(max(p, 0.001), 0.999)
    return -math.log(p if y else 1 - p)


def cluster_ci(rows, fn):
    """Per-contract mean of fn(row), game-clustered percentile bootstrap (hub constants)."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["game_id"], []).append(fn(r))
    if not groups:
        return None, None, None
    sums = [(sum(g), len(g)) for g in groups.values()]
    est = sum(s for s, _ in sums) / sum(n for _, n in sums)
    rng, keys, draws = random.Random(SEED), len(sums), []
    for _ in range(ITERS):
        pick = [sums[rng.randrange(keys)] for _ in range(keys)]
        draws.append(sum(s for s, _ in pick) / sum(n for _, n in pick))
    draws.sort()
    return est, draws[int(0.025 * (ITERS - 1))], draws[min(ITERS - 1, int(0.975 * (ITERS - 1)))]


def load_kalshi():
    """{ticker: (last pre-snapshot-sorted rows)} for the four served stats."""
    by_ticker: dict[str, list[dict]] = {}
    for path in sorted(glob.glob("data/lines/kalshi_nba_lines_*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("series") in SERIES and r.get("player"):
                    by_ticker.setdefault(r["ticker"], []).append(r)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: r["snapshot_ts"])
    return by_ticker


def grading(game_id: str) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{game_id}.json"
    if not path.exists():
        for attempt in range(4):  # the live site 502s under a cold grading cache; back off, don't hammer it
            resp = requests.get(f"{SITE}/hub/prop-grading/games/{game_id}", timeout=120)
            if resp.status_code < 500:
                break
            time.sleep(10 * (attempt + 1))
        if resp.status_code == 404:
            path.write_text("[]")
        else:
            resp.raise_for_status()
            path.write_text(json.dumps(resp.json()["rows"]))
    return json.loads(path.read_text())


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


FIELDS = ["ticker", "game_id", "tip_off", "player", "stat", "line", "predicted", "interval_low",
          "interval_high", "p_yes", "mid", "yes_ask", "no_ask", "quote_ts", "volume", "side",
          "edge_c", "outcome_yes", "pnl_c", "split", "ll_model", "ll_market"]


def build_eligible() -> tuple:
    schedule = json.load(open("data/public_snapshot.json"))["schedule"]
    by_ticker = load_kalshi()
    counts = {k: 0 for k in ("tickers", "settled", "game_matched", "player_matched", "pre_tip_projection",
                              "interval", "pre_tip_quote", "eligible", "bets")}
    rows = []
    for ticker, snaps in by_ticker.items():
        counts["tickers"] += 1
        final = [r for r in snaps if r["status"] == "finalized" and r["result"] in ("yes", "no")]
        if not final:
            continue
        counts["settled"] += 1
        first = snaps[0]
        teams = first["event_ticker"].split("-", 1)[1][7:]
        games = [g for g in schedule if g.get("tip_off") and teams in (g["away_team"] + g["home_team"], g["home_team"] + g["away_team"])
                 and _hours_apart(first["occurrence"], g["tip_off"]) <= MATCH_HOURS]
        if len(games) != 1:
            continue
        game = games[0]
        counts["game_matched"] += 1
        name, stat = norm_name(first["player"]), first["stat"]
        hits = [g for g in grading(game["game_id"]) if g["stat"] == stat and norm_name(g.get("player_name") or "") == name]
        if len(hits) != 1:
            continue
        counts["player_matched"] += 1
        m = hits[0]
        if not m["made_before_tip"]:
            continue
        counts["pre_tip_projection"] += 1
        if m.get("interval_low") is None or m.get("interval_high") is None:
            continue
        counts["interval"] += 1
        tip = ts(game["tip_off"])
        pre = [r for r in snaps if ts(r["snapshot_ts"]) < tip and r["yes_bid"] and r["yes_ask"] and 0 < r["yes_bid"] < r["yes_ask"] < 1
               and r["no_ask"] is not None]
        if not pre:
            continue
        counts["pre_tip_quote"] += 1
        q = pre[-1]  # frozen: last two-sided pre-tip quote
        mid = (q["yes_bid"] + q["yes_ask"]) / 2
        p = p_yes(m["predicted"], m["interval_low"], m["interval_high"], first["line"])
        side, edge = decide(p, q["yes_ask"], q["no_ask"], mid)
        if edge is None:
            continue
        counts["eligible"] += 1
        y = 1 if final[-1]["result"] == "yes" else 0
        ask = q["yes_ask"] if side == "yes" else q["no_ask"]
        won = (y if side == "yes" else 1 - y) if side else None
        pnl = 100 * won - 100 * ask - fee_c(100 * ask) if side else None
        counts["bets"] += bool(side)
        rows.append({
            "ticker": ticker, "game_id": game["game_id"], "game_date": game["game_date"], "tip_off": game["tip_off"],
            "player": first["player"], "stat": stat,
            "line": first["line"], "predicted": m["predicted"], "interval_low": m["interval_low"], "interval_high": m["interval_high"],
            "p_yes": round(p, 6), "mid": round(mid, 4), "yes_ask": q["yes_ask"], "no_ask": q["no_ask"], "quote_ts": q["snapshot_ts"],
            "volume": q.get("volume"), "side": side or "", "edge_c": round(edge, 3), "outcome_yes": y,
            "pnl_c": "" if pnl is None else round(pnl, 3),
            "split": "pre173" if ts(game["tip_off"]) < ts(SPLIT_AT) else "post173",
            "ll_model": round(ll(p, y), 6), "ll_market": round(ll(mid, y), 6),
            "_p": p, "_mid": mid, "_edge": edge,
        })
    return rows, counts


def run() -> int:
    rows, counts = build_eligible()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(counts))

    def cell(label, sub, haircut):
        bets = [r for r in sub if r["side"]]
        net = cluster_ci(bets, lambda r: float(r["pnl_c"]) - haircut)
        dll = cluster_ci(sub, lambda r: r["ll_market"] - r["ll_model"])
        fmt = lambda t: "n/a" if t[0] is None else f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
        print(f"| {label} | {len(sub)} | {len({r['game_id'] for r in sub})} | {len(bets)} | {fmt(net)} | {fmt(dll)} |")

    for split, hc in (("pre173", HAIRCUT_C), ("post173", 0.0)):
        sub = [r for r in rows if r["split"] == split]
        print(f"\n{split} (haircut {hc}c)\n| Cell | Contracts | Games | Bets | Net c/contract [95%] | LL market - model [95%] |\n|---|---|---|---|---|---|")
        cell("Pooled", sub, hc)
        for stat in SERIES.values():
            cell(stat, [r for r in sub if r["stat"] == stat], hc)
        for side in ("yes", "no"):
            cell(f"{side.upper()} bets", [r for r in sub if r["side"] == side], hc)
        if split == "pre173":
            cell("Pooled, no haircut", sub, 0.0)
    return 0


def selfcheck() -> None:
    assert fee_c(50) == 1.75 and fee_c(1) == 0.07 and fee_c(0) == 0
    assert abs(p_yes(20.0, 13.0, 27.0, 20.0) - 0.5) < 1e-9
    assert abs(p_yes(20.0, 13.0, 27.0, 27.0) - 0.1) < 1e-3  # line at the top of the 80% range -> ~10%
    assert p_yes(1.0, 0.0, 3.0, 0.5) > 0.5  # clipped-at-zero range uses hi - pred
    assert decide(0.70, 0.60, 0.42, 0.59) == ("yes", 70 - 60 - fee_c(60))
    assert decide(0.30, 0.60, 0.42, 0.59)[0] == "no"
    assert decide(0.62, 0.60, 0.42, 0.59)[0] is None  # edge below 4c
    assert decide(0.90, 0.70, 0.45, 0.60) == (None, None)  # both asks > 3c over mid
    assert norm_name("Nikola Jokić") == "nikola jokic" and norm_name("Jaren Jackson Jr.") == "jaren jackson"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        sys.exit(run())