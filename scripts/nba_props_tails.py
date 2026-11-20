"""NBA props fatter tails (C1 empirical residuals) walk-forward, prereg
docs/research/2026-11-19-nba-props-tails-prereg.md.

    PYTHONPATH=src python scripts/nba_props_tails.py            # fetch + score
    PYTHONPATH=src python scripts/nba_props_tails.py --selfcheck
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from nba_props_vs_kalshi import build_eligible, cluster_ci, decide, fee_c, ll, SERIES

CACHE_TRAIN: dict = {}
CACHE_ROWS: dict = {}


def training_rows(stat: str, before_date: str) -> list:
    key = (stat, before_date)
    if key in CACHE_ROWS:
        return CACHE_ROWS[key]
    rows = []
    schedule = json.load(open("data/public_snapshot.json"))["schedule"]
    for g in schedule:
        gd = g.get("game_date")
        if not gd or not g.get("tip_off"):
            continue
        if not is_training_eligible(gd, before_date):
            continue
        cache_path = Path("data/cache/prop_grading") / f"{g['game_id']}.json"
        if not cache_path.exists():
            continue
        from nba_props_vs_kalshi import grading
        for m in grading(g["game_id"]):
            if m.get("stat") != stat:
                continue
            if not m.get("made_before_tip"):
                continue
            if m.get("interval_low") is None or m.get("interval_high") is None:
                continue
            if m.get("actual") is None:
                continue
            pred = m["predicted"]
            lo = m["interval_low"]
            hi = m["interval_high"]
            h = half_of(pred, lo, hi)
            if h == 0:
                continue
            rows.append({"predicted": pred, "actual": m["actual"], "half": h})
    CACHE_ROWS[key] = rows
    return rows


def half_of(pred: float, lo: float, hi: float) -> float:
    h = (hi - lo) / 2.0
    if lo == 0 and pred < h:
        h = hi - pred
    return h


def is_training_eligible(game_date: str, before_date: str) -> bool:
    return game_date < before_date


def training_zs(stat: str, before_date: str) -> list:
    key = (stat, before_date)
    if key in CACHE_TRAIN:
        return CACHE_TRAIN[key]
    zs = [(r["actual"] - r["predicted"]) / r["half"] for r in training_rows(stat, before_date)]
    CACHE_TRAIN[key] = zs
    return zs


def p_c1(pred: float, half: float, line: float, train_zs: list) -> float:
    k = sum(1 for z in train_zs if pred + z * half > line)
    n = len(train_zs)
    return (k + 0.5) / (n + 1)


FIELDS = ["ticker", "game_id", "game_date", "stat", "line", "predicted", "half", "p_raw",
          "p_c1", "n_train", "mid", "outcome_yes", "side_c1", "pnl_c1", "ll_raw",
          "ll_c1", "ll_market"]

OUT = Path("docs/research/2026-11-19-nba-props-tails-rows.csv")


def run() -> int:
    rows, counts = build_eligible()
    test_rows = [r for r in rows if r["game_date"] >= "2026-10-27"]

    out_rows = []
    skipped = set()
    for r in test_rows:
        stat = r["stat"]
        before = r["game_date"]
        zs = training_zs(stat, before)
        n = len(zs)
        if n < 500:
            skipped.add((stat, before))
            continue
        pred = r["predicted"]
        lo = r["interval_low"]
        hi = r["interval_high"]
        line = r["line"]
        h = half_of(pred, lo, hi)
        p_raw = r["_p"]
        mid = r["_mid"]
        outcome = r["outcome_yes"]
        pc1 = p_c1(pred, h, line, zs)
        ll_raw = ll(p_raw, outcome)
        ll_c1v = ll(pc1, outcome)
        ll_mkt = ll(mid, outcome)

        side_c1, _ = decide(pc1, r["yes_ask"], r["no_ask"], mid)
        side_str = side_c1 if side_c1 else ""
        if side_c1:
            ask = r["yes_ask"] if side_c1 == "yes" else r["no_ask"]
            won = outcome if side_c1 == "yes" else 1 - outcome
            pnl = 100 * won - 100 * ask - fee_c(100 * ask)
        else:
            pnl = None

        out_rows.append({
            "ticker": r["ticker"], "game_id": r["game_id"], "game_date": before,
            "stat": stat, "line": line, "predicted": pred, "half": round(h, 6),
            "p_raw": round(p_raw, 6), "p_c1": round(pc1, 6), "n_train": n,
            "mid": round(mid, 4), "outcome_yes": outcome, "side_c1": side_str,
            "pnl_c1": "" if pnl is None else round(pnl, 3),
            "ll_raw": round(ll_raw, 6), "ll_c1": round(ll_c1v, 6),
            "ll_market": round(ll_mkt, 6),
            "_ll_raw": ll_raw, "_ll_c1": ll_c1v, "_ll_market": ll_mkt,
            "_p_c1": pc1, "_mid": mid, "_pnl_c1": pnl,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # counts
    games = {r["game_id"] for r in out_rows}
    print("counts:")
    print(json.dumps({
        "test_contracts": len(out_rows),
        "games": len(games),
        "stat_days_skipped_under_500": len(skipped),
        "skipped": sorted(list(skipped)),
        "base_counts": counts,
    }, indent=2))

    # pooled delta LL(raw)-LL(C1)
    def fmt(t):
        if t[0] is None:
            return "n/a"
        return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"

    print("\npooled delta LL(raw) - LL(C1), clustered:")
    print(fmt(cluster_ci(out_rows, lambda r: r["_ll_raw"] - r["_ll_c1"])))

    print("\nper stat delta LL(raw) - LL(C1):")
    for stat in SERIES.values():
        sub = [r for r in out_rows if r["stat"] == stat]
        print(f"  {stat}: n={len(sub)} " + fmt(cluster_ci(sub, lambda r: r["_ll_raw"] - r["_ll_c1"])))

    print("\nLL mid - LL(C1), clustered:")
    print("  pooled: " + fmt(cluster_ci(out_rows, lambda r: r["_ll_market"] - r["_ll_c1"])))
    for stat in SERIES.values():
        sub = [r for r in out_rows if r["stat"] == stat]
        print(f"  {stat}: n={len(sub)} " + fmt(cluster_ci(sub, lambda r: r["_ll_market"] - r["_ll_c1"])))

    # 4c betting rerun with p_C1, 1c haircut
    print("\n4c betting rerun with p_C1 (1c haircut):")
    bets = [r for r in out_rows if r["side_c1"]]
    print(f"  bets={len(bets)}")
    print("  pooled net c/contract: " + fmt(cluster_ci(bets, lambda r: r["_pnl_c1"] - 1.0)))
    for stat in SERIES.values():
        sub = [r for r in bets if r["stat"] == stat]
        print(f"  {stat}: bets={len(sub)} net=" + fmt(cluster_ci(sub, lambda r: r["_pnl_c1"] - 1.0)))

    # calibration table by p_C1 bucket
    print("\ncalibration by p_C1 bucket:")
    print("| bucket | n | mean p_c1 | mean mid | YES rate |")
    print("|---|---|---|---|---|")
    buckets = [("<.1", 0.0, 0.1), (".1-.3", 0.1, 0.3), (".3-.5", 0.3, 0.5),
               (".5-.7", 0.5, 0.7), (".7-.9", 0.7, 0.9), (">=.9", 0.9, 1.01)]
    for label, lo, hi in buckets:
        sub = [r for r in out_rows if lo <= r["_p_c1"] < hi]
        if not sub:
            print(f"| {label} | 0 | n/a | n/a | n/a |")
            continue
        mp = sum(r["_p_c1"] for r in sub) / len(sub)
        mm = sum(r["_mid"] for r in sub) / len(sub)
        yr = sum(r["outcome_yes"] for r in sub) / len(sub)
        print(f"| {label} | {len(sub)} | {mp:.3f} | {mm:.3f} | {yr:.3f} |")

    return 0


def selfcheck() -> None:
    # imported selfcheck sanity
    assert fee_c(50) == 1.75
    # tiny training set: residuals [-1,0,1,2], h=1, pred=10, line 10.5 -> k=2, p=2.5/5
    zs = [-1.0, 0.0, 1.0, 2.0]
    assert p_c1(10.0, 1.0, 10.5, zs) == 2.5 / 5.0
    # leakage guard: row dated ON test day excluded
    assert is_training_eligible("2026-10-27", "2026-10-27") is False
    assert is_training_eligible("2026-10-26", "2026-10-27") is True
    # half rule clipped
    assert half_of(1.0, 0.0, 3.0) == 3.0 - 1.0
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        raise SystemExit(run())