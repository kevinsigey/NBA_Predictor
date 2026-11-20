"""NBA props projection shrink (C2): linear recalibration + C1 tails, prereg
docs/research/2026-11-19-nba-props-shrink-prereg.md.

    PYTHONPATH=src:scripts uv run python scripts/nba_props_shrink.py
    PYTHONPATH=src:scripts uv run python scripts/nba_props_shrink.py --selfcheck
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from nba_props_tails import (
    build_eligible, half_of, is_training_eligible, p_c1, training_rows,
)
from nba_props_vs_kalshi import cluster_ci, decide, fee_c, ll, SERIES


def ols_fit(rows: list) -> tuple:
    n = len(rows)
    if n == 0:
        return 0.0, 1.0
    sx = sum(r["predicted"] for r in rows)
    sy = sum(r["actual"] for r in rows)
    sxx = sum(r["predicted"] * r["predicted"] for r in rows)
    sxy = sum(r["predicted"] * r["actual"] for r in rows)
    denom = n * sxx - sx * sx
    if denom == 0:
        b = 0.0
    else:
        b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def p_c2(pred: float, half: float, line: float, train_rows: list) -> float:
    a, b = ols_fit(train_rows)
    pred_shrunk = a + b * pred
    zs = []
    for r in train_rows:
        pred_tr = a + b * r["predicted"]
        zs.append((r["actual"] - pred_tr) / r["half"])
    k = sum(1 for z in zs if pred_shrunk + z * half > line)
    n = len(train_rows)
    return (k + 0.5) / (n + 1)


FIELDS = ["ticker", "game_id", "game_date", "stat", "line", "predicted", "pred_shrunk",
          "a", "b", "half", "p_c1", "p_c2", "n_train", "mid", "outcome_yes",
          "ll_c1", "ll_c2", "ll_market", "side_c2", "pnl_c2"]

OUT = Path("docs/research/2026-11-19-nba-props-shrink-rows.csv")


def run() -> int:
    rows, counts = build_eligible()
    test_rows = [r for r in rows if r["game_date"] >= "2026-10-27"]

    out_rows = []
    skipped = set()
    for r in test_rows:
        stat = r["stat"]
        before = r["game_date"]
        tr = training_rows(stat, before)
        n = len(tr)
        if n < 500:
            skipped.add((stat, before))
            continue
        pred = r["predicted"]
        lo = r["interval_low"]
        hi = r["interval_high"]
        line = r["line"]
        h = half_of(pred, lo, hi)
        a, b = ols_fit(tr)
        pred_shrunk = a + b * pred
        p_raw_c1 = r["_p"]
        mid = r["_mid"]
        outcome = r["outcome_yes"]
        # p_C1 exactly as tails does
        zs_c1 = [(rr["actual"] - rr["predicted"]) / rr["half"] for rr in tr]
        pc1 = p_c1(pred, h, line, zs_c1)
        pc2 = p_c2(pred, h, line, tr)

        ll_c1v = ll(pc1, outcome)
        ll_c2v = ll(pc2, outcome)
        ll_mkt = ll(mid, outcome)

        side_c2, _ = decide(pc2, r["yes_ask"], r["no_ask"], mid)
        side_str = side_c2 if side_c2 else ""
        if side_c2:
            ask = r["yes_ask"] if side_c2 == "yes" else r["no_ask"]
            won = outcome if side_c2 == "yes" else 1 - outcome
            pnl = 100 * won - 100 * ask - fee_c(100 * ask)
        else:
            pnl = None

        out_rows.append({
            "ticker": r["ticker"], "game_id": r["game_id"], "game_date": before,
            "stat": stat, "line": line, "predicted": pred,
            "pred_shrunk": round(pred_shrunk, 6), "a": round(a, 6), "b": round(b, 6),
            "half": round(h, 6), "p_c1": round(pc1, 6), "p_c2": round(pc2, 6),
            "n_train": n, "mid": round(mid, 4), "outcome_yes": outcome,
            "ll_c1": round(ll_c1v, 6), "ll_c2": round(ll_c2v, 6),
            "ll_market": round(ll_mkt, 6), "side_c2": side_str,
            "pnl_c2": "" if pnl is None else round(pnl, 3),
            "_ll_c1": ll_c1v, "_ll_c2": ll_c2v, "_ll_market": ll_mkt,
            "_p_c2": pc2, "_mid": mid, "_pnl_c2": pnl,
            "_a": a, "_b": b,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    games = {r["game_id"] for r in out_rows}
    skipped_list = sorted(list(skipped))
    print("counts:")
    print(json.dumps({
        "test_contracts": len(out_rows),
        "games": len(games),
        "stat_days_skipped_under_500": len(skipped),
        "skipped": skipped_list,
        "base_counts": counts,
    }, indent=2))

    def fmt(t):
        if t[0] is None:
            return "n/a"
        return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"

    # pooled delta LL(p_C1) - LL(p_C2)
    print("\npooled delta LL(p_C1) - LL(p_C2), clustered:")
    print(fmt(cluster_ci(out_rows, lambda r: r["_ll_c1"] - r["_ll_c2"])))
    print("\nper stat delta LL(p_C1) - LL(p_C2):")
    for stat in SERIES.values():
        sub = [r for r in out_rows if r["stat"] == stat]
        print(f"  {stat}: n={len(sub)} " + fmt(cluster_ci(sub, lambda r: r["_ll_c1"] - r["_ll_c2"])))

    # MAE delta on every test-day grading row
    print("\nMAE(predicted) - MAE(pred'), test-day grading rows:")
    mae_rows = []
    schedule = json.load(open("data/public_snapshot.json"))["schedule"]
    for g in schedule:
        gd = g.get("game_date")
        if not gd or gd < "2026-10-27":
            continue
        if not g.get("tip_off"):
            continue
        cache_path = Path("data/cache/prop_grading") / f"{g['game_id']}.json"
        if not cache_path.exists():
            continue
        from nba_props_vs_kalshi import grading
        for m in grading(g["game_id"]):
            if not m.get("made_before_tip"):
                continue
            if m.get("interval_low") is None or m.get("interval_high") is None:
                continue
            if m.get("actual") is None:
                continue
            stat = m["stat"]
            before = gd
            tr = training_rows(stat, before)
            if len(tr) < 500:
                continue
            a, b = ols_fit(tr)
            pred = m["predicted"]
            actual = m["actual"]
            pred_shrunk = a + b * pred
            delta = abs(actual - pred) - abs(actual - pred_shrunk)
            mae_rows.append({"game_id": g["game_id"], "stat": stat, "delta": delta})
    print(f"  rows={len(mae_rows)} games={len({r['game_id'] for r in mae_rows})}")
    print("  pooled: " + fmt(cluster_ci(mae_rows, lambda r: r["delta"])))
    for stat in SERIES.values():
        sub = [r for r in mae_rows if r["stat"] == stat]
        print(f"  {stat}: n={len(sub)} " + fmt(cluster_ci(sub, lambda r: r["delta"])))

    # a, b per stat on last test day
    last_day = max(r["game_date"] for r in test_rows)
    print(f"\na, b per stat on last test day ({last_day}):")
    for stat in SERIES.values():
        tr = training_rows(stat, last_day)
        a, b = ols_fit(tr)
        print(f"  {stat}: a={a:.6f} b={b:.6f} n={len(tr)}")

    print("\nLL mid - LL(p_C2), clustered:")
    print("  pooled: " + fmt(cluster_ci(out_rows, lambda r: r["_ll_market"] - r["_ll_c2"])))
    for stat in SERIES.values():
        sub = [r for r in out_rows if r["stat"] == stat]
        print(f"  {stat}: n={len(sub)} " + fmt(cluster_ci(sub, lambda r: r["_ll_market"] - r["_ll_c2"])))

    # 4c betting rerun with p_C2, 1c haircut
    print("\n4c betting rerun with p_C2 (1c haircut):")
    bets = [r for r in out_rows if r["side_c2"]]
    print(f"  bets={len(bets)}")
    print("  pooled net c/contract: " + fmt(cluster_ci(bets, lambda r: r["_pnl_c2"] - 1.0)))
    for stat in SERIES.values():
        sub = [r for r in bets if r["stat"] == stat]
        print(f"  {stat}: bets={len(sub)} net=" + fmt(cluster_ci(sub, lambda r: r["_pnl_c2"] - 1.0)))

    # calibration table by p_C2 bucket
    print("\ncalibration by p_C2 bucket:")
    print("| bucket | n | mean p_c2 | mean mid | YES rate |")
    print("|---|---|---|---|---|")
    buckets = [("<.1", 0.0, 0.1), (".1-.3", 0.1, 0.3), (".3-.5", 0.3, 0.5),
               (".5-.7", 0.5, 0.7), (".7-.9", 0.7, 0.9), (">=.9", 0.9, 1.01)]
    for label, lo, hi in buckets:
        sub = [r for r in out_rows if lo <= r["_p_c2"] < hi]
        if not sub:
            print(f"| {label} | 0 | n/a | n/a | n/a |")
            continue
        mp = sum(r["_p_c2"] for r in sub) / len(sub)
        mm = sum(r["_mid"] for r in sub) / len(sub)
        yr = sum(r["outcome_yes"] for r in sub) / len(sub)
        print(f"| {label} | {len(sub)} | {mp:.3f} | {mm:.3f} | {yr:.3f} |")

    return 0


def selfcheck() -> None:
    # OLS exact points
    pts = [{"predicted": 1, "actual": 3}, {"predicted": 2, "actual": 5}, {"predicted": 3, "actual": 7}]
    a, b = ols_fit(pts)
    assert a == 1 and b == 2, (a, b)
    # p_C2 tiny hand-computed set: z'=0 all, k=3, p=3.5/4
    tr = [{"predicted": 1, "actual": 3, "half": 1.0},
          {"predicted": 2, "actual": 5, "half": 1.0},
          {"predicted": 3, "actual": 7, "half": 1.0}]
    assert p_c2(2.0, 1.0, 2.5, tr) == 3.5 / 4.0
    # leakage guard
    assert is_training_eligible("2026-10-27", "2026-10-27") is False
    assert is_training_eligible("2026-10-26", "2026-10-27") is True
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        raise SystemExit(run())