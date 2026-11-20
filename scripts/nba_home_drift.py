"""NBA in-season home-margin drift (R1): walk-forward over season folds.

    PYTHONPATH=src uv run python scripts/nba_home_drift.py
    PYTHONPATH=src uv run python scripts/nba_home_drift.py --selfcheck
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.candidate_race import run_candidate_race
from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_regression
from nba_predictor.models.game_outcome import train_margin_model
from nba_predictor.models.probability import MAE_TO_SIGMA
from nba_predictor.pipeline.history import load_history
from nba_predictor.pipeline.retrain import _nba_season_start_year

SEED = 20261017
ITERS = 4000
CLIP = 0.001
ROWS_OUT = Path("docs/research/2026-11-20-nba-home-drift-rows.csv")
CACHE = Path("data/cache/picks_history.json")
SITE_URL = "https://nba.40-160-91-131.sslip.io/picks/history?limit=100000"


def ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def ll(p, y):
    p = min(max(p, CLIP), 1 - CLIP)
    return -math.log(p if y else 1 - p)


def bootstrap_mean(vals, seed=SEED, iters=ITERS):
    n = len(vals)
    if n == 0:
        return None, None, None
    arr = list(vals)
    mean = sum(arr) / n
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += arr[rng.randrange(n)]
        draws.append(s / n)
    draws.sort()
    lo = draws[int(0.025 * (iters - 1))]
    hi = draws[min(iters - 1, int(0.975 * (iters - 1)))]
    return mean, lo, hi


def fmt(t):
    if t[0] is None:
        return "n/a"
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def load_games_players():
    games, players = load_history("2100-01-01")
    gdf = pd.DataFrame(games)
    gdf["season"] = gdf["game_date"].apply(_nba_season_start_year)
    return gdf, players


def train_fold(train_games, train_players, test_games, all_players):
    train_df, feature_cols = build_training_frame(train_games, players=train_players)
    # candidate race
    race = run_candidate_race(train_df, feature_cols=feature_cols, windows=4)
    margin_candidate = race["winner"].get("margin")
    if margin_candidate is None:
        raise RuntimeError("no margin candidate")
    # walk-forward regression for sigma
    wf_df = train_df.assign(home_margin=train_df["home_pts"] - train_df["away_pts"])
    wf = walk_forward_regression(
        wf_df,
        model_factory=lambda tr: (lambda X: train_margin_model(
            tr[feature_cols], tr["home_margin"], candidate=margin_candidate
        ).predict(X[feature_cols])),
        target="home_margin",
        windows=4,
        date_col="game_date",
    )
    sigma = wf["pooled"]["mae"] * MAE_TO_SIGMA
    # train final model
    train_df2 = train_df.assign(home_margin=train_df["home_pts"] - train_df["away_pts"])
    margin_model = train_margin_model(train_df2[feature_cols], train_df2["home_margin"], candidate=margin_candidate)
    # test frame with mature rolling features
    combined = pd.concat([train_games, test_games], ignore_index=True)
    test_players = all_players[all_players["game_id"].isin(set(test_games["game_id"]))]
    train_players_all = all_players[all_players["game_id"].isin(set(train_games["game_id"]))]
    all_p = pd.concat([train_players_all, test_players], ignore_index=True)
    test_frame, _ = build_training_frame(combined, players=all_p)
    test_frame = test_frame[test_frame["game_id"].isin(set(test_games["game_id"]))].reset_index(drop=True)
    preds = margin_model.predict(test_frame[feature_cols])
    # map back to game order
    pred_map = dict(zip(test_frame["game_id"], preds))
    # causality assert
    assert str(train_df["game_date"].max()) < str(test_games["game_date"].min()), (train_df["game_date"].max(), test_games["game_date"].min())
    return pred_map, sigma, margin_candidate


def run() -> int:
    gdf, players = load_games_players()
    seasons = sorted(gdf["season"].unique())
    held_out = seasons[-4:]
    print("seasons (held-out):", [int(s) for s in held_out])

    all_rows = []  # per-game rows for CSV
    # per-season collections
    season_results = {}
    for S in held_out:
        test_games = gdf[gdf["season"] == S].copy().sort_values("game_date").reset_index(drop=True)
        train_games = gdf[gdf["season"] < S].copy().reset_index(drop=True)
        train_players = players[players["game_id"].isin(set(train_games["game_id"]))]
        all_players = players  # full
        pred_map, sigma, cand = train_fold(train_games, train_players, test_games, all_players)
        # build per-game list in date order
        test_games["pred_margin"] = test_games["game_id"].map(pred_map)
        test_games["actual_margin"] = test_games["home_pts"] - test_games["away_pts"]
        test_games = test_games.sort_values("game_date").reset_index(drop=True)
        # compute delta per game
        deltas = []
        game_list = test_games.to_dict("records")
        # precompute cumulative (pred-actual) by date
        # for each game, mean over games with game_date < current date
        pred_arr = test_games["pred_margin"].to_numpy(dtype=float)
        actual_arr = test_games["actual_margin"].to_numpy(dtype=float)
        dates = test_games["game_date"].astype(str).to_numpy()
        # running
        delta_per_game = []
        for i in range(len(test_games)):
            mask = dates < dates[i]
            n_prior = int(mask.sum())
            if n_prior < 100:
                d = 0.0
            else:
                d = float(np.mean(pred_arr[mask] - actual_arr[mask]))
            delta_per_game.append(d)
        test_games["delta"] = delta_per_game
        test_games["margin_r1"] = test_games["pred_margin"] - test_games["delta"]
        test_games["sigma"] = sigma
        test_games["p"] = test_games["pred_margin"].apply(lambda m: float(norm.cdf(m / sigma)))
        test_games["p_r1"] = test_games["margin_r1"].apply(lambda m: float(norm.cdf(m / sigma)))
        test_games["y"] = (test_games["home_pts"] > test_games["away_pts"]).astype(int)

        # LL and MAE
        ll_p = [ll(p, y) for p, y in zip(test_games["p"], test_games["y"])]
        ll_r = [ll(pr, y) for pr, y in zip(test_games["p_r1"], test_games["y"])]
        ll_delta = [a - b for a, b in zip(ll_p, ll_r)]
        mae_base = [abs(pm - am) for pm, am in zip(test_games["pred_margin"], test_games["actual_margin"])]
        mae_r1 = [abs(mr - am) for mr, am in zip(test_games["margin_r1"], test_games["actual_margin"])]
        mae_delta = [a - b for a, b in zip(mae_base, mae_r1)]

        season_results[S] = {
            "n": len(test_games),
            "ll_delta_mean": sum(ll_delta) / len(ll_delta),
            "ll_delta_ci": bootstrap_mean(ll_delta),
            "mae_delta_mean": sum(mae_delta) / len(mae_delta),
            "mae_delta_ci": bootstrap_mean(mae_delta),
            "delta_end": float(delta_per_game[-1]) if delta_per_game else 0.0,
        }

        for _, row in test_games.iterrows():
            all_rows.append({
                "season": S,
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "pred_margin": round(float(row["pred_margin"]), 6),
                "actual_margin": round(float(row["actual_margin"]), 6),
                "delta": round(float(row["delta"]), 6),
                "sigma": round(float(sigma), 6),
                "p": round(float(row["p"]), 6),
                "p_r1": round(float(row["p_r1"]), 6),
                "y": int(row["y"]),
            })

    # pooled
    pooled_ll_delta = []
    pooled_mae_delta = []
    for S in held_out:
        # recompute per-game deltas from all_rows for this season
        pass
    # Easier: collect per-game deltas during loop. Rebuild from rows? We have ll_delta not stored. Let's store.
    # We'll recompute by re-running? Instead, collect in lists.
    # (Handled below by re-deriving from all_rows is not enough.) Let's instead
    # accumulate during loop.
    # We need to redo: accumulate.
    # Since we didn't accumulate, let's recompute from all_rows? all_rows lacks ll deltas.
    # Rework: accumulate lists.
    # (This function structure is getting complex; instead accumulate now by
    # re-running fold? Too slow.) We'll instead collect during the loop above.
    # Let's refactor: accumulate ll_delta_all and mae_delta_all.
    # NOTE: the above loop didn't accumulate. We'll patch by re-running? No.
    # Instead, restructure: accumulate.
    # We'll just recompute deltas from rows? Need pred_margin etc. all_rows has them.
    # Recompute deltas from all_rows:
    pooled_ll_delta = []
    pooled_mae_delta = []
    # Need per-game p, p_r1, y, pred_margin, actual_margin, sigma, delta
    # all_rows has pred_margin, actual_margin, delta, sigma, p, p_r1, y
    for r in all_rows:
        pm = r["pred_margin"]; am = r["actual_margin"]; dlt = r["delta"]; sig = r["sigma"]
        pr = pm - dlt
        pp = float(norm.cdf(pm / sig))
        pp_r = float(norm.cdf(pr / sig))
        y = r["y"]
        pooled_ll_delta.append(ll(pp, y) - ll(pp_r, y))
        pooled_mae_delta.append(abs(pm - am) - abs(pr - am))

    # primary pooled LL
    print("primary pooled LL(p) - LL(p'), game bootstrap:")
    print(fmt(bootstrap_mean(pooled_ll_delta)))
    print("\nper-season LL delta:")
    for S in held_out:
        # recompute per-season ll deltas from all_rows
        srows = [r for r in all_rows if r["season"] == S]
        # need per-game ll delta; recompute
        deltas = []
        for r in srows:
            pm = r["pred_margin"]; am = r["actual_margin"]; dlt = r["delta"]; sig = r["sigma"]; y = r["y"]
            pp = float(norm.cdf(pm / sig)); pr = float(norm.cdf((pm - dlt) / sig))
            deltas.append(ll(pp, y) - ll(pr, y))
        print(f"  {S}: n={len(srows)} " + fmt(bootstrap_mean(deltas)))
    print("\nmargin MAE(base) - MAE(margin') pooled:")
    print(fmt(bootstrap_mean(pooled_mae_delta)))
    print("\nper-season MAE delta:")
    for S in held_out:
        srows = [r for r in all_rows if r["season"] == S]
        deltas = []
        for r in srows:
            pm = r["pred_margin"]; am = r["actual_margin"]; dlt = r["delta"]
            deltas.append(abs(pm - am) - abs((pm - dlt) - am))
        print(f"  {S}: n={len(srows)} " + fmt(bootstrap_mean(deltas)))
    print("\ndelta at season end:")
    for S in held_out:
        print(f"  {S}: {season_results[S]['delta_end']:+.4f}")
    print("\ncounts:")
    print(json.dumps({"seasons": [int(s) for s in held_out], "games": len(all_rows), "per_season": {str(S): int(season_results[S]["n"]) for S in held_out}}, indent=2))

    # write CSV
    ROWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ROWS_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["season", "game_id", "game_date", "pred_margin", "actual_margin", "delta", "sigma", "p", "p_r1", "y"])
        w.writeheader()
        for r in all_rows:
            pm = r["pred_margin"]; am = r["actual_margin"]; dlt = r["delta"]; sig = r["sigma"]
            w.writerow({
                "season": r["season"],
                "game_id": r["game_id"],
                "game_date": r["game_date"],
                "pred_margin": r["pred_margin"],
                "actual_margin": r["actual_margin"],
                "delta": r["delta"],
                "sigma": r["sigma"],
                "p": r["p"],
                "p_r1": r["p_r1"],
                "y": r["y"],
            })

    # Descriptive 2026-27
    print("\n--- descriptive 2026-27 served ---")
    descriptive_2027()

    return 0


def descriptive_2027():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        import urllib.request
        with urllib.request.urlopen(SITE_URL) as resp:
            data = resp.read()
        CACHE.write_bytes(data)
    data = json.loads(CACHE.read_text())
    rows = data["rows"]
    # last row per game created before tip_off
    schedule = json.load(open("data/public_snapshot.json"))["schedule"]
    tip = {g["game_id"]: g.get("tip_off") for g in schedule}
    outcomes = {g["game_id"]: g for g in schedule}
    # group rows by game
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game_id"], []).append(r)
    served = []
    for gid, rs in by_game.items():
        t = tip.get(gid)
        if not t:
            continue
        tip_dt = ts(t)
        cand = [r for r in rs if ts(r["created_at"]) < tip_dt]
        if not cand:
            continue
        last = max(cand, key=lambda r: ts(r["created_at"]))
        g = outcomes.get(gid)
        if not g or not g.get("completed"):
            continue
        if g.get("home_pts") is None or g.get("away_pts") is None:
            continue
        served.append((gid, g, last))
    # sort by game_date
    served.sort(key=lambda x: x[1]["game_date"])
    # compute delta per game
    gids = [s[0] for s in served]
    preds = np.array([s[2]["predicted_margin"] for s in served], dtype=float)
    actuals = np.array([s[1]["home_pts"] - s[1]["away_pts"] for s in served], dtype=float)
    dates = np.array([s[1]["game_date"] for s in served])
    sigmas = np.array([s[2]["margin_sigma"] for s in served], dtype=float)
    deltas = []
    for i in range(len(served)):
        mask = dates < dates[i]
        n_prior = int(mask.sum())
        if n_prior < 100:
            d = 0.0
        else:
            d = float(np.mean(preds[mask] - actuals[mask]))
        deltas.append(d)
    deltas = np.array(deltas)
    margins_r = preds - deltas
    ps = np.array([float(norm.cdf(preds[i] / sigmas[i])) for i in range(len(served))])
    prs = np.array([float(norm.cdf(margins_r[i] / sigmas[i])) for i in range(len(served))])
    ys = np.array([1 if s[1]["home_pts"] > s[1]["away_pts"] else 0 for s in served])
    ll_delta = [ll(ps[i], ys[i]) - ll(prs[i], ys[i]) for i in range(len(served))]
    print(f"  games={len(served)}")
    print("  LL delta: " + fmt(bootstrap_mean(ll_delta)))


def compute_delta(preds, actuals, dates, i):
    mask = [dates[j] < dates[i] for j in range(len(preds))]
    n_prior = sum(mask)
    if n_prior < 100:
        return 0.0
    return float(np.mean([preds[j] - actuals[j] for j in range(len(preds)) if mask[j]]))


def selfcheck() -> None:
    # 99 prior games -> delta 0
    preds = [1.0] * 100
    actuals = [0.0] * 100
    dates = [f"2020-01-{i+1:02d}" for i in range(99)] + ["2020-04-01"]
    assert compute_delta(preds, actuals, dates, 99) == 0.0
    # 100 prior games -> equals mean
    preds2 = [1.0] * 101
    actuals2 = [0.0] * 101
    dates2 = [f"2020-01-{i+1:02d}" if i < 31 else f"2020-02-{i-30:02d}" for i in range(100)] + ["2020-06-01"]
    assert sum(1 for d in dates2 if d < dates2[100]) == 100
    assert abs(compute_delta(preds2, actuals2, dates2, 100) - 1.0) < 1e-9
    # same date not included
    dates3 = ["2020-01-01", "2020-01-01"]
    mask = [d < dates3[1] for d in dates3]
    assert mask[0] is False
    # Phi(0)=0.5 -> p=0.5 for margin 0
    assert abs(float(norm.cdf(0.0 / 1.0)) - 0.5) < 1e-12
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        raise SystemExit(run())