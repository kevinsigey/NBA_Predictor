#!/usr/bin/env python3
"""Why is the props model's live error worse than its holdout error? Reproducible.

From the repo root:  PYTHONPATH=src python scripts/nba_live_gap.py [--espn-fallback DIR ...]

1. Live graded lines: GET {API}/hub/prop-grading/games/{game_id} for every completed game in
   2026-10-03..2026-11-02 (the public post-game review endpoint; 404 = no counted rows).
2. Player-game log: data/history/nba_history.json.gz (games before 2025-10-01) plus ESPN player box
   scores for the schedule from 2025-10-01 on. Cached ESPN files are read where they exist, from the
   repo cache or any --espn-fallback directory (read-only); missing ones are fetched.
3. Causal features: features/prop_model_frame.build_prop_model_frame over the whole log. Every
   feature is shift(1)-based, so a row only sees games before it. The shipped pickles score each
   live player-game. The naive baseline is the same frame's <stat>_roll column: the manifest's
   naive_mae definition (the player's own rolling mean of prior games).
4. Writes data/cache/nba_live_gap/live_lines.csv (per line, gitignored) and
   docs/nba-live-vs-holdout-gap.json (summaries only).

Nothing here touches the VPS store or the served models; it is read-only against the API.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://nba.40-160-91-131.sslip.io"
WINDOW_START, WINDOW_END = "2026-10-03", "2026-11-02"
SEASON_START = "2026-10-01"  # preseason is dropped by regular_and_post_only; first regular game sets the season
HISTORY_CUTOFF = "2025-10-01"
HISTORY_PATH = ROOT / "data" / "history" / "nba_history.json.gz"
MODELS = ROOT / "models"
OUT_JSON = ROOT / "docs" / "nba-live-vs-holdout-gap.json"
OUT_CSV = ROOT / "data" / "cache" / "nba_live_gap" / "live_lines.csv"

# API stat name -> frame column of the actual (threes are fg3m, as in models/prop_holdout.MARKET_TARGETS).
ACTUAL_COL = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "fg3m"}
ROLL_COL = {stat: f"{col}_roll" for stat, col in ACTUAL_COL.items()}
BOOT_RESAMPLES = 2000
FETCH_WORKERS = 4
FETCH_TRIES = 3


# ---------------------------------------------------------------- pure helpers (unit-tested)

def gap_contributions(df: pd.DataFrame, by: str, err_col: str, holdout: float) -> tuple[pd.DataFrame, float]:
    """Additive split of a live MAE gap over the slices of `by`.

    The pooled MAE is the row-weighted mean of the slice MAEs, so
    sum_s (n_s / N) * (mae_s - holdout) == pooled_mae - holdout exactly. Returns the
    per-slice table and that total, so a caller can check the split adds up.
    """
    n_total = len(df)
    table = (
        df.groupby(by, observed=True)[err_col]
        .agg(n="size", mae="mean")
        .reset_index()
    )
    table["share_of_rows"] = table["n"] / n_total
    table["excess_over_holdout"] = table["mae"] - holdout
    table["contribution"] = table["share_of_rows"] * table["excess_over_holdout"]
    return table, float(df[err_col].mean() - holdout)


def minutes_bucket(minutes: pd.Series) -> pd.Series:
    return pd.cut(minutes, [-np.inf, 15, 25, np.inf], labels=["<15 min", "15-24.9 min", "25+ min"], right=False).astype(str)


def games_bucket(gp: pd.Series) -> pd.Series:
    bins = [-1, 0, 4, 9, np.inf]
    return pd.cut(gp, bins, labels=["0 (opener)", "1-4", "5-9", "10+"]).astype(str)


def paired_bootstrap_diff(a_err: np.ndarray, b_err: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a_err - b_err) with a 95% percentile interval over rows (paired). Negative = a is better."""
    diff = np.asarray(a_err, float) - np.asarray(b_err, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(BOOT_RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


# ---------------------------------------------------------------- data

def load_manifest_bands() -> dict[str, dict[str, float]]:
    metrics = json.loads((MODELS / "player_props_manifest.json").read_text())["metrics"]
    return {stat: {"holdout_mae": metrics[stat]["mae"], "naive_mae": metrics[stat]["naive_mae"]} for stat in ACTUAL_COL}


def fetch_live_lines(schedule: list[dict], api: str = API) -> tuple[pd.DataFrame, int, list[str]]:
    game_ids = [
        g["game_id"] for g in schedule
        if g.get("completed") and WINDOW_START <= g["game_date"] <= WINDOW_END
    ]
    # Each call takes ~15s on the live host, so fetch in parallel and keep the raw payloads on disk
    # (gitignored) so a rerun does not hit the API again.
    cache = OUT_CSV.parent / "api"
    cache.mkdir(parents=True, exist_ok=True)

    def one(gid: str):
        """(game_id, payload, status): status is 'ok', 'no_rows' (404) or 'skipped' (3 failed tries)."""
        path = cache / f"{gid}.json"
        if path.exists():
            return gid, json.loads(path.read_text()), "ok"
        for attempt in range(1, FETCH_TRIES + 1):
            try:
                resp = requests.get(f"{api}/hub/prop-grading/games/{gid}", timeout=120)
                if resp.status_code == 404:
                    return gid, None, "no_rows"
                resp.raise_for_status()
                payload = resp.json()
                path.write_text(json.dumps(payload))
                return gid, payload, "ok"
            except (requests.RequestException, ValueError) as err:
                print(f"game {gid} try {attempt}/{FETCH_TRIES} failed: {err}", file=sys.stderr, flush=True)
                if attempt < FETCH_TRIES:
                    time.sleep(2 ** attempt)
        return gid, None, "skipped"

    rows, no_rows, skipped = [], [], []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for gid, payload, status in pool.map(one, game_ids):
            if status == "no_rows":
                no_rows.append(gid)
            elif status == "skipped":
                skipped.append(gid)
            else:
                rows.extend(payload["rows"])
    return pd.DataFrame(rows), len(game_ids), no_rows, skipped


def box_rows(game_id: str, fallback_dirs: list[Path]) -> list[dict]:
    from nba_predictor.data import espn

    own = espn._cache_path("player_boxscore", game_id)
    if not own.exists():
        for directory in fallback_dirs:
            candidate = directory / own.name
            if candidate.exists():
                return json.loads(candidate.read_text())["rows"]
    return espn.get_player_boxscore(game_id)


def player_log(schedule: list[dict], fallback_dirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    from nba_predictor.pipeline.ingest import is_preseason, regular_and_post_only, to_player_training_frame

    with gzip.open(HISTORY_PATH, "rt") as fh:
        history = pd.DataFrame(json.load(fh)["players"])

    games = [
        g for g in schedule
        if g.get("completed") and HISTORY_CUTOFF <= g["game_date"] <= WINDOW_END and not is_preseason(g)
    ]
    import requests as _requests

    boxes, no_box = {}, []
    for game in games:
        try:
            boxes[game["game_id"]] = box_rows(game["game_id"], fallback_dirs)
        except _requests.HTTPError:
            no_box.append(game["game_id"])
    espn_rows = regular_and_post_only(to_player_training_frame(games, boxes))

    log = pd.concat([history, espn_rows], ignore_index=True)
    for col in ("player_id", "game_id", "game_date"):
        log[col] = log[col].astype(str)
    log = log.drop_duplicates(["player_id", "game_id"], keep="last")
    return log.sort_values(["player_id", "game_date"]).reset_index(drop=True), no_box


def scored_frame(log: pd.DataFrame) -> pd.DataFrame:
    from nba_predictor.features.prop_model_frame import build_prop_model_frame
    import pickle

    frame, _ = build_prop_model_frame(log)
    frame = frame.copy()
    for stat in ACTUAL_COL:
        # The repo's own shipped XGBoost artifacts, loaded from the local models/ dir (same as the serving code).
        with open(MODELS / f"player_{stat}_model.pkl", "rb") as fh:
            model = pickle.load(fh)
        fcols = [str(c) for c in model.feature_names_in_]
        frame[f"pred_{stat}"] = model.predict(frame[fcols].astype(float))
    return frame


def live_table(lines: pd.DataFrame, log: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["prior_total"] = log.groupby("player_id").cumcount()
    prev_team = log.groupby("player_id")["team"].shift(1)
    log["new_team"] = prev_team.notna() & (prev_team != log["team"])
    in_season = log["game_date"] >= SEASON_START
    log["gp_season_before"] = np.nan
    log.loc[in_season, "gp_season_before"] = log[in_season].groupby("player_id").cumcount()

    keep = ["game_id", "player_id", "player_name", "team", "game_date", "minutes", "prior_total", "new_team", "gp_season_before"]
    wide = log[keep].merge(
        frame[["game_id", "player_id"] + [f"pred_{s}" for s in ACTUAL_COL] + list(ROLL_COL.values())],
        on=["game_id", "player_id"], how="left",
    )
    wide = wide.merge(log[["game_id", "player_id", "points", "rebounds", "assists", "fg3m"]], on=["game_id", "player_id"])
    long = pd.concat(
        [
            wide.assign(
                stat=stat,
                pickle_pred=wide[f"pred_{stat}"],
                naive=wide[ROLL_COL[stat]],
                espn_actual=wide[ACTUAL_COL[stat]],
            )
            for stat in ACTUAL_COL
        ],
        ignore_index=True,
    )
    keys = long[["game_id", "player_id", "stat", "player_name", "team", "game_date", "minutes", "prior_total",
                 "new_team", "gp_season_before", "pickle_pred", "naive", "espn_actual"]]
    table = lines.copy()
    table["game_id"] = table["game_id"].astype(str)
    table["player_id"] = table["player_id"].astype(str)
    table = table.merge(keys, on=["game_id", "player_id", "stat"], how="left", suffixes=("", "_box"), indicator="box_match")
    table["box_match"] = table["box_match"].eq("both")
    table = table.rename(columns={"predicted": "projected", "baseline": "api_baseline"})
    table["err_served"] = (table["actual"] - table["projected"]).abs()
    table["err_pickle"] = (table["actual"] - table["pickle_pred"]).abs()
    table["err_naive"] = (table["actual"] - table["naive"]).abs()
    table["err_api_baseline"] = (table["actual"] - table["api_baseline"]).abs()
    table["actual_matches_espn"] = (table["actual"] - table["espn_actual"]).abs() < 1e-6
    table["month"] = table["game_date"].str[:7]
    start = pd.Timestamp(WINDOW_START)
    table["week"] = (pd.to_datetime(table["game_date"]) - start).dt.days // 7 + 1
    table["minutes_bucket"] = minutes_bucket(table["minutes"])
    table["games_this_season"] = games_bucket(table["gp_season_before"])
    table["history"] = np.where(table["prior_total"].fillna(0) == 0, "no prior games",
                                np.where(table["new_team"].fillna(False).astype(bool), "new team", "established"))
    table["projection_tercile"] = table.groupby("stat")["projected"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    )
    return table


# ---------------------------------------------------------------- report

def summarise(table: pd.DataFrame, bands: dict) -> dict:
    out: dict = {"per_stat": {}, "slices": {}}
    for stat in ACTUAL_COL:
        sub = table[(table["stat"] == stat) & table["box_match"]]
        common = sub.dropna(subset=["err_served", "err_naive", "err_pickle"])
        holdout = bands[stat]["holdout_mae"]
        d_naive, lo_n, hi_n = paired_bootstrap_diff(common["err_served"], common["err_naive"])
        d_pick, lo_p, hi_p = paired_bootstrap_diff(common["err_served"], common["err_pickle"])
        out["per_stat"][stat] = {
            "lines": int((table["stat"] == stat).sum()),
            "matched_to_box": int(len(sub)),
            "common_rows": int(len(common)),
            "holdout_mae": holdout,
            "holdout_naive_mae": bands[stat]["naive_mae"],
            "live_served_mae": float(sub["err_served"].mean()),
            "live_served_mae_common": float(common["err_served"].mean()),
            "live_naive_mae_common": float(common["err_naive"].mean()),
            "live_pickle_mae_common": float(common["err_pickle"].mean()),
            "live_api_baseline_mae": float(sub["err_api_baseline"].dropna().mean()),
            "served_minus_naive": {"mean": d_naive, "ci95": [lo_n, hi_n]},
            "served_minus_pickle": {"mean": d_pick, "ci95": [lo_p, hi_p]},
            "served_vs_pickle_abs_gap_mean": float((sub["projected"] - sub["pickle_pred"]).abs().mean()),
            "actual_matches_espn_rate": float(sub["actual_matches_espn"].mean()),
        }
        out["slices"][stat] = {}
        for by in ["month", "week", "minutes_bucket", "projection_tercile", "games_this_season", "history"]:
            table_by, total = gap_contributions(sub, by, "err_served", holdout)
            assert abs(table_by["contribution"].sum() - total) < 1e-9
            out["slices"][stat][by] = {
                "total_gap": total,
                "rows": table_by.round(4).to_dict(orient="records"),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--espn-fallback", action="append", default=[], help="read-only dir of cached player_boxscore_*.json")
    parser.add_argument("--api", default=API)
    args = parser.parse_args(argv)

    from nba_predictor import config
    (config.CACHE_DIR / "espn").mkdir(parents=True, exist_ok=True)

    schedule = json.loads((ROOT / "data" / "cache" / "schedule" / "games.json").read_text())
    bands = load_manifest_bands()

    lines, n_games, no_rows, skipped = fetch_live_lines(schedule, args.api)
    log, no_box = player_log(schedule, [Path(p) for p in args.espn_fallback])
    frame = scored_frame(log)
    table = live_table(lines, log, frame)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    result = {
        "window": {"start": WINDOW_START, "end": WINDOW_END, "season_start": SEASON_START},
        "inputs": {
            "completed_games_in_window": n_games,
            "games_without_counted_rows_404": len(no_rows),
            "games_skipped_after_3_failed_tries": len(skipped),
            "skipped_game_ids": skipped,
            "live_lines": int(len(lines)),
            "lines_with_box_score_row": int(table["box_match"].sum()),
            "games_with_no_box_score": len(no_box),
            "log_rows": int(len(log)),
        },
        **summarise(table, bands),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("inputs", "per_stat")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
