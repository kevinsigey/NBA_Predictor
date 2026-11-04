"""Season-holdout MAE for the four player-prop regressors (points, rebounds, assists, threes).

Why: models/player_props_manifest.json's per-stat `mae` used to be read as a holdout figure, but the shipped
pickles are fitted on every row and their own in-sample MAE is what the manifest's `in_sample_mae` reports. This
script measures the error a *new* season would have seen, with no leakage:

  * For holdout season S, the model is retrained from scratch on every player-game row from seasons BEFORE S.
  * Its MAE is measured on the rows of season S only.
  * Rolling features are causal (shift(1)), and the frame is truncated at season S before the features are built,
    so no holdout row is ever computed from a game that happened after the holdout season started.
  * Two holdouts are reported: the latest complete season (headline, drives the manifest `mae`) and the season
    before it (a stability check).

Data: the frozen history (data/history/nba_history.json.gz, seasons before 2025-10-01) plus the 2025-26 window
fetched from ESPN (cached under data/cache/espn). The row filters mirror pipeline/ingest.py::train_player_prop_models
(minutes > 0, preseason dropped), so these numbers describe the same population the production writer trains on.

Also reported, not used to decide anything by itself: each retrained model against the SHIPPED pickle on the
latest holdout, as a paired, game-clustered bootstrap. The shipped pickles were fitted on that season, so this
comparison favours them by construction; it is recorded so the pickle-replacement decision is visible.

Usage (from the repo root, single process):
    PYTHONPATH=src python scripts/props_holdout_mae.py            # write docs/props-holdout-mae.{json,md}
    PYTHONPATH=src python scripts/props_holdout_mae.py --write-manifest   # also update the manifest metrics
The shipped pickles are never written by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor.features.prop_model_frame import build_prop_model_frame
from nba_predictor.models.player_props import train_player_stat_model
from nba_predictor.models.prop_holdout import MARKET_TARGETS
from nba_predictor.pipeline import ingest
from nba_predictor.pipeline.history import merge_history

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "player_props_manifest.json"
OUT_JSON = ROOT / "docs" / "props-holdout-mae.json"
OUT_MD = ROOT / "docs" / "props-holdout-mae.md"

WINDOW_START = "2025-10-01"   # first day of the 2025-26 season; the frozen history stops before it
WINDOW_END = "2026-06-30"     # generous end; the schedule fetch keeps only what ESPN lists as played
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261103
SHIPPED_PICKLE = {m: f"player_{m}_model.pkl" for m in MARKET_TARGETS}


def season_start(dates: pd.Series) -> pd.Series:
    """NBA season start year: Oct-Dec belong to the season starting that year, Jan-Sep to the one before."""
    parsed = pd.to_datetime(dates)
    return parsed.dt.year - (parsed.dt.month < 10).astype(int)


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_window_players() -> pd.DataFrame:
    """2025-26 player-game rows, built exactly as the production ingest builds them (ESPN, cached per game)."""
    games = ingest.fetch_schedule_range(WINDOW_START, WINDOW_END)
    boxes = ingest.fetch_player_boxscores(games)
    return ingest.to_player_training_frame(games, boxes)


def load_all_players() -> tuple[pd.DataFrame, dict]:
    """Frozen history + the 2025-26 window, concatenated the same way run_ingest does, then production filters."""
    t0 = time.time()
    window = fetch_window_players()
    info = {"window_fetch_seconds": round(time.time() - t0, 1), "window_player_rows": int(len(window))}
    # merge_history reads the frozen file (game_date < WINDOW_START) and puts it in front of the window rows.
    _, players = merge_history([], window, WINDOW_START)
    players = players[players["minutes"] > 0].copy()
    players = ingest.regular_and_post_only(players).reset_index(drop=True)
    players["season"] = season_start(players["game_date"])
    info["combined_player_rows"] = int(len(players))
    info["seasons"] = {season_label(int(y)): int(n) for y, n in players["season"].value_counts().sort_index().items()}
    return players, info


def assert_complete(players: pd.DataFrame, start_year: int) -> None:
    """The holdout season must have finished: its last game is in late spring of the following calendar year."""
    last = players.loc[players["season"] == start_year, "game_date"].max()
    if pd.isna(last) or str(last) < f"{start_year + 1}-04-01":
        raise SystemExit(f"season {season_label(start_year)} is not complete in the data (last game {last})")


def shipped_predictions(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, np.ndarray | None]:
    """The committed pickles' predictions on the holdout rows, or None where the feature set does not match."""
    out: dict[str, np.ndarray | None] = {}
    for market, filename in SHIPPED_PICKLE.items():
        path = ROOT / "models" / filename
        if not path.exists():
            out[market] = None
            continue
        # Loading the repo's own committed pickles (the same files api/ loads); not untrusted input.
        model = joblib.load(path)
        names = list(getattr(model, "feature_names_in_", []))
        if names and names != feature_cols:
            out[market] = None
            continue
        out[market] = np.asarray(model.predict(frame[feature_cols]), dtype=float)
    return out


def paired_game_bootstrap(diff_by_row: np.ndarray, game_ids: np.ndarray) -> dict:
    """Mean of (retrained error - shipped error) over rows, bootstrapped by game (rows share games, so rows are not
    independent). Returns the point estimate, a 95% interval and the share of resamples where retrained is worse."""
    codes, uniques = pd.factorize(game_ids)
    sums = np.bincount(codes, weights=diff_by_row, minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 200):
        idx = rng.integers(0, len(uniques), size=(min(200, BOOTSTRAP_DRAWS - start), len(uniques)))
        draws[start:start + idx.shape[0]] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_diff": float(diff_by_row.mean()),
        "ci95": [float(lo), float(hi)],
        "share_draws_retrained_worse": float((draws > 0).mean()),
        "games": int(len(uniques)),
    }


def run_holdout(players: pd.DataFrame, start_year: int, with_shipped: bool) -> dict:
    """One season holdout: retrain on seasons < start_year, score season == start_year."""
    label = season_label(start_year)
    assert_complete(players, start_year)
    visible = players[players["season"] <= start_year].drop(columns=["season"]).reset_index(drop=True)
    frame, feature_cols = build_prop_model_frame(visible)
    frame_season = season_start(frame["game_date"])
    train = frame[frame_season < start_year]
    test = frame[frame_season == start_year]
    if len(train) == 0 or len(test) == 0:
        raise SystemExit(f"holdout {label}: {len(train)} train / {len(test)} test rows; nothing to measure")

    shipped = shipped_predictions(test, feature_cols) if with_shipped else {}
    markets: dict[str, dict] = {}
    comparison: dict[str, dict] = {}
    for market, target in MARKET_TARGETS.items():
        model = train_player_stat_model(train[feature_cols], train[target])
        pred = np.asarray(model.predict(test[feature_cols]), dtype=float)
        err = np.abs(pred - test[target].to_numpy(dtype=float))
        in_sample = float(np.abs(np.asarray(model.predict(train[feature_cols]), dtype=float) - train[target]).mean())
        naive = float(np.abs(test[f"{target}_roll"].to_numpy(dtype=float) - test[target].to_numpy(dtype=float)).mean()) \
            if f"{target}_roll" in test.columns else None
        markets[market] = {
            "holdout_mae": float(err.mean()),
            "naive_mae": naive,
            "in_sample_mae": in_sample,
            "generalisation_gap": float(err.mean()) - in_sample,
            "n_train": int(len(train)),
            "n_holdout": int(len(test)),
        }
        if shipped.get(market) is not None:
            shipped_err = np.abs(shipped[market] - test[target].to_numpy(dtype=float))
            comparison[market] = {
                "shipped_mae_on_holdout_rows": float(shipped_err.mean()),
                "retrained_mae_on_holdout_rows": float(err.mean()),
                "retrained_minus_shipped": paired_game_bootstrap(err - shipped_err, test["game_id"].to_numpy()),
                "note": "shipped pickle was fitted on these rows; this comparison favours it by construction",
            }
    return {
        "holdout_season": label,
        "train_seasons": f"{season_label(int(frame_season.min()))} .. {season_label(start_year - 1)}",
        "train_max_date": str(train["game_date"].max()),
        "holdout_min_date": str(test["game_date"].min()),
        "holdout_max_date": str(test["game_date"].max()),
        "markets": markets,
        "retrained_vs_shipped": comparison,
    }


def write_manifest(results: dict) -> None:
    manifest = json.loads(MANIFEST.read_text())
    head = results["holdouts"]["latest"]
    shipped_in_sample = {stat: m.get("in_sample_mae") for stat, m in manifest["metrics"].items()}
    for market in MARKET_TARGETS:
        stat_metrics = manifest["metrics"][market]
        stat_metrics["mae"] = head["markets"][market]["holdout_mae"]
        stat_metrics["naive_mae"] = head["markets"][market]["naive_mae"]
        stat_metrics["in_sample_mae"] = shipped_in_sample[market]
    manifest["training"]["in_sample_metrics"] = False
    manifest["training"]["holdout_season"] = head["holdout_season"]
    manifest["training"]["holdout_method"] = (
        f"mae = season holdout: retrained on {head['train_seasons']}, scored on the {head['holdout_season']} season "
        "(scripts/props_holdout_mae.py, see docs/props-holdout-mae.md). in_sample_mae = the shipped pickles on "
        "their own training rows. The shipped pickles are NOT the models that produced mae; model_version is unchanged."
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def markdown(results: dict) -> str:
    lines = [
        "# Player props: season-holdout MAE",
        "",
        "Generated by `scripts/props_holdout_mae.py`; raw numbers in `docs/props-holdout-mae.json`.",
        "",
        "For each holdout season, the four regressors are retrained from scratch on earlier seasons only and scored on",
        "the holdout season. Rolling features are causal and the frame is truncated at the holdout season first.",
        "`in_sample_mae` is the same kind of retrained model scored on its own training rows, for the gap.",
        "",
    ]
    for key in ("latest", "earlier"):
        h = results["holdouts"][key]
        lines += [
            f"## Holdout {h['holdout_season']} ({'headline' if key == 'latest' else 'stability check'})",
            "",
            f"Trained on {h['train_seasons']}; scored on {h['holdout_min_date']} to {h['holdout_max_date']}.",
            "",
            "| stat | holdout MAE | naive (rolling mean) MAE | in-sample MAE | gap | holdout rows |",
            "|---|---|---|---|---|---|",
        ]
        for market, m in h["markets"].items():
            naive = f"{m['naive_mae']:.4f}" if m["naive_mae"] is not None else "n/a"
            lines.append(
                f"| {market} | {m['holdout_mae']:.4f} | {naive} | {m['in_sample_mae']:.4f} | "
                f"{m['generalisation_gap']:+.4f} | {m['n_holdout']} |"
            )
        lines.append("")
    comp = results["holdouts"]["latest"]["retrained_vs_shipped"]
    if comp:
        lines += [
            "## Retrained vs shipped pickles (latest holdout)",
            "",
            "Paired, game-clustered bootstrap of (retrained error - shipped error). Negative = retrained is better.",
            "The shipped pickles were fitted on these rows, so they are favoured by construction.",
            "",
            "| stat | shipped MAE | retrained MAE | mean diff | 95% CI | share of draws retrained worse |",
            "|---|---|---|---|---|---|",
        ]
        for market, c in comp.items():
            b = c["retrained_minus_shipped"]
            lines.append(
                f"| {market} | {c['shipped_mae_on_holdout_rows']:.4f} | {c['retrained_mae_on_holdout_rows']:.4f} | "
                f"{b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['share_draws_retrained_worse']:.3f} |"
            )
        lines += [
            "",
            "Decision: the shipped pickles are not replaced by this script. The retrained models are not shipped, and",
            "the comparison above cannot show they are no worse, because the shipped pickles have already seen the rows.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-manifest", action="store_true", help="update models/player_props_manifest.json metrics")
    ap.add_argument("--latest-season", default="2025-26", help="headline holdout season label")
    args = ap.parse_args()

    latest_start = int(args.latest_season.split("-")[0])
    earlier_start = latest_start - 1
    started = time.time()
    players, data_info = load_all_players()
    print(f"[{time.strftime('%H:%M:%S')}] data: {data_info}", flush=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "split": "season holdout; train on all seasons before the holdout season, score the holdout season",
            "features": "pipeline features (build_prop_model_frame), causal shift(1); frame truncated at the holdout season",
            "filters": "minutes > 0, preseason dropped (as train_player_prop_models)",
            "models": "train_player_stat_model defaults (XGBRegressor 200 trees, depth 4, lr 0.05)",
            "bootstrap": f"paired, clustered by game, {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}",
            "shipped_pickles_replaced": False,
        },
        "data": data_info,
        "holdouts": {},
    }
    results["holdouts"]["latest"] = run_holdout(players, latest_start, with_shipped=True)
    print(f"[{time.strftime('%H:%M:%S')}] latest holdout done", flush=True)
    results["holdouts"]["earlier"] = run_holdout(players, earlier_start, with_shipped=True)
    print(f"[{time.strftime('%H:%M:%S')}] earlier holdout done", flush=True)
    results["runtime_seconds"] = round(time.time() - started, 1)

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float) + "\n")
    OUT_MD.write_text(markdown(results))
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    if args.write_manifest:
        write_manifest(results)
        print(f"updated {MANIFEST}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
