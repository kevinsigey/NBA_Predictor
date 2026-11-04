"""NBA-4: do per-team rest/streak, the 7-day rest cap and the Utah altitude indicator earn their place?

Three arms, the same games, the same walk-forward folds:

* baseline        the production default (role-split rest, 99 sentinel, role-split streak)
* rest_altitude   rest and streak per TEAM, rest capped at 7 with a first-game flag, UTA added to altitude
* +interactions   rest_altitude plus visitor back-to-back x travel miles and back-to-back-at-altitude

Folds: expanding-window walk-forward over the whole chronological frame (4 windows, first 40% of dates
train). Each arm is scored only on out-of-fold predictions.

Metrics, per game: win log-loss and Brier (lower is better), margin absolute error (lower is better).
Pooled point estimates also: win AUC (higher is better), margin MAE and RMSE (lower is better).

Ship rule (strict, from the NBA-4 task): an arm ships only if BOTH primary paired-bootstrap 95% intervals
(win log-loss difference, margin absolute-error difference; candidate minus baseline, 2000 resamples)
exclude zero in the improving direction, AND no reported metric's point estimate is worse than baseline.
Nothing here retrains or writes models/. A shipped arm would still need its own reviewed PR.

Reads data/cache/training/games_all.json (gitignored; 2018-2026 games). Writes docs/nba-4-rest-altitude.json.

Run (single process):
    PYTHONPATH=src uv run --python 3.13 python tools/nba4_rest_altitude_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    walk_forward_metrics,
    walk_forward_regression,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

CACHE = ROOT / "data" / "cache" / "training" / "games_all.json"
OUT = ROOT / "docs" / "nba-4-rest-altitude.json"
WINDOWS = 4
N_RESAMPLES = 2000
SEED = 20261103
CLIP = 1e-12

ARMS = {
    "baseline": {},
    "rest_altitude": {"rest_altitude": True},
    "+interactions": {"rest_altitude": True, "rest_altitude_interactions": True},
}


def _per_game_win(y, p):
    p = np.clip(p, CLIP, 1 - CLIP)
    log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"log_loss": log_loss, "brier": (p - y) ** 2}


def _per_game_margin(y, p):
    err = p - y
    return {"abs_error": np.abs(err)}


def _paired_mean_ci(diff, seed=SEED):
    """95% percentile interval of the mean per-game difference (candidate - baseline), resampling games."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        means[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _run_arm(frame_all, cols):
    """Walk-forward win and margin for one arm. Returns pooled arrays keyed by game_id."""
    frame = frame_all.sort_values("game_date").reset_index(drop=True)
    win = walk_forward_metrics(
        frame, model_factory=lambda tr: default_candidates(cols)["logistic"]["win"](tr), windows=WINDOWS,
    )
    margin = walk_forward_regression(
        frame, model_factory=lambda tr: default_candidates(cols)["ridge"]["margin"](tr),
        target="home_margin", windows=WINDOWS,
    )
    return win, margin


def _pooled_summary(win, margin):
    y = np.asarray(win["pooled"]["y"], dtype=float)
    p = np.asarray(win["pooled"]["preds"], dtype=float)
    my = np.asarray(margin["pooled"]["y"], dtype=float)
    mp = np.asarray(margin["pooled"]["preds"], dtype=float)
    pw = _per_game_win(y, p)
    pm = _per_game_margin(my, mp)
    return {
        "n_win": int(len(y)),
        "n_margin": int(len(my)),
        "win_log_loss": float(pw["log_loss"].mean()),
        "win_brier": float(pw["brier"].mean()),
        "win_auc": float(roc_auc_score(y, np.clip(p, CLIP, 1 - CLIP))),
        "margin_mae": float(pm["abs_error"].mean()),
        "margin_rmse": float(np.sqrt(np.mean((mp - my) ** 2))),
    }, pw, pm


def main() -> int:
    if not CACHE.exists():
        print(f"no training cache at {CACHE}; this is a local measurement, not a test")
        return 1

    raw = pd.DataFrame(json.loads(CACHE.read_text()))
    raw = raw.sort_values("game_date").reset_index(drop=True)

    arms, per_game, ids_ref = {}, {}, None
    for name, kwargs in ARMS.items():
        frame, cols = build_training_frame(raw, **kwargs)
        frame = frame.assign(home_margin=frame.home_pts - frame.away_pts)
        win, margin = _run_arm(frame, cols)
        summary, pw, pm = _pooled_summary(win, margin)
        ids = list(win["pooled"]["game_ids"])
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise SystemExit(f"arm {name!r} scored different games or a different order; refusing to pair by position")
        arms[name] = {
            "columns": cols,
            "pooled": summary,
            "window_sizes": [int(w["n"]) for w in win["windows"] if "n" in w],
        }
        per_game[name] = {**pw, **pm}
        print(f"{name:<15} n={summary['n_win']} logloss={summary['win_log_loss']:.5f} "
              f"brier={summary['win_brier']:.5f} auc={summary['win_auc']:.4f} "
              f"margin_mae={summary['margin_mae']:.4f} margin_rmse={summary['margin_rmse']:.4f}")

    base = per_game["baseline"]
    base_summary = arms["baseline"]["pooled"]
    verdicts = {}
    for name in ARMS:
        if name == "baseline":
            continue
        cand = per_game[name]
        primary = {}
        for metric in ("log_loss", "abs_error"):
            diff = cand[metric] - base[metric]
            lo, hi = _paired_mean_ci(diff)
            improving = hi < 0
            primary[metric] = {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                               "excludes_zero_improving": bool(improving)}
        cs = arms[name]["pooled"]
        point = {
            "win_log_loss": cs["win_log_loss"] - base_summary["win_log_loss"],
            "win_brier": cs["win_brier"] - base_summary["win_brier"],
            "win_auc": cs["win_auc"] - base_summary["win_auc"],
            "margin_mae": cs["margin_mae"] - base_summary["margin_mae"],
            "margin_rmse": cs["margin_rmse"] - base_summary["margin_rmse"],
        }
        worse = [k for k, v in point.items() if (v > 0 if k != "win_auc" else v < 0)]
        primary_ok = primary["log_loss"]["excludes_zero_improving"] and primary["abs_error"]["excludes_zero_improving"]
        ships = bool(primary_ok and not worse)
        verdicts[name] = {"primary_bootstrap": primary, "point_delta": point, "metrics_worse_at_point": worse,
                          "ships": ships}
        print(f"\n{name} vs baseline")
        for k, v in primary.items():
            print(f"  {k:<10} mean diff {v['mean_diff']:+.5f}  95% CI [{v['ci_low']:+.5f}, {v['ci_high']:+.5f}]  "
                  f"{'excludes zero, improving' if v['excludes_zero_improving'] else 'not significant in the improving direction'}")
        print(f"  point deltas: " + ", ".join(f"{k} {v:+.5f}" for k, v in point.items()))
        print(f"  worse at point: {worse or 'none'}")
        print(f"  SHIP RULE: {'SHIPS' if ships else 'does NOT ship (flag stays default-off)'}")

    result = {
        "task": "NBA-4 rest / streak per team, 7-day rest cap, Utah altitude, visitor b2b interactions",
        "cache": CACHE.name,
        "n_games_total": int(len(raw)),
        "walk_forward": {"windows": WINDOWS, "min_train_fraction": 0.4, "expanding": True},
        "bootstrap": {"resamples": N_RESAMPLES, "seed": SEED, "statistic": "mean per-game difference"},
        "baseline": arms["baseline"],
        "arms": {k: v for k, v in arms.items() if k != "baseline"},
        "verdicts": verdicts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
