"""Run star-factor trial 2: the star_out-GATED variant vs the base 12 features.

Preregistered in docs/star-factor-gated-preregistration-2026-11.md (committed before this
script ran; the spec may not be edited to rescue a marginal result). The harness, split,
factories and bootstrap are trial 1's: prop_holdout_metrics, chronological last-20%
date-aligned holdout on the frozen history, production train_player_stat_model factories,
paired bootstrap with 2,000 resamples.

Usage (from the repo root):  PYTHONPATH=src python scripts/star_factor_gated_race.py

Writes docs/star-factor-gated-race-2026-11.json (raw numbers) and .md (the tables).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.features.player_stats import PLAYER_FEATURE_COLUMNS
from nba_predictor.features.prop_model_frame import (
    EXTRA_FEATURE_COLUMNS,
    PROP_FEATURE_COLUMNS,
    build_prop_model_frame,
)
from nba_predictor.features.star_factor import STAR_FACTOR_COLUMNS
from nba_predictor.models.player_props import train_player_stat_model
from nba_predictor.models.prop_holdout import (
    MARKETS,
    MARKET_TARGETS,
    _date_aligned_split,
    prop_holdout_metrics,
)
from nba_predictor.pipeline.history import load_history

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history" / "nba_history.json.gz"
OUT_JSON = ROOT / "docs" / "star-factor-gated-race-2026-11.json"
OUT_MD = ROOT / "docs" / "star-factor-gated-race-2026-11.md"

BASE_COLUMNS = PLAYER_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS          # the 12 production features
CANDIDATE_COLUMNS = BASE_COLUMNS + STAR_FACTOR_COLUMNS                  # + the 7 gated star columns
assert CANDIDATE_COLUMNS == PROP_FEATURE_COLUMNS, "candidate arm must be exactly what would ship"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261102                                               # frozen in the preregistration

#: Trial 1's published numbers (PR #164): (naive MAE, base MAE) per market. The base arm of
#: this race must reproduce them to rounding, or the setup is not trial 1's track.
TRIAL1_ANCHOR = {
    "points": (4.696, 4.606),
    "rebounds": (1.956, 1.926),
    "assists": (1.374, 1.355),
    "threes": (0.896, 0.893),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _base_builder(df: pd.DataFrame):
    frame, _ = build_prop_model_frame(df)
    return frame, BASE_COLUMNS


def _candidate_builder(df: pd.DataFrame):
    frame, _ = build_prop_model_frame(df)
    return frame, CANDIDATE_COLUMNS


def main() -> int:
    _, players = load_history("2100-01-01", HISTORY)
    players = players[players["minutes"].fillna(0) > 0].reset_index(drop=True)
    _log(f"history: {len(players)} player rows with minutes > 0")
    factories = {m: train_player_stat_model for m in MARKETS}

    _log("harness: base arm")
    base_res = prop_holdout_metrics(players, factories, frame_builder=_base_builder)
    _log("harness: candidate arm (gated)")
    cand_res = prop_holdout_metrics(players, factories, frame_builder=_candidate_builder)

    # Per-row errors along the harness's exact path (same split function, same frame build,
    # same factories), so the paired bootstrap and the slices describe the same models the
    # harness scored. The aggregates are asserted equal below; a mismatch is a setup failure.
    train, holdout = _date_aligned_split(players, 0.2)
    combined = pd.concat([train, holdout], ignore_index=True)
    frame, _ = build_prop_model_frame(combined)
    holdout_ids = set(holdout["game_id"])
    train_frame = frame[~frame["game_id"].isin(holdout_ids)]
    holdout_frame = frame[frame["game_id"].isin(holdout_ids)].reset_index(drop=True)
    _log(f"per-row path: {len(train_frame)} train rows, {len(holdout_frame)} holdout rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    markets: dict[str, dict] = {}
    max_consistency_diff = 0.0
    for market in MARKETS:
        target = MARKET_TARGETS[market]
        y = holdout_frame[target].to_numpy(float)
        preds = {}
        for arm, cols in (("base", BASE_COLUMNS), ("candidate", CANDIDATE_COLUMNS)):
            model = train_player_stat_model(train_frame[cols], train_frame[target])
            preds[arm] = np.asarray(model.predict(holdout_frame[cols]), dtype=float)
        err_base = preds["base"] - y
        err_cand = preds["candidate"] - y
        mae_base = float(np.abs(err_base).mean())
        mae_cand = float(np.abs(err_cand).mean())
        max_consistency_diff = max(
            max_consistency_diff,
            abs(mae_base - base_res[market]["holdout_mae"]),
            abs(mae_cand - cand_res[market]["holdout_mae"]),
        )
        d = np.abs(err_cand) - np.abs(err_base)
        idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
        boot = d[idx].mean(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

        slices = {}
        for name, mask in (
            ("star_out_1", (holdout_frame["star_out"] == 1).to_numpy()),
            ("star_out_0", (holdout_frame["star_out"] == 0).to_numpy()),
        ):
            slices[name] = {
                "n": int(mask.sum()),
                "mae_base": float(np.abs(err_base[mask]).mean()) if mask.any() else None,
                "mae_candidate": float(np.abs(err_cand[mask]).mean()) if mask.any() else None,
                "bias_base": float(err_base[mask].mean()) if mask.any() else None,
                "bias_candidate": float(err_cand[mask].mean()) if mask.any() else None,
            }
        anchor_naive, anchor_base = TRIAL1_ANCHOR[market]
        markets[market] = {
            "naive_mae": base_res[market]["naive_mae"],
            "base_mae": mae_base,
            "candidate_mae": mae_cand,
            "delta": mae_cand - mae_base,
            "ci95": ci,
            "anchor": {
                "trial1_naive_mae": anchor_naive,
                "trial1_base_mae": anchor_base,
                "reproduced": abs(base_res[market]["naive_mae"] - anchor_naive) < 0.0005
                and abs(mae_base - anchor_base) < 0.0005,
            },
            "slices": slices,
        }
        _log(f"{market}: base {mae_base:.4f} cand {mae_cand:.4f} delta {mae_cand - mae_base:+.4f} "
             f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    if max_consistency_diff > 1e-6:
        raise AssertionError(
            f"per-row path disagrees with the harness by {max_consistency_diff:.2e}; setup failure, not a result"
        )

    improved = [m for m in MARKETS if markets[m]["delta"] < 0]
    confirmed = [m for m in MARKETS if markets[m]["delta"] < 0 and markets[m]["ci95"][1] < 0]
    worsened_beyond_noise = [
        m for m in MARKETS if markets[m]["delta"] > 0 and markets[m]["ci95"][0] > 0
    ]
    gate = {
        "clause_a_any_improves_and_none_worsens_beyond_noise": bool(improved) and not worsened_beyond_noise,
        "clause_b_improving_stat_ci_excludes_zero": bool(confirmed),
        "clause_c_ci_green": None,  # decided on the PR, after this run
        "improved": improved,
        "confirmed": confirmed,
        "worsened_beyond_noise": worsened_beyond_noise,
    }
    gate["passes_pre_ci"] = gate["clause_a_any_improves_and_none_worsens_beyond_noise"] and gate["clause_b_improving_stat_ci_excludes_zero"]

    result = {
        "preregistration": "docs/star-factor-gated-preregistration-2026-11.md",
        "data": {"file": "data/history/nba_history.json.gz", "player_rows_minutes_gt_0": len(players)},
        "split": {
            "holdout_fraction": 0.2, "date_aligned": True,
            "train_max_date": base_res["points"]["train_max_date"],
            "holdout_min_date": base_res["points"]["holdout_min_date"],
            "n_holdout": int(len(holdout_frame)),
            "star_out_1_rows": int((holdout_frame["star_out"] == 1).sum()),
            "star_out_0_rows": int((holdout_frame["star_out"] == 0).sum()),
        },
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "max_consistency_diff_vs_harness": max_consistency_diff,
        "anchor_reproduced_all": all(markets[m]["anchor"]["reproduced"] for m in MARKETS),
        "markets": markets,
        "gate": gate,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float) + "\n")
    OUT_MD.write_text(_md(result))
    _log(f"wrote {OUT_JSON.name} and {OUT_MD.name}; gate pre-CI: {gate['passes_pre_ci']}")
    return 0


def _md(r: dict) -> str:
    lines = [
        "# Star factor trial 2 (star_out-gated): race results",
        "",
        "Preregistered in `docs/star-factor-gated-preregistration-2026-11.md`. Raw numbers: "
        "`docs/star-factor-gated-race-2026-11.json`.",
        "",
        f"Holdout: {r['split']['n_holdout']} rows (train through {r['split']['train_max_date']}, "
        f"holdout from {r['split']['holdout_min_date']}); star_out == 1 on {r['split']['star_out_1_rows']} "
        f"rows, star_out == 0 on {r['split']['star_out_0_rows']}. Paired bootstrap "
        f"{r['bootstrap']['resamples']} resamples, seed {r['bootstrap']['seed']}.",
        "",
        f"Replication anchor (trial 1's base/naive MAEs reproduced): **{r['anchor_reproduced_all']}**. "
        f"Per-row path vs harness max MAE difference: {r['max_consistency_diff_vs_harness']:.2e}.",
        "",
        "| stat | naive MAE | base MAE | candidate MAE | delta (cand-base) | paired 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for m in MARKETS:
        x = r["markets"][m]
        lines.append(
            f"| {m} | {x['naive_mae']:.3f} | {x['base_mae']:.3f} | {x['candidate_mae']:.3f} | "
            f"{x['delta']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] |"
        )
    lines += ["", "## Slices (transparency, not the gate)", "",
              "| stat | slice | n | MAE base -> candidate | bias base -> candidate |",
              "|---|---|---|---|---|"]
    for m in MARKETS:
        for sname, label in (("star_out_1", "star_out == 1"), ("star_out_0", "star_out == 0")):
            s = r["markets"][m]["slices"][sname]
            lines.append(
                f"| {m} | {label} | {s['n']} | {s['mae_base']:.4f} -> {s['mae_candidate']:.4f} | "
                f"{s['bias_base']:+.4f} -> {s['bias_candidate']:+.4f} |"
            )
    g = r["gate"]
    lines += ["", "## Gate (frozen in the preregistration)", "",
              f"- (a) >=1 stat improves and none worsens beyond noise: **{g['clause_a_any_improves_and_none_worsens_beyond_noise']}** "
              f"(improved: {g['improved'] or 'none'}; worsened beyond noise: {g['worsened_beyond_noise'] or 'none'})",
              f"- (b) the improving stat's CI excludes zero: **{g['clause_b_improving_stat_ci_excludes_zero']}** "
              f"(confirmed: {g['confirmed'] or 'none'})",
              "- (c) branch CI fully green: decided on the PR.",
              "",
              f"**Gate before clause (c): {'PASS' if g['passes_pre_ci'] else 'FAIL'}**",
              ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
