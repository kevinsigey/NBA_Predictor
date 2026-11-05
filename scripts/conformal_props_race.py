"""Run the conformal-interval race for player props.

Preregistered in docs/conformal-props-preregistration-2026-11.md (committed
before this script ran; the spec may not be edited to rescue a marginal
result). The track is the star-factor races' track: the frozen history, the
production frame builder, the production factories, the harness's
date-aligned last-20% holdout. On that holdout's residuals, three interval
constructions are compared walk-forward:

* the status-quo MAE bands (projection +/- k x the manifest holdout MAE),
* split conformal (quantile frozen over the first 56 holdout game days),
* rolling conformal (quantile over the trailing 14 game days, per day).

Usage (from the repo root):  PYTHONPATH=src python scripts/conformal_props_race.py

Writes docs/conformal-props-race-2026-11.json (raw numbers). The results
document is written from that JSON, not from this script's stdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.features.prop_model_frame import build_prop_model_frame
from nba_predictor.models.conformal_props import (
    LEVELS,
    conformal_half_width,
    evaluate_stat,
    interval_stats,
    method_verdict,
    tercile_coverage,
    time_third_coverage,
    _trailing_window,
)
from nba_predictor.models.player_props import train_player_stat_model
from nba_predictor.models.prop_holdout import (
    MARKETS,
    MARKET_TARGETS,
    _date_aligned_split,
)
from nba_predictor.pipeline.history import load_history
from nba_predictor.tracking.prop_grading import prop_bands

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history" / "nba_history.json.gz"
SNAPSHOT = ROOT / "data" / "public_snapshot.json"
OUT_JSON = ROOT / "docs" / "conformal-props-race-2026-11.json"

CALIBRATION_DAY_FRACTION = 0.25  # first quarter of holdout game days, frozen
ROLLING_WINDOW_DAYS = 14         # frozen in the preregistration

#: The star-factor races' published base arm (PRs #164/#165). This race's fit
#: must reproduce it to rounding or the track is not the same track and the
#: race is a setup failure, not a result.
ANCHOR = {
    "points": (4.696, 4.606),
    "rebounds": (1.956, 1.926),
    "assists": (1.374, 1.355),
    "threes": (0.896, 0.893),
}
ANCHOR_TOLERANCE = 0.0015


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _residual_frames(players: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fit the production models on the train segment; per-stat holdout
    residual frames (game_date, projected, actual, abs_resid)."""
    train, holdout = _date_aligned_split(players, 0.2)
    combined = pd.concat([train, holdout], ignore_index=True)
    frame, feature_cols = build_prop_model_frame(combined)
    holdout_ids = set(holdout["game_id"])
    train_frame = frame[~frame["game_id"].isin(holdout_ids)]
    holdout_frame = frame[frame["game_id"].isin(holdout_ids)].reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {}
    anchors: dict[str, dict] = {}
    for market in MARKETS:
        target = MARKET_TARGETS[market]
        model = train_player_stat_model(train_frame[feature_cols], train_frame[target])
        predicted = np.asarray(model.predict(holdout_frame[feature_cols]), dtype=float)
        actual = holdout_frame[target].to_numpy(float)
        resid = actual - predicted
        frames[market] = pd.DataFrame(
            {
                "game_date": holdout_frame["game_date"].to_numpy(),
                "projected": predicted,
                "actual": actual,
                "abs_resid": np.abs(resid),
            }
        )
        mae = float(np.abs(resid).mean())
        naive = float(np.abs(holdout_frame[f"{target}_roll"].to_numpy(float) - actual).mean())
        naive_anchor, mae_anchor = ANCHOR[market]
        ok = abs(mae - mae_anchor) <= ANCHOR_TOLERANCE and abs(naive - naive_anchor) <= ANCHOR_TOLERANCE
        anchors[market] = {
            "holdout_mae": mae,
            "naive_mae": naive,
            "anchor_holdout_mae": mae_anchor,
            "anchor_naive_mae": naive_anchor,
            "reproduced": bool(ok),
        }
        _log(f"{market}: holdout MAE {mae:.4f} (anchor {mae_anchor}), naive {naive:.4f} (anchor {naive_anchor})")
        if not ok:
            raise SystemExit(
                f"SETUP FAILURE: {market} does not reproduce the star-factor anchor "
                f"(MAE {mae:.4f} vs {mae_anchor}, naive {naive:.4f} vs {naive_anchor}); "
                "the race is stopped, per the preregistration."
            )
    return frames, anchors


def _live_cross_checks(
    frames: dict[str, pd.DataFrame],
    calibration_days: list[str],
    bands: dict[str, float],
) -> dict:
    """The two cross-checks the committed snapshot supports -- and only those.

    1. Status-quo coverage over the whole live counted record, from the
       snapshot aggregates' band shares.
    2. This race's conformal quantiles transported onto the snapshot's latest
       live day (per-row projections/actuals for one day, a different model
       vintage -- directional only).
    The full live per-row record lives in the VPS tracking store, which is
    not in this repo; nothing here reaches for it.
    """
    snapshot = json.loads(SNAPSHOT.read_text())
    grading = snapshot["prop_grading"]
    aggregates = grading["aggregates"]

    whole_record = {}
    for per_stat in aggregates["per_stat"]:
        shares = {b["grade"]: b["pct"] for b in per_stat["bands"]}
        whole_record[per_stat["stat"]] = {
            "n": per_stat["n"],
            "mae": per_stat["mae"],
            "band_1x_coverage": (shares.get("On the number", 0.0) + shares.get("Close", 0.0)) / 100.0,
            "band_2x_coverage": (
                shares.get("On the number", 0.0) + shares.get("Close", 0.0) + shares.get("Off", 0.0)
            ) / 100.0,
        }

    day_rows = [
        row
        for game in (grading["latest_day"] or {}).get("games", [])
        for row in game["rows"]
    ]
    transported: dict[str, dict] = {}
    for market in MARKETS:
        rows = [r for r in day_rows if r["stat"] == market]
        if not rows:
            continue
        projected = np.array([r["predicted"] for r in rows], dtype=float)
        actual = np.array([r["actual"] for r in rows], dtype=float)
        frame = frames[market]
        calib = frame[frame["game_date"].isin(calibration_days)]
        last_day = sorted(frame["game_date"].unique())[-1]
        final_window = _trailing_window(frame, "9999-12-31", ROLLING_WINDOW_DAYS)
        assert final_window["game_date"].max() == last_day
        entry: dict[str, dict] = {}
        for multiple in (1.0, 2.0):
            entry[f"mae_band_{multiple:g}x"] = interval_stats(projected, actual, bands[market] * multiple)
        for level in LEVELS:
            entry[f"split_{level:g}"] = interval_stats(
                projected, actual, conformal_half_width(calib["abs_resid"], level)
            )
            entry[f"rolling_{level:g}"] = interval_stats(
                projected, actual, conformal_half_width(final_window["abs_resid"], level)
            )
        transported[market] = entry

    return {
        "snapshot_generated_at": snapshot.get("generated_at"),
        "aggregates_window": {
            "from_date": aggregates["from_date"],
            "to_date": aggregates["to_date"],
            "n_days": aggregates["n_days"],
            "total_pre_tip": aggregates["total"],
        },
        "whole_record_status_quo": whole_record,
        "latest_day": {
            "date": (grading["latest_day"] or {}).get("date"),
            "n_rows": len(day_rows),
            "transported": transported,
        },
    }


def main() -> int:
    bands = prop_bands(ROOT / "models")
    if bands is None or any(m not in bands for m in MARKETS):
        raise SystemExit("SETUP FAILURE: manifest MAE bands unavailable")
    _log(f"manifest MAE bands: { {m: round(bands[m], 4) for m in MARKETS} }")

    _, players = load_history("2100-01-01", HISTORY)
    players = players[players["minutes"].fillna(0) > 0].reset_index(drop=True)
    _log(f"history: {len(players)} player rows with minutes > 0")

    frames, anchors = _residual_frames(players)

    days = sorted(frames[MARKETS[0]]["game_date"].unique())
    n_calib_days = int(len(days) * CALIBRATION_DAY_FRACTION)
    calibration_days = days[:n_calib_days]
    _log(
        f"holdout days: {len(days)} ({days[0]} -> {days[-1]}); "
        f"calibration: first {n_calib_days} days (->{calibration_days[-1]}); "
        f"evaluation: {len(days) - n_calib_days} days"
    )

    results: dict = {
        "population": {
            "history_rows_minutes_gt_0": int(len(players)),
            "holdout_rows": int(len(frames[MARKETS[0]])),
            "holdout_days": len(days),
            "holdout_span": [days[0], days[-1]],
            "calibration_days": [calibration_days[0], calibration_days[-1]],
            "evaluation_days": [days[n_calib_days], days[-1]],
            "manifest_mae_bands": {m: bands[m] for m in MARKETS},
        },
        "anchors": anchors,
        "per_stat": {},
    }

    pooled_rows: list[pd.DataFrame] = []
    for market in MARKETS:
        _log(f"evaluating {market}")
        res = evaluate_stat(
            frames[market],
            calibration_days,
            band_mae=bands[market],
            rolling_window_days=ROLLING_WINDOW_DAYS,
        )
        verdicts = {
            method: method_verdict(res["methods"], method, band_mae=bands[market])
            for method in ("split_0.8", "rolling_0.8")
        }
        eval_rows = res.pop("eval_rows")
        eval_rows = eval_rows.assign(stat=market)
        pooled_rows.append(eval_rows)
        results["per_stat"][market] = {
            **res,
            "verdicts_80": verdicts,
            "terciles": tercile_coverage(eval_rows),
            "time_thirds": time_third_coverage(res["daily"]),
        }
        for method, v in verdicts.items():
            _log(
                f"  {market} {method}: coverage {v['coverage']:.3f}, width {v['mean_width']:.2f} "
                f"vs {v['comparator_band']} band (cov {v['comparator_coverage']:.3f}, width "
                f"{v['comparator_mean_width']:.2f}) -> {'WIN' if v['win'] else 'no win'}"
            )

    pooled = pd.concat(pooled_rows, ignore_index=True)
    miss = pooled["abs_resid"].to_numpy(float)
    pooled_methods: dict[str, dict] = {}
    for key, hw_col in (("split", "hw_split"), ("rolling", "hw_rolling")):
        hw = pooled[hw_col].to_numpy(float)
        pooled_methods[f"{key}_0.8"] = {
            "coverage": float((miss <= hw + 1e-9).mean()),
            "mean_width": float((2.0 * hw).mean()),
            "n": int(len(pooled)),
        }
    band_hw = pooled["stat"].map(bands).to_numpy(float)
    for multiple in (1.0, 2.0):
        pooled_methods[f"mae_band_{multiple:g}x"] = {
            "coverage": float((miss <= band_hw * multiple + 1e-9).mean()),
            "mean_width": float((2.0 * band_hw * multiple).mean()),
            "n": int(len(pooled)),
        }
    monthly = []
    pooled = pooled.assign(month=pooled["game_date"].str[:7])
    for month, part in pooled.groupby("month"):
        m = part["abs_resid"].to_numpy(float)
        monthly.append(
            {
                "month": month,
                "n": int(len(part)),
                "split_coverage_80": float((m <= part["hw_split"].to_numpy(float) + 1e-9).mean()),
                "rolling_coverage_80": float((m <= part["hw_rolling"].to_numpy(float) + 1e-9).mean()),
                "band_1x_coverage": float((m <= part["hw_band_1x"].to_numpy(float) + 1e-9).mean()),
            }
        )
    results["pooled"] = {"methods_80": pooled_methods, "monthly": monthly}

    _log("live cross-checks from the committed snapshot")
    results["live_cross_check"] = _live_cross_checks(frames, calibration_days, bands)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _log(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
