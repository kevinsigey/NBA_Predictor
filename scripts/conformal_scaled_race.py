"""Run the trial-2 conformal shape race for player props.

Preregistered in docs/conformal-scaled-preregistration-2026-11.md
(committed before this script ran; the spec may not be edited to rescue a
marginal result). The track is trial 1's track: the frozen history, the
production frame builder, the production factories, the harness's
date-aligned last-20% holdout, calibration = the first 56 holdout game
days. On that holdout's residuals, three constructions are compared
walk-forward:

* the champion -- rolling symmetric conformal (trial 1's winner),
* scaled (normalized) rolling conformal -- sigma(projection) fitted on
  the calibration segment only, isotonic decile bins, rolling quantile
  over normalized scores,
* Mondrian rolling conformal -- a rolling quantile per projection
  tercile, edges frozen from the calibration segment, pooled fallback
  under 100 window residuals in a bin.

Usage (from the repo root):  PYTHONPATH=src python scripts/conformal_scaled_race.py

Writes docs/conformal-scaled-race-2026-11.json (raw numbers). The results
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
    evaluate_stat,
    evaluate_stat_trial2,
    shape_fix_verdict,
    tercile_coverage_methods,
    time_third_coverage_methods,
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
OUT_JSON = ROOT / "docs" / "conformal-scaled-race-2026-11.json"

CALIBRATION_DAY_FRACTION = 0.25  # first quarter of holdout game days, frozen (trial 1)
ROLLING_WINDOW_DAYS = 14         # frozen in both preregistrations

#: Trial 1's published rolling numbers (docs/conformal-props-results-2026-11.md).
#: This race's champion arm must reproduce them to the preregistered
#: tolerance -- coverage within 0.15 points, mean width within 0.05 -- or
#: the track is not the same track and the race is a setup failure.
TRIAL1_ROLLING = {
    "points": (0.798, 14.37),
    "rebounds": (0.798, 5.91),
    "assists": (0.800, 4.13),
    "threes": (0.798, 2.86),
}
ANCHOR_COVERAGE_TOLERANCE = 0.0015
ANCHOR_WIDTH_TOLERANCE = 0.05

#: The star-factor races' published base arm (PRs #164/#165), as in trial 1.
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
    residual frames (game_date, projected, actual, abs_resid). Identical
    construction to trial 1's race script."""
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


def main() -> int:
    bands = prop_bands(ROOT / "models")
    if bands is None or any(m not in bands for m in MARKETS):
        raise SystemExit("SETUP FAILURE: manifest MAE bands unavailable")

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
    verdict_inputs: dict[str, dict[str, dict]] = {"scaled": {}, "mondrian": {}}
    champion_anchor: dict[str, dict] = {}
    for market in MARKETS:
        _log(f"evaluating {market}")
        # Trial 1's harness, for the anchor: the champion recomputed by
        # trial 2 must match BOTH this and trial 1's published table.
        trial1 = evaluate_stat(
            frames[market],
            calibration_days,
            band_mae=bands[market],
            rolling_window_days=ROLLING_WINDOW_DAYS,
        )
        res = evaluate_stat_trial2(
            frames[market],
            calibration_days,
            rolling_window_days=ROLLING_WINDOW_DAYS,
        )
        champ = res["methods"]["champion_0.8"]
        pub_cov, pub_width = TRIAL1_ROLLING[market]
        anchor_ok = (
            abs(champ["coverage"] - pub_cov) <= ANCHOR_COVERAGE_TOLERANCE
            and abs(champ["mean_width"] - pub_width) <= ANCHOR_WIDTH_TOLERANCE
            and abs(champ["coverage"] - trial1["methods"]["rolling_0.8"]["coverage"]) < 1e-9
            and abs(champ["mean_width"] - trial1["methods"]["rolling_0.8"]["mean_width"]) < 1e-9
        )
        champion_anchor[market] = {
            "coverage": champ["coverage"],
            "mean_width": champ["mean_width"],
            "published_coverage": pub_cov,
            "published_mean_width": pub_width,
            "reproduced": bool(anchor_ok),
        }
        _log(
            f"  champion anchor: cov {champ['coverage']:.4f} (published {pub_cov}), "
            f"width {champ['mean_width']:.2f} (published {pub_width}) -> "
            f"{'OK' if anchor_ok else 'SETUP FAILURE'}"
        )
        if not anchor_ok:
            raise SystemExit(
                f"SETUP FAILURE: {market} champion does not reproduce trial 1's "
                "rolling numbers; the race is stopped, per the preregistration."
            )

        eval_rows = res.pop("eval_rows")
        eval_rows = eval_rows.assign(stat=market)
        pooled_rows.append(eval_rows)
        terciles = tercile_coverage_methods(
            eval_rows,
            {"champion": "hw_champion", "scaled": "hw_scaled", "mondrian": "hw_mondrian"},
        )
        results["per_stat"][market] = {
            **res,
            "terciles": terciles,
            "time_thirds": time_third_coverage_methods(res["daily"]),
        }
        for challenger in ("scaled", "mondrian"):
            verdict_inputs[challenger][market] = {
                "champion": {
                    "coverage": res["methods"]["champion_0.8"]["coverage"],
                    "mean_width": res["methods"]["champion_0.8"]["mean_width"],
                },
                "challenger": {
                    "coverage": res["methods"][f"{challenger}_0.8"]["coverage"],
                    "mean_width": res["methods"][f"{challenger}_0.8"]["mean_width"],
                },
                "terciles": [
                    t["coverage_80"][challenger] for t in terciles
                ],
            }
            v = res["methods"][f"{challenger}_0.8"]
            _log(
                f"  {market} {challenger}: cov {v['coverage']:.3f}, width {v['mean_width']:.2f}, "
                f"terciles {[round(t['coverage_80'][challenger], 3) for t in terciles]}"
            )

    results["champion_anchor"] = champion_anchor
    verdicts = {
        challenger: shape_fix_verdict(verdict_inputs[challenger])
        for challenger in ("scaled", "mondrian")
    }
    results["verdicts"] = verdicts
    for challenger, verdict in verdicts.items():
        _log(f"{challenger}: fixes_shape = {verdict['fixes_shape']} ({verdict['clauses']})")

    # Pooled numbers (all stats' evaluation rows together): the tie-break
    # reads pooled width; pooled coverage is reported second, as in trial 1.
    pooled = pd.concat(pooled_rows, ignore_index=True)
    miss = pooled["abs_resid"].to_numpy(float)
    pooled_methods: dict[str, dict] = {}
    for method, column in (
        ("champion", "hw_champion"),
        ("scaled", "hw_scaled"),
        ("mondrian", "hw_mondrian"),
    ):
        hw = pooled[column].to_numpy(float)
        pooled_methods[f"{method}_0.8"] = {
            "coverage": float((miss <= hw + 1e-9).mean()),
            "mean_width": float((2.0 * hw).mean()),
            "n": int(len(pooled)),
        }
    monthly = []
    pooled = pooled.assign(month=pooled["game_date"].str[:7])
    for month, part in pooled.groupby("month"):
        m = part["abs_resid"].to_numpy(float)
        entry = {"month": month, "n": int(len(part))}
        for method, column in (
            ("champion", "hw_champion"),
            ("scaled", "hw_scaled"),
            ("mondrian", "hw_mondrian"),
        ):
            entry[f"{method}_coverage_80"] = float(
                (m <= part[column].to_numpy(float) + 1e-9).mean()
            )
        monthly.append(entry)
    results["pooled"] = {"methods_80": pooled_methods, "monthly": monthly}

    # The frozen selection rule.
    scaled_fix = verdicts["scaled"]["fixes_shape"]
    mondrian_fix = verdicts["mondrian"]["fixes_shape"]
    if scaled_fix and mondrian_fix:
        sw = pooled_methods["scaled_0.8"]["mean_width"]
        mw = pooled_methods["mondrian_0.8"]["mean_width"]
        if abs(sw - mw) <= 0.005 * min(sw, mw):
            selected, why = "scaled", "tie on pooled width within 0.5%; frozen tie-break to scaled"
        elif sw < mw:
            selected, why = "scaled", f"both fix the shape; scaled is narrower pooled ({sw:.2f} < {mw:.2f})"
        else:
            selected, why = "mondrian", f"both fix the shape; mondrian is narrower pooled ({mw:.2f} < {sw:.2f})"
    elif scaled_fix:
        selected, why = "scaled", "only scaled fixes the shape"
    elif mondrian_fix:
        selected, why = "mondrian", "only mondrian fixes the shape"
    else:
        selected, why = "champion", "neither challenger fixes the shape; the champion ships"
    results["selection"] = {"method": selected, "reason": why}
    _log(f"SELECTION: {selected} -- {why}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _log(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
