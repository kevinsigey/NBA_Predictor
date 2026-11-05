"""Conformal intervals for the player-prop projections, raced against the
MAE-multiple bands a reader can currently derive.

The props models publish point projections. The only uncertainty statement in
the product is the grade band (`tracking/prop_grading.py`): a miss within
1x the stat's holdout MAE is "Close", within 2x is "Off". Those multiples
were never chosen to contain any particular share of outcomes -- they are a
grading ruler, not an interval. Conformal prediction is the standard way to
build an interval from a forecaster's own residuals with a coverage target:
projection +/- the (1-alpha) quantile of past |residuals|, with the
finite-sample correction that makes the target honest at finite n.

This module is the measurement, built for the preregistered race in
`docs/conformal-props-preregistration-2026-11.md`. It changes no production
behaviour: nothing served imports it. Three constructions are compared on
one fixed model's holdout residuals, strictly chronologically:

* **MAE bands** -- projection +/- k x the manifest's published holdout MAE.
  No nominal level; their empirical coverage is the finding.
* **Split conformal** -- the quantile is computed once over a fixed
  calibration segment and frozen, the way a pre-season fit is lived with.
* **Rolling conformal** -- the quantile is recomputed for each game day over
  the trailing window of game days strictly before it, so the interval can
  follow form drift.

Chronology is enforced structurally, not by caller discipline: calibration
rows are selected by `game_date < day` filters, and a calibration date on or
after an evaluated date raises rather than silently leaking.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Nominal levels the race reports, as coverage targets (1 - alpha).
LEVELS = (0.5, 0.8, 0.9)

#: The verdict level, frozen in the preregistration.
VERDICT_LEVEL = 0.8

#: Coverage tolerance at the verdict level: empirical coverage within this
#: many points of nominal satisfies clause (a).
VERDICT_TOLERANCE = 0.03

#: MAE band multiples. The verdict comparison uses {1x, 2x}; 0.5x is the
#: "On the number" boundary and is recorded for completeness.
BAND_MULTIPLES = (0.5, 1.0, 2.0)
VERDICT_BAND_MULTIPLES = (1.0, 2.0)

#: Same epsilon discipline as the grading bands: a miss of exactly the
#: half-width is inside the interval.
_EPSILON = 1e-9


def conformal_half_width(abs_residuals: np.ndarray | pd.Series, level: float) -> float:
    """The split-conformal half-width at `level` from calibration |residuals|.

    The k-th smallest |residual| with k = ceil((n + 1) * level), capped at n:
    the finite-sample correction that makes the coverage statement hold at
    finite n rather than asymptotically. No interpolation -- the interval is
    an order statistic of errors the model actually made.
    """
    values = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(values)
    if n == 0:
        raise ValueError("conformal_half_width needs at least one calibration residual")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    k = min(max(math.ceil((n + 1) * level), 1), n)
    return float(values[k - 1])


def interval_stats(projected: np.ndarray, actual: np.ndarray, half_width: float) -> dict:
    """Empirical coverage and mean width of projection +/- half_width."""
    miss = np.abs(np.asarray(actual, dtype=float) - np.asarray(projected, dtype=float))
    return {
        "coverage": float((miss <= half_width + _EPSILON).mean()),
        "mean_width": 2.0 * float(half_width),
        "n": int(len(miss)),
    }


def _trailing_window(
    frame: pd.DataFrame, day: str, window_days: int
) -> pd.DataFrame:
    """Rows from the `window_days` game days strictly before `day`.

    The `game_date < day` filter is the leakage proof: a row dated on or
    after the day being scored cannot enter its own calibration, whatever
    the caller passes in.
    """
    prior = frame[frame["game_date"] < day]
    days = sorted(prior["game_date"].unique())[-window_days:]
    return prior[prior["game_date"].isin(days)]


def evaluate_stat(
    frame: pd.DataFrame,
    calibration_days: list[str],
    *,
    band_mae: float,
    levels: tuple[float, ...] = LEVELS,
    band_multiples: tuple[float, ...] = BAND_MULTIPLES,
    rolling_window_days: int = 14,
    min_rolling_residuals: int = 500,
) -> dict:
    """Walk-forward interval evaluation for one stat's holdout residuals.

    `frame` carries one row per holdout player-game with `game_date`,
    `projected` and `actual`. The calibration segment is `calibration_days`
    (a prefix of the frame's game days); every later game day is an
    evaluation block, scored with calibrations drawn only from strictly
    earlier rows. Returns the aggregate scorecard, the per-day coverage
    records, and the per-row evaluation frame (each row carrying every
    method's 80% half-width) that the diagnostics read.
    """
    days = sorted(frame["game_date"].unique())
    calib_days = sorted(calibration_days)
    if not calib_days or calib_days != days[: len(calib_days)]:
        raise ValueError("calibration_days must be a prefix of the frame's game days")
    eval_days = days[len(calib_days) :]
    if not eval_days:
        raise ValueError("no evaluation days after the calibration segment")
    if calib_days[-1] >= eval_days[0]:
        raise AssertionError(
            f"calibration reaches the evaluation segment: {calib_days[-1]} >= {eval_days[0]}"
        )

    calib = frame[frame["game_date"].isin(calib_days)]
    split_hw = {
        level: conformal_half_width(calib["abs_resid"], level) for level in levels
    }

    eval_rows = frame[frame["game_date"].isin(eval_days)].copy()
    rolling_hw_by_day: dict[str, dict[float, float]] = {}
    for day in eval_days:
        window = _trailing_window(frame, day, rolling_window_days)
        if len(window) < min_rolling_residuals:
            raise AssertionError(
                f"rolling window before {day} holds {len(window)} residuals "
                f"(< {min_rolling_residuals} preregistered minimum)"
            )
        rolling_hw_by_day[day] = {
            level: conformal_half_width(window["abs_resid"], level) for level in levels
        }
    eval_rows["hw_rolling"] = [
        rolling_hw_by_day[d][VERDICT_LEVEL] for d in eval_rows["game_date"]
    ]
    eval_rows["hw_split"] = split_hw[VERDICT_LEVEL]
    eval_rows["hw_band_1x"] = band_mae * 1.0

    projected = eval_rows["projected"].to_numpy(float)
    actual = eval_rows["actual"].to_numpy(float)

    methods: dict[str, dict] = {}
    for multiple in band_multiples:
        methods[f"mae_band_{multiple:g}x"] = interval_stats(
            projected, actual, band_mae * multiple
        )
    for level in levels:
        methods[f"split_{level:g}"] = interval_stats(projected, actual, split_hw[level])
    # Rolling half-widths vary by day, so coverage is counted per row rather
    # than through one shared width; mean width is the mean of the daily widths.
    miss = np.abs(actual - projected)
    for level in levels:
        hw = np.array([rolling_hw_by_day[d][level] for d in eval_rows["game_date"]])
        methods[f"rolling_{level:g}"] = {
            "coverage": float((miss <= hw + _EPSILON).mean()),
            "mean_width": float((2.0 * hw).mean()),
            "n": int(len(miss)),
        }

    daily = []
    for day in eval_days:
        rows = eval_rows[eval_rows["game_date"] == day]
        day_miss = rows["abs_resid"].to_numpy(float)
        daily.append(
            {
                "date": day,
                "n": int(len(rows)),
                "mae": float(day_miss.mean()),
                "split_coverage_80": float(
                    (day_miss <= split_hw[VERDICT_LEVEL] + _EPSILON).mean()
                ),
                "rolling_coverage_80": float(
                    (day_miss <= rolling_hw_by_day[day][VERDICT_LEVEL] + _EPSILON).mean()
                ),
                "band_1x_coverage": float((day_miss <= band_mae + _EPSILON).mean()),
            }
        )

    return {
        "n_calibration": int(len(calib)),
        "n_eval": int(len(eval_rows)),
        "calibration_days": [calib_days[0], calib_days[-1]],
        "evaluation_days": [eval_days[0], eval_days[-1]],
        "split_half_widths": {f"{level:g}": split_hw[level] for level in levels},
        "methods": methods,
        "daily": daily,
        "eval_rows": eval_rows,
    }


def method_verdict(
    methods: dict[str, dict],
    method_key: str,
    *,
    band_mae: float,
    nominal: float = VERDICT_LEVEL,
    tolerance: float = VERDICT_TOLERANCE,
) -> dict:
    """The preregistered verdict for one conformal method on one stat.

    Win = (a) empirical coverage within `tolerance` of nominal, AND
    (b) mean width no greater than the narrowest MAE band (among the verdict
    multiples) whose coverage reaches the method's -- a method may not win
    by simply being wider than everything that covers as much. If no band
    reaches the method's coverage, (b) is judged against the widest band.
    """
    got = methods[method_key]
    coverage = got["coverage"]
    clause_a = abs(coverage - nominal) <= tolerance + _EPSILON

    bands = [
        (multiple, methods[f"mae_band_{multiple:g}x"])
        for multiple in VERDICT_BAND_MULTIPLES
    ]
    qualifying = [(m, b) for m, b in bands if b["coverage"] >= coverage - _EPSILON]
    if qualifying:
        comparator_multiple, comparator = min(qualifying, key=lambda mb: mb[1]["mean_width"])
        fallback = False
    else:
        comparator_multiple, comparator = max(bands, key=lambda mb: mb[1]["mean_width"])
        fallback = True
    clause_b = got["mean_width"] <= comparator["mean_width"] + _EPSILON

    return {
        "method": method_key,
        "coverage": coverage,
        "mean_width": got["mean_width"],
        "clause_a": bool(clause_a),
        "clause_b": bool(clause_b),
        "comparator_band": f"{comparator_multiple:g}x",
        "comparator_coverage": comparator["coverage"],
        "comparator_mean_width": comparator["mean_width"],
        "comparator_was_fallback": fallback,
        "win": bool(clause_a and clause_b),
    }


def tercile_coverage(eval_rows: pd.DataFrame) -> list[dict]:
    """Coverage at the verdict level by projected-value tercile (diagnostic).

    Tercile cut points come from the evaluation segment's own projections --
    descriptive only; nothing gated reads this. A method that covers on
    average but abandons the top tercile (the stars) is visible here.
    """
    rows = eval_rows.copy()
    cuts = rows["projected"].quantile([1 / 3, 2 / 3]).to_numpy(float)
    rows["tercile"] = np.where(
        rows["projected"] <= cuts[0],
        "low",
        np.where(rows["projected"] <= cuts[1], "mid", "high"),
    )
    out = []
    for tercile in ("low", "mid", "high"):
        part = rows[rows["tercile"] == tercile]
        miss = part["abs_resid"].to_numpy(float)
        out.append(
            {
                "tercile": tercile,
                "n": int(len(part)),
                "projection_range": [float(part["projected"].min()), float(part["projected"].max())],
                "split_coverage_80": float((miss <= part["hw_split"].to_numpy(float) + _EPSILON).mean()),
                "rolling_coverage_80": float((miss <= part["hw_rolling"].to_numpy(float) + _EPSILON).mean()),
                "band_1x_coverage": float((miss <= part["hw_band_1x"].to_numpy(float) + _EPSILON).mean()),
            }
        )
    return out


def time_third_coverage(daily: list[dict]) -> list[dict]:
    """Split vs rolling coverage over early/middle/late thirds of the
    evaluation segment (diagnostic): row-weighted coverage per third."""
    n = len(daily)
    thirds = []
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    for label, (lo, hi) in zip(("early", "middle", "late"), bounds):
        part = daily[lo:hi]
        total = sum(d["n"] for d in part)
        thirds.append(
            {
                "third": label,
                "days": [part[0]["date"], part[-1]["date"]],
                "n": int(total),
                "split_coverage_80": float(
                    sum(d["split_coverage_80"] * d["n"] for d in part) / total
                ),
                "rolling_coverage_80": float(
                    sum(d["rolling_coverage_80"] * d["n"] for d in part) / total
                ),
                "band_1x_coverage": float(
                    sum(d["band_1x_coverage"] * d["n"] for d in part) / total
                ),
            }
        )
    return thirds
