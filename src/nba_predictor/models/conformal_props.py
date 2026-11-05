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


# ---------------------------------------------------------------------------
# Trial 2 (docs/conformal-scaled-preregistration-2026-11.md): the shape race.
#
# Trial 1's champion -- rolling symmetric conformal -- covers on average but
# not by projection size: low lines over-cover (~92-95% at "80%"), star lines
# under-cover (~61-67%). Two projection-aware challengers race it on the same
# track, under the same chronology discipline:
#
# * **Scaled (normalized) rolling conformal** -- the half-width scales with a
#   volatility curve sigma(projection) fitted on the calibration segment
#   only (decile bins of |residual|, isotonic-smoothed, floored); the rolling
#   quantile is taken over normalized scores |residual| / sigma(projection).
# * **Mondrian rolling conformal** -- a separate rolling quantile per
#   projection tercile, tercile edges frozen from the calibration segment,
#   with the pooled rolling half-width as the frozen fallback for a bin whose
#   window runs thin.
#
# Chronology is enforced the same way as trial 1: every calibration input is
# selected by `game_date < day` filters (the sigma fit and the tercile edges
# read the calibration segment only, by construction of the arguments they
# are handed), and a calibration date on or after an evaluated date raises.
# ---------------------------------------------------------------------------

#: Scaled method: number of equal-frequency projection bins for sigma.
SIGMA_BINS = 10

#: Scaled method: sigma is floored at this share of the calibration
#: segment's mean |residual|, so a degenerate low bin cannot produce a
#: zero-width interval.
SIGMA_FLOOR_SHARE = 0.10

#: Mondrian method: a bin whose trailing window holds fewer residuals than
#: this falls back to the day's pooled rolling half-width (frozen rule).
MONDRIAN_MIN_BIN_RESIDUALS = 100


def _pava_non_decreasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted isotonic regression under a non-decreasing constraint
    (pool-adjacent-violators). Returns one fitted value per input."""
    blocks: list[list[float]] = []  # [weighted sum, weight sum, count]
    for value, weight in zip(values, weights):
        blocks.append([value * weight, weight, 1.0])
        while (
            len(blocks) >= 2
            and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
        ):
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] += top[2]
    fitted: list[float] = []
    for weighted_sum, weight_sum, count in blocks:
        fitted.extend([weighted_sum / weight_sum] * int(count))
    return fitted


def fit_sigma_model(
    calibration: pd.DataFrame,
    *,
    n_bins: int = SIGMA_BINS,
    floor_share: float = SIGMA_FLOOR_SHARE,
) -> dict:
    """The scaled method's sigma(projection), fitted on calibration rows ONLY.

    Equal-frequency bins by projected value (edges = the calibration
    segment's quantile cut points, duplicates collapsed); each bin's raw
    scale is its mean |residual|; the scales are isotonic-smoothed so the
    curve cannot decrease as the projection grows; a floor of `floor_share`
    x the segment's mean |residual| applies at read time. The returned model
    is plain data (edges, scales, floor) so a test -- or a reader -- can see
    exactly what was fitted.
    """
    projected = calibration["projected"].to_numpy(float)
    abs_resid = calibration["abs_resid"].to_numpy(float)
    if len(calibration) == 0:
        raise ValueError("fit_sigma_model needs at least one calibration row")
    quantiles = [i / n_bins for i in range(1, n_bins)]
    edges = sorted({float(q) for q in np.quantile(projected, quantiles)})
    assignment = np.searchsorted(edges, projected, side="right")
    n_effective = len(edges) + 1
    raw: list[float] = []
    counts: list[float] = []
    for b in range(n_effective):
        part = abs_resid[assignment == b]
        if len(part) == 0:
            raise AssertionError(
                f"sigma bin {b} is empty on {len(calibration)} calibration rows; "
                "equal-frequency bins cannot be empty -- the fit is not trusted"
            )
        raw.append(float(part.mean()))
        counts.append(float(len(part)))
    scales = _pava_non_decreasing(raw, counts)
    mean_abs = float(abs_resid.mean())
    return {
        "edges": edges,
        "scales": scales,
        "raw_scales": raw,
        "bin_counts": [int(c) for c in counts],
        "floor": floor_share * mean_abs,
        "mean_abs_resid": mean_abs,
        "n": int(len(calibration)),
    }


def sigma_values(sigma_model: dict, projected: np.ndarray | pd.Series) -> np.ndarray:
    """sigma(projection) for arbitrary rows under a fitted sigma model:
    the isotonic scale of the row's bin (nearest edge bin outside the
    fitted range), floored."""
    idx = np.searchsorted(
        sigma_model["edges"], np.asarray(projected, dtype=float), side="right"
    )
    scales = np.asarray(sigma_model["scales"], dtype=float)[idx]
    return np.maximum(scales, sigma_model["floor"])


def tercile_edges(calibration: pd.DataFrame) -> tuple[float, float]:
    """The Mondrian bin edges: the calibration segment's projected-value
    terciles (frozen at fit time, like the sigma model)."""
    cuts = calibration["projected"].quantile([1 / 3, 2 / 3]).to_numpy(float)
    return float(cuts[0]), float(cuts[1])


def tercile_assignment(
    projected: np.ndarray | pd.Series, edges: tuple[float, float]
) -> np.ndarray:
    """Bin index 0/1/2 per row under frozen edges: <= edge 1 is low,
    <= edge 2 is mid, above is high (trial 1's boundary convention)."""
    values = np.asarray(projected, dtype=float)
    return np.where(values <= edges[0], 0, np.where(values <= edges[1], 1, 2))


def evaluate_stat_trial2(
    frame: pd.DataFrame,
    calibration_days: list[str],
    *,
    levels: tuple[float, ...] = LEVELS,
    rolling_window_days: int = 14,
    min_rolling_residuals: int = 500,
    mondrian_min_bin: int = MONDRIAN_MIN_BIN_RESIDUALS,
) -> dict:
    """Walk-forward evaluation of the trial-2 methods for one stat.

    Same frame contract as `evaluate_stat` (`game_date`, `projected`,
    `actual`, `abs_resid`; `calibration_days` a prefix of the frame's game
    days). The champion (rolling symmetric) is recomputed here so all three
    methods read the identical windows; its numbers are asserted against
    trial 1's by the race script's anchor check, not assumed.

    Returns the per-method scorecards at every level, the per-day records,
    the evaluation rows carrying each method's verdict-level half-width
    (`hw_champion`, `hw_scaled`, `hw_mondrian`), the fitted sigma model,
    the Mondrian edges, and the Mondrian fallback accounting.
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
    sigma_model = fit_sigma_model(calib)
    edges = tercile_edges(calib)

    eval_rows = frame[frame["game_date"].isin(eval_days)].copy()
    eval_rows["sigma"] = sigma_values(sigma_model, eval_rows["projected"])
    eval_rows["mondrian_bin"] = tercile_assignment(eval_rows["projected"], edges)

    # Per-day half-widths per method per level; rows then carry their day's.
    hw_by_day: dict[str, dict[str, dict[float, float]]] = {
        method: {} for method in ("champion", "scaled", "mondrian")
    }
    mondrian_fallbacks = 0
    mondrian_bin_days = 0
    mondrian_min_window_bin = math.inf
    for day in eval_days:
        window = _trailing_window(frame, day, rolling_window_days)
        if len(window) < min_rolling_residuals:
            raise AssertionError(
                f"rolling window before {day} holds {len(window)} residuals "
                f"(< {min_rolling_residuals} preregistered minimum)"
            )
        pooled_q = {
            level: conformal_half_width(window["abs_resid"], level) for level in levels
        }
        hw_by_day["champion"][day] = pooled_q

        window_scores = window["abs_resid"].to_numpy(float) / sigma_values(
            sigma_model, window["projected"]
        )
        q_scaled = {
            level: conformal_half_width(window_scores, level) for level in levels
        }
        # Scaled half-widths are per-row (q x the row's own sigma); the day
        # record stores q so rows can be priced from it below.
        hw_by_day["scaled"][day] = q_scaled

        window_bins = tercile_assignment(window["projected"], edges)
        mondrian_q: dict[float, dict[int, float]] = {}
        for level in levels:
            mondrian_q[level] = {}
            for b in range(3):
                part = window["abs_resid"].to_numpy(float)[window_bins == b]
                mondrian_min_window_bin = min(mondrian_min_window_bin, len(part))
                if len(part) < mondrian_min_bin:
                    mondrian_q[level][b] = pooled_q[level]
                    if level == VERDICT_LEVEL:
                        mondrian_fallbacks += 1
                else:
                    mondrian_q[level][b] = conformal_half_width(part, level)
                if level == VERDICT_LEVEL:
                    mondrian_bin_days += 1
        hw_by_day["mondrian"][day] = {
            level: mondrian_q[level] for level in levels
        }  # type: ignore[assignment]

    def row_half_widths(method: str, level: float) -> np.ndarray:
        if method == "champion":
            return np.array(
                [hw_by_day["champion"][d][level] for d in eval_rows["game_date"]]
            )
        if method == "scaled":
            q = np.array([hw_by_day["scaled"][d][level] for d in eval_rows["game_date"]])
            return q * eval_rows["sigma"].to_numpy(float)
        bins = eval_rows["mondrian_bin"].to_numpy(int)
        return np.array(
            [
                hw_by_day["mondrian"][d][level][b]  # type: ignore[index]
                for d, b in zip(eval_rows["game_date"], bins)
            ]
        )

    miss = eval_rows["abs_resid"].to_numpy(float)
    methods: dict[str, dict] = {}
    for method in ("champion", "scaled", "mondrian"):
        for level in levels:
            hw = row_half_widths(method, level)
            methods[f"{method}_{level:g}"] = {
                "coverage": float((miss <= hw + _EPSILON).mean()),
                "mean_width": float((2.0 * hw).mean()),
                "n": int(len(miss)),
            }
        eval_rows[f"hw_{method}"] = row_half_widths(method, VERDICT_LEVEL)

    daily = []
    for day in eval_days:
        rows = eval_rows[eval_rows["game_date"] == day]
        day_miss = rows["abs_resid"].to_numpy(float)
        entry = {"date": day, "n": int(len(rows))}
        for method in ("champion", "scaled", "mondrian"):
            hw = rows[f"hw_{method}"].to_numpy(float)
            entry[f"{method}_coverage_80"] = float((day_miss <= hw + _EPSILON).mean())
        daily.append(entry)

    return {
        "n_calibration": int(len(calib)),
        "n_eval": int(len(eval_rows)),
        "calibration_days": [calib_days[0], calib_days[-1]],
        "evaluation_days": [eval_days[0], eval_days[-1]],
        "sigma_model": sigma_model,
        "mondrian_edges": [edges[0], edges[1]],
        "mondrian_fallback_bin_days": int(mondrian_fallbacks),
        "mondrian_bin_days": int(mondrian_bin_days),
        "mondrian_min_window_bin": (
            None if mondrian_min_window_bin == math.inf else int(mondrian_min_window_bin)
        ),
        "methods": methods,
        "daily": daily,
        "eval_rows": eval_rows,
    }


def tercile_coverage_methods(
    eval_rows: pd.DataFrame, hw_columns: dict[str, str]
) -> list[dict]:
    """Coverage at the verdict level by projected-value tercile, for any
    set of methods whose per-row half-widths ride on `eval_rows`.

    Tercile cut points come from the evaluation segment's own projections
    -- trial 1's diagnostic construction, kept identical so the shape table
    reads against trial 1's cell for cell. `hw_columns` maps the reported
    method name to the half-width column.
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
        entry = {
            "tercile": tercile,
            "n": int(len(part)),
            "projection_range": [
                float(part["projected"].min()),
                float(part["projected"].max()),
            ],
            "coverage_80": {},
        }
        for method, column in hw_columns.items():
            hw = part[column].to_numpy(float)
            entry["coverage_80"][method] = float((miss <= hw + _EPSILON).mean())
        out.append(entry)
    return out


def time_third_coverage_methods(
    daily: list[dict], methods: tuple[str, ...] = ("champion", "scaled", "mondrian")
) -> list[dict]:
    """Per-method coverage over early/middle/late thirds of the evaluation
    segment (row-weighted), from `evaluate_stat_trial2`'s daily records."""
    n = len(daily)
    thirds = []
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    for label, (lo, hi) in zip(("early", "middle", "late"), bounds):
        part = daily[lo:hi]
        total = sum(d["n"] for d in part)
        entry = {
            "third": label,
            "days": [part[0]["date"], part[-1]["date"]],
            "n": int(total),
            "coverage_80": {},
        }
        for method in methods:
            entry["coverage_80"][method] = float(
                sum(d[f"{method}_coverage_80"] * d["n"] for d in part) / total
            )
        thirds.append(entry)
    return thirds


#: The frozen shape band at the verdict level: every reporting tercile of a
#: shape-fixing method lands inside it (preregistration clause 1).
SHAPE_BAND = (0.74, 0.86)

#: The frozen floor under any stat's worst tercile (clause 2).
SHAPE_WORST_TERCILE_FLOOR = 0.70

#: The frozen width cap against the champion, per stat (clause 4).
SHAPE_WIDTH_CAP = 1.10

#: How many stats must satisfy clause 1 (of the four).
SHAPE_MIN_STATS = 3


def shape_fix_verdict(per_stat: dict[str, dict]) -> dict:
    """The preregistered trial-2 verdict for ONE challenger.

    `per_stat` maps stat -> {"champion": {"coverage", "mean_width"},
    "challenger": {"coverage", "mean_width"}, "terciles": [low, mid, high]}
    (the challenger's coverage at 80% on each reporting tercile). All four
    frozen clauses are judged per stat and reported; `fixes_shape` is the
    conjunction the preregistration froze -- nothing here can be tuned
    after the numbers exist.
    """
    stats_out: dict[str, dict] = {}
    clause1_stats = 0
    all_marginal = True
    all_width = True
    worst_tercile_overall = 1.0
    for stat, entry in per_stat.items():
        terciles = entry["terciles"]
        worst = min(terciles)
        worst_tercile_overall = min(worst_tercile_overall, worst)
        shape_ok = all(SHAPE_BAND[0] <= c <= SHAPE_BAND[1] for c in terciles)
        clause1_stats += int(shape_ok)
        marginal_ok = abs(entry["challenger"]["coverage"] - VERDICT_LEVEL) <= (
            VERDICT_TOLERANCE + _EPSILON
        )
        width_ok = entry["challenger"]["mean_width"] <= (
            SHAPE_WIDTH_CAP * entry["champion"]["mean_width"] + _EPSILON
        )
        all_marginal = all_marginal and marginal_ok
        all_width = all_width and width_ok
        stats_out[stat] = {
            "tercile_coverage": terciles,
            "worst_tercile": worst,
            "clause1_shape_this_stat": bool(shape_ok),
            "clause3_marginal_ok": bool(marginal_ok),
            "clause4_width_ok": bool(width_ok),
            "champion_coverage": entry["champion"]["coverage"],
            "champion_mean_width": entry["champion"]["mean_width"],
            "challenger_coverage": entry["challenger"]["coverage"],
            "challenger_mean_width": entry["challenger"]["mean_width"],
        }
    clauses = {
        "clause1_shape_3_of_4": clause1_stats >= SHAPE_MIN_STATS,
        "clause1_stats_passing": clause1_stats,
        "clause2_no_stat_abandoned": worst_tercile_overall
        >= SHAPE_WORST_TERCILE_FLOOR - _EPSILON,
        "worst_tercile_overall": worst_tercile_overall,
        "clause3_marginal_all_stats": bool(all_marginal),
        "clause4_width_all_stats": bool(all_width),
    }
    fixes = (
        clauses["clause1_shape_3_of_4"]
        and clauses["clause2_no_stat_abandoned"]
        and clauses["clause3_marginal_all_stats"]
        and clauses["clause4_width_all_stats"]
    )
    return {"per_stat": stats_out, "clauses": clauses, "fixes_shape": bool(fixes)}


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
