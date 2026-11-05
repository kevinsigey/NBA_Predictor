"""Tests for the trial-2 conformal constructions (scaled + Mondrian).

The chronology tests are the point of the file, as in trial 1: a
calibration input dated on or after an evaluated row must be impossible.
The sigma fit and the Mondrian edges are handed ONLY the calibration
segment -- each test below plants exactly the leak it forbids (an enormous
residual where a leaky implementation would absorb it) and asserts the
number does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.conformal_props import (
    _pava_non_decreasing,
    evaluate_stat,
    evaluate_stat_trial2,
    fit_sigma_model,
    shape_fix_verdict,
    sigma_values,
    tercile_assignment,
    tercile_coverage_methods,
    tercile_edges,
    time_third_coverage_methods,
)


def _frame(days: int = 30, per_day: int = 40, seed: int = 7, scale_fn=None) -> pd.DataFrame:
    """A synthetic residual frame: `days` game days, `per_day` rows each.

    Projections spread 0..30; |residual| scales with `scale_fn(projected)`
    (default: flat), so a test can build a world whose volatility grows
    with the projection and watch the shape of each method's coverage.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        date = f"2025-01-{d + 1:02d}"
        projected = rng.uniform(0.0, 30.0, per_day)
        scale = scale_fn(projected) if scale_fn else np.full(per_day, 3.0)
        resid = rng.normal(0.0, 1.0, per_day) * scale
        rows.append(
            pd.DataFrame(
                {
                    "game_date": date,
                    "projected": projected,
                    "actual": projected + resid,
                    "abs_resid": np.abs(resid),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _calib_days(frame: pd.DataFrame, n: int = 8) -> list[str]:
    return sorted(frame["game_date"].unique())[:n]


# ---------------------------------------------------------------------------
# PAVA: the isotonic smoother behind sigma.
# ---------------------------------------------------------------------------


def test_pava_pools_violations_and_keeps_monotone_input():
    assert _pava_non_decreasing([3.0, 1.0, 2.0], [1.0, 1.0, 1.0]) == [2.0, 2.0, 2.0]
    assert _pava_non_decreasing([1.0, 3.0, 2.0], [1.0, 1.0, 1.0]) == [1.0, 2.5, 2.5]
    assert _pava_non_decreasing([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == [1.0, 2.0, 3.0]


def test_pava_respects_weights():
    # The heavy block drags the pooled value toward itself.
    out = _pava_non_decreasing([4.0, 0.0], [1.0, 3.0])
    assert out == [1.0, 1.0]


# ---------------------------------------------------------------------------
# The sigma fit: calibration-segment only, monotone, floored.
# ---------------------------------------------------------------------------


def test_sigma_scales_are_monotone_and_floored():
    frame = _frame(days=8, per_day=400, scale_fn=lambda p: 0.2 + 0.2 * p)
    model = fit_sigma_model(frame)
    assert model["scales"] == sorted(model["scales"])
    assert model["floor"] == pytest.approx(0.10 * model["mean_abs_resid"])
    values = sigma_values(model, np.array([0.0, 15.0, 30.0]))
    assert values[0] >= model["floor"]
    assert values[0] <= values[1] <= values[2]
    assert values[2] > 2.0 * values[0]  # the fitted curve follows the planted growth


def test_sigma_fit_reads_only_what_it_is_handed():
    frame = _frame(days=8, per_day=200, scale_fn=lambda p: 0.2 + 0.2 * p)
    model = fit_sigma_model(frame)
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    refit = fit_sigma_model(shuffled)
    assert refit["edges"] == model["edges"]
    assert refit["scales"] == pytest.approx(model["scales"])  # float sum order only


def test_evaluation_segment_cannot_reach_the_sigma_fit_or_the_edges():
    frame = _frame()
    calib = _calib_days(frame)
    base = evaluate_stat_trial2(frame, calib, min_rolling_residuals=30)
    days = sorted(frame["game_date"].unique())
    poisoned = frame.copy()
    mask = poisoned["game_date"] == days[20]
    poisoned.loc[mask, "abs_resid"] = 1e9
    poisoned.loc[mask, "projected"] = 1e6
    rerun = evaluate_stat_trial2(poisoned, calib, min_rolling_residuals=30)
    # Sigma and the Mondrian edges were fitted on the calibration segment
    # alone: an evaluation-segment plant, however large, cannot move them.
    assert rerun["sigma_model"]["edges"] == base["sigma_model"]["edges"]
    assert rerun["sigma_model"]["scales"] == base["sigma_model"]["scales"]
    assert rerun["mondrian_edges"] == base["mondrian_edges"]


def test_sigma_outside_the_fitted_range_takes_the_edge_bin():
    frame = _frame(days=8, per_day=200)
    model = fit_sigma_model(frame)
    below, above = sigma_values(model, np.array([-5.0, 1e6]))
    assert below == pytest.approx(max(model["scales"][0], model["floor"]))
    assert above == pytest.approx(max(model["scales"][-1], model["floor"]))


def test_tercile_assignment_boundaries():
    edges = (10.0, 20.0)
    got = tercile_assignment(np.array([5.0, 10.0, 15.0, 20.0, 25.0]), edges)
    assert list(got) == [0, 0, 1, 1, 2]  # <= edge is the lower bin (trial 1's convention)


def test_tercile_edges_come_from_the_frame_handed_in():
    frame = _frame(days=8, per_day=300)
    e1, e2 = tercile_edges(frame)
    assert e1 < e2
    assert e1 == pytest.approx(frame["projected"].quantile(1 / 3))


# ---------------------------------------------------------------------------
# The trial-2 evaluation: champion consistency, chronology, shape behaviour.
# ---------------------------------------------------------------------------


def test_trial2_champion_reproduces_trial1_rolling():
    frame = _frame()
    calib = _calib_days(frame)
    trial1 = evaluate_stat(frame, calib, band_mae=3.0, min_rolling_residuals=30)
    trial2 = evaluate_stat_trial2(frame, calib, min_rolling_residuals=30)
    for level in ("0.5", "0.8", "0.9"):
        a = trial1["methods"][f"rolling_{level}"]
        b = trial2["methods"][f"champion_{level}"]
        assert a["coverage"] == pytest.approx(b["coverage"])
        assert a["mean_width"] == pytest.approx(b["mean_width"])


def test_eval_day_residuals_never_enter_their_own_calibration():
    frame = _frame()
    calib = _calib_days(frame)
    base = evaluate_stat_trial2(frame, calib, min_rolling_residuals=30)
    days = sorted(frame["game_date"].unique())
    target_day = days[15]

    poisoned = frame.copy()
    mask = poisoned["game_date"] == target_day
    poisoned.loc[mask, "abs_resid"] = 1e9
    poisoned.loc[mask, "actual"] = poisoned.loc[mask, "projected"] + 1e9
    rerun = evaluate_stat_trial2(poisoned, calib, min_rolling_residuals=30)

    # The poisoned day's own half-widths (every method) are untouched: the
    # window is strictly before the day, the sigma fit and edges never saw
    # the evaluation segment at all.
    for method in ("champion", "scaled", "mondrian"):
        a = base["eval_rows"].loc[base["eval_rows"]["game_date"] == target_day, f"hw_{method}"]
        b = rerun["eval_rows"].loc[rerun["eval_rows"]["game_date"] == target_day, f"hw_{method}"]
        assert list(a) == list(b), method
    # ...and the NEXT day's champion half-width DOES move: the poison enters
    # the trailing window once it is strictly in the past. The pin is that
    # the timing is exactly this, not that the window ignores history.
    next_day = days[16]
    a = base["eval_rows"].loc[base["eval_rows"]["game_date"] == next_day, "hw_champion"].iloc[0]
    b = rerun["eval_rows"].loc[rerun["eval_rows"]["game_date"] == next_day, "hw_champion"].iloc[0]
    assert b > a


def test_calibration_must_be_a_prefix():
    frame = _frame()
    days = sorted(frame["game_date"].unique())
    with pytest.raises(ValueError):
        evaluate_stat_trial2(frame, days[2:10], min_rolling_residuals=30)


def test_scaled_flattens_the_shape_in_a_growing_volatility_world():
    frame = _frame(days=40, per_day=120, scale_fn=lambda p: 0.3 + 0.25 * p)
    calib = _calib_days(frame, n=10)
    res = evaluate_stat_trial2(frame, calib, min_rolling_residuals=50)
    terciles = tercile_coverage_methods(
        res["eval_rows"], {"champion": "hw_champion", "scaled": "hw_scaled"}
    )
    champ = [t["coverage_80"]["champion"] for t in terciles]
    scaled = [t["coverage_80"]["scaled"] for t in terciles]
    # The planted defect: the champion's coverage falls as projections grow.
    assert champ[0] > champ[2]
    # The scaled method's spread across terciles is the smaller one.
    assert (max(scaled) - min(scaled)) < (max(champ) - min(champ))


def test_mondrian_fallback_fires_and_is_counted_when_a_bin_runs_thin():
    # Calibration projections span 0..30, but every evaluation projection is
    # tiny: the window's high bin is empty, so the fallback must serve it.
    calib_frame = _frame(days=8, per_day=100)
    eval_rows = []
    rng = np.random.default_rng(3)
    for d in range(8, 20):
        projected = rng.uniform(0.0, 2.0, 100)
        resid = rng.normal(0.0, 2.0, 100)
        eval_rows.append(
            pd.DataFrame(
                {
                    "game_date": f"2025-02-{d + 1:02d}",
                    "projected": projected,
                    "actual": projected + resid,
                    "abs_resid": np.abs(resid),
                }
            )
        )
    frame = pd.concat([calib_frame, *eval_rows], ignore_index=True)
    calib = _calib_days(frame)
    res = evaluate_stat_trial2(frame, calib, min_rolling_residuals=50, mondrian_min_bin=100)
    assert res["mondrian_fallback_bin_days"] > 0
    # Fallback rows are priced with the champion's half-width that day.
    rows = res["eval_rows"]
    high = rows[rows["mondrian_bin"] == 2]
    if len(high):
        assert (high["hw_mondrian"] == high["hw_champion"]).all()


def test_time_thirds_and_daily_records_cover_all_methods():
    frame = _frame()
    res = evaluate_stat_trial2(frame, _calib_days(frame), min_rolling_residuals=30)
    thirds = time_third_coverage_methods(res["daily"])
    assert [t["third"] for t in thirds] == ["early", "middle", "late"]
    for t in thirds:
        assert set(t["coverage_80"]) == {"champion", "scaled", "mondrian"}
        assert all(0.0 <= c <= 1.0 for c in t["coverage_80"].values())


# ---------------------------------------------------------------------------
# The frozen verdict rule.
# ---------------------------------------------------------------------------


def _verdict_input(**overrides):
    base = {}
    for stat in ("points", "rebounds", "assists", "threes"):
        base[stat] = {
            "champion": {"coverage": 0.798, "mean_width": 10.0},
            "challenger": {"coverage": 0.80, "mean_width": 10.5},
            "terciles": [0.80, 0.81, 0.79],
        }
    for stat, patch in overrides.items():
        base[stat].update(patch)
    return base


def test_verdict_passes_when_every_clause_holds():
    assert shape_fix_verdict(_verdict_input())["fixes_shape"] is True


def test_verdict_clause1_needs_three_stats_in_the_band():
    bad = {"challenger": {"coverage": 0.80, "mean_width": 10.5}, "terciles": [0.95, 0.80, 0.65],
           "champion": {"coverage": 0.798, "mean_width": 10.0}}
    verdict = shape_fix_verdict(_verdict_input(points=bad, rebounds=bad))
    assert verdict["clauses"]["clause1_stats_passing"] == 2
    assert verdict["fixes_shape"] is False


def test_verdict_clause2_catches_an_abandoned_stat():
    bad = {"challenger": {"coverage": 0.80, "mean_width": 10.5}, "terciles": [0.86, 0.80, 0.69],
           "champion": {"coverage": 0.798, "mean_width": 10.0}}
    verdict = shape_fix_verdict(_verdict_input(threes=bad))
    assert verdict["clauses"]["clause2_no_stat_abandoned"] is False
    assert verdict["fixes_shape"] is False


def test_verdict_clause3_and_clause4_are_real_gates():
    wide = {"challenger": {"coverage": 0.80, "mean_width": 11.2}, "terciles": [0.80, 0.81, 0.79],
            "champion": {"coverage": 0.798, "mean_width": 10.0}}
    assert shape_fix_verdict(_verdict_input(points=wide))["fixes_shape"] is False
    off = {"challenger": {"coverage": 0.845, "mean_width": 10.5}, "terciles": [0.80, 0.81, 0.79],
           "champion": {"coverage": 0.798, "mean_width": 10.0}}
    assert shape_fix_verdict(_verdict_input(points=off))["fixes_shape"] is False
