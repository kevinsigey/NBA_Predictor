"""Tests for the conformal-interval race harness (models/conformal_props.py).

The chronology tests are the point of the file: a calibration row dated on
or after an evaluated row must be impossible, and each test below plants
exactly the leak it forbids (an enormous residual where a leaky
implementation would absorb it) and asserts the number does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.conformal_props import (
    conformal_half_width,
    evaluate_stat,
    interval_stats,
    method_verdict,
    tercile_coverage,
    time_third_coverage,
    _trailing_window,
)


# ---------------------------------------------------------------------------
# The quantile convention: k = ceil((n + 1) * level), an order statistic.
# ---------------------------------------------------------------------------


def test_quantile_uses_finite_sample_correction():
    values = np.arange(1, 101, dtype=float)  # 1..100
    assert conformal_half_width(values, 0.8) == 81.0  # ceil(101 * 0.8) = 81
    assert conformal_half_width(values, 0.5) == 51.0  # ceil(101 * 0.5) = 51
    assert conformal_half_width(values, 0.9) == 91.0  # ceil(101 * 0.9) = 91


def test_quantile_caps_at_the_largest_residual():
    values = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_half_width(values, 0.9) == 9.0  # ceil(10 * 0.9) = 9 = n
    assert conformal_half_width([40.0, 10.0, 30.0, 20.0], 0.5) == 30.0  # ceil(5*.5)=3rd


def test_quantile_rejects_empty_and_bad_level():
    with pytest.raises(ValueError):
        conformal_half_width([], 0.8)
    with pytest.raises(ValueError):
        conformal_half_width([1.0, 2.0], 1.0)


def test_interval_stats_counts_the_boundary_as_inside():
    stats = interval_stats(np.zeros(3), np.array([0.5, 1.0, 2.5]), 1.0)
    assert stats["coverage"] == pytest.approx(2 / 3)
    assert stats["mean_width"] == 2.0
    assert stats["n"] == 3


# ---------------------------------------------------------------------------
# Chronology: the leakage pins.
# ---------------------------------------------------------------------------


def _frame(days: list[str], rows_per_day: int, resid: float = 1.0) -> pd.DataFrame:
    rows = []
    for day in days:
        for _ in range(rows_per_day):
            rows.append(
                {
                    "game_date": day,
                    "projected": 10.0,
                    "actual": 10.0 + resid,
                    "abs_resid": abs(resid),
                }
            )
    return pd.DataFrame(rows)


def _days(n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-01", periods=n)]


def test_trailing_window_never_includes_the_scored_day():
    days = _days(5)
    frame = _frame(days, 4)
    window = _trailing_window(frame, days[3], window_days=14)
    assert sorted(window["game_date"].unique()) == days[:3]


def test_rolling_half_width_ignores_the_scored_days_own_rows():
    days = _days(20)
    frame = _frame(days, 10, resid=1.0)
    # Plant a leak tripwire: enormous residuals ON an evaluation day. A
    # rolling window that touched its own day would jump to ~1000.
    frame.loc[frame["game_date"] == days[9], ["actual", "abs_resid"]] = [1010.0, 1000.0]
    result = evaluate_stat(
        frame, days[:5], band_mae=2.0, rolling_window_days=14, min_rolling_residuals=10
    )
    eval_rows = result["eval_rows"]
    planted = eval_rows[eval_rows["game_date"] == days[9]]
    assert (planted["hw_rolling"] == 1.0).all()
    # And the planted day therefore covers nothing -- the honest consequence.
    day = next(d for d in result["daily"] if d["date"] == days[9])
    assert day["rolling_coverage_80"] == 0.0


def test_split_half_width_ignores_the_whole_evaluation_segment():
    days = _days(20)
    frame = _frame(days, 10, resid=1.0)
    frame.loc[frame["game_date"].isin(days[:5]), ["actual", "abs_resid"]] = [12.0, 2.0]
    frame.loc[frame["game_date"].isin(days[5:]), ["actual", "abs_resid"]] = [5010.0, 5000.0]
    result = evaluate_stat(
        frame, days[:5], band_mae=2.0, rolling_window_days=14, min_rolling_residuals=10
    )
    # Calibration residuals are all 2.0; evaluation noise cannot move these.
    assert result["split_half_widths"] == {"0.5": 2.0, "0.8": 2.0, "0.9": 2.0}


def test_calibration_must_be_a_prefix_of_the_days():
    days = _days(10)
    frame = _frame(days, 10)
    with pytest.raises(ValueError):
        evaluate_stat(frame, days[1:6], band_mae=2.0, min_rolling_residuals=10)
    # A calibration set reaching INTO the evaluation segment is the same defect.
    with pytest.raises(ValueError):
        evaluate_stat(frame, days[:5] + [days[7]], band_mae=2.0, min_rolling_residuals=10)


def test_evaluate_stat_reports_every_method_and_day():
    days = _days(30)
    frame = _frame(days, 20, resid=1.5)
    result = evaluate_stat(
        frame, days[:8], band_mae=2.0, rolling_window_days=14, min_rolling_residuals=10
    )
    assert result["n_calibration"] == 160
    assert result["n_eval"] == 440
    assert len(result["daily"]) == 22
    for key in ("mae_band_1x", "mae_band_2x", "split_0.8", "rolling_0.8", "split_0.5", "rolling_0.9"):
        assert key in result["methods"]
        assert 0.0 <= result["methods"][key]["coverage"] <= 1.0


# ---------------------------------------------------------------------------
# The verdict rule: coverage near nominal AND not winning by width.
# ---------------------------------------------------------------------------


def _methods(split_cov: float, split_width: float) -> dict:
    return {
        "mae_band_1x": {"coverage": 0.55, "mean_width": 9.26, "n": 100},
        "mae_band_2x": {"coverage": 0.84, "mean_width": 18.52, "n": 100},
        "split_0.8": {"coverage": split_cov, "mean_width": split_width, "n": 100},
    }


def test_verdict_win_against_the_wider_covering_band():
    v = method_verdict(_methods(0.79, 10.0), "split_0.8", band_mae=4.63)
    # Only the 2x band covers >= 0.79, so it is the comparator; 10 <= 18.52.
    assert v["win"] and v["comparator_band"] == "2x" and not v["comparator_was_fallback"]


def test_verdict_fails_when_coverage_is_off_nominal():
    v = method_verdict(_methods(0.70, 10.0), "split_0.8", band_mae=4.63)
    assert not v["win"] and not v["clause_a"]


def test_verdict_fails_when_wider_than_the_comparator():
    v = method_verdict(_methods(0.81, 20.0), "split_0.8", band_mae=4.63)
    assert not v["win"] and v["clause_a"] and not v["clause_b"]


def test_verdict_compares_against_the_narrowest_covering_band():
    methods = _methods(0.79, 10.0)
    methods["mae_band_1x"] = {"coverage": 0.80, "mean_width": 9.26, "n": 100}
    v = method_verdict(methods, "split_0.8", band_mae=4.63)
    # Now the 1x band also covers enough, and 10 > 9.26: winning by width is out.
    assert v["comparator_band"] == "1x" and not v["win"]


def test_verdict_fallback_when_no_band_covers_as_much():
    methods = _methods(0.82, 15.0)
    methods["mae_band_2x"] = {"coverage": 0.80, "mean_width": 18.52, "n": 100}
    v = method_verdict(methods, "split_0.8", band_mae=4.63)
    assert v["comparator_was_fallback"] and v["comparator_band"] == "2x" and v["win"]


# ---------------------------------------------------------------------------
# Diagnostics shape.
# ---------------------------------------------------------------------------


def test_tercile_coverage_partitions_the_rows():
    rng = np.random.default_rng(7)
    n = 300
    eval_rows = pd.DataFrame(
        {
            "projected": rng.uniform(0, 30, n),
            "abs_resid": rng.uniform(0, 5, n),
            "hw_split": 3.0,
            "hw_rolling": 2.5,
            "hw_band_1x": 4.63,
        }
    )
    terciles = tercile_coverage(eval_rows)
    assert [t["tercile"] for t in terciles] == ["low", "mid", "high"]
    assert sum(t["n"] for t in terciles) == n


def test_time_thirds_weight_coverage_by_rows():
    daily = [
        {"date": f"2025-01-{i + 1:02d}", "n": 10, "mae": 1.0,
         "split_coverage_80": 0.5, "rolling_coverage_80": 0.9, "band_1x_coverage": 0.4}
        for i in range(9)
    ]
    thirds = time_third_coverage(daily)
    assert sum(t["n"] for t in thirds) == 90
    assert thirds[0]["split_coverage_80"] == pytest.approx(0.5)
    assert thirds[2]["rolling_coverage_80"] == pytest.approx(0.9)
