"""Live 80%-range coverage split by projected-value tercile.

The pooled coverage figure can hide a band the range misses, so the
aggregates also publish the same inside flags cut into three projection
bands. These tests pin the cuts, the counts and the percentages, and the
honest silence below 30 ranged lines.
"""

from __future__ import annotations

import statistics

import pytest

from nba_predictor.tracking.prop_grading import _prop_aggregates, _tercile_coverage


def _rows(predicted: list[float], inside: list[bool]) -> list[dict]:
    """Pre-tip graded rows for one stat, each carrying the inside flag we
    want it to have. `delta`/`grade` exist only so the other aggregate
    fields compute; they play no part in the tercile split."""
    return [
        {
            "game_date": "2026-11-01",
            "stat": "points",
            "predicted": float(p),
            "delta": 0.0,
            "grade": None,
            "made_before_tip": True,
            "inside_interval": bool(ok),
        }
        for p, ok in zip(predicted, inside)
    ]


def test_tercile_coverage_cuts_counts_and_percentages():
    predicted = list(range(1, 31))
    # low: all 10 inside; mid: 8 of 10; high: 7 of 10.
    inside = [True] * 10 + [True] * 8 + [False] * 2 + [True] * 7 + [False] * 3
    agg = _prop_aggregates(_rows(predicted, inside), None)
    stat = next(s for s in agg["per_stat"] if s["stat"] == "points")

    q1, q2 = 10.666666666666666, 20.333333333333332
    # The cuts are the inclusive tertiles of the stat's ranged lines.
    assert (q1, q2) == pytest.approx(
        statistics.quantiles(predicted, n=3, method="inclusive")
    )

    by = stat["interval_coverage_by_tercile"]
    assert by is not None
    assert [t["tercile"] for t in by] == ["low", "mid", "high"]
    assert by[0]["upper"] == pytest.approx(q1)
    assert by[1]["upper"] == pytest.approx(q2)
    assert by[2]["upper"] is None
    assert [(t["n"], t["coverage"]) for t in by] == [
        (10, 100.0),
        (10, 80.0),
        (10, 70.0),
    ]
    # The pooled figure sits beside the split, unchanged.
    assert stat["interval_n"] == 30
    assert stat["interval_coverage"] == pytest.approx(round(100.0 * 25 / 30, 1))


def test_fewer_than_30_ranged_lines_yields_no_split():
    rows = _rows(list(range(1, 30)), [True] * 29)  # 29 ranged lines
    stat = _prop_aggregates(rows, None)["per_stat"][0]
    assert stat["interval_n"] == 29
    assert stat["interval_coverage_by_tercile"] is None
    assert _tercile_coverage(rows) is None


def test_an_empty_tercile_has_null_coverage_not_zero():
    # Heavy ties at the bottom push the middle tertile between 0 and its
    # upper cut, leaving it with no lines at all.
    rows = _rows([0.0] * 20 + [100.0] * 10, [True] * 30)
    by = _tercile_coverage(rows)
    assert by is not None
    mid = next(t for t in by if t["tercile"] == "mid")
    assert mid["n"] == 0
    assert mid["coverage"] is None
