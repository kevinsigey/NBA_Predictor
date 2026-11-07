"""The decomposition helpers behind scripts/nba_live_gap.py.

The gap split has to add up: the per-slice contributions must sum to the pooled live MAE minus the
holdout MAE, or the report would be explaining a number that is not the gap.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nba_live_gap.py"
_spec = importlib.util.spec_from_file_location("nba_live_gap", _SCRIPT)
gap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gap)


def _frame():
    return pd.DataFrame(
        {
            "month": ["2026-10", "2026-10", "2026-11", "2026-11", "2026-11"],
            "err": [2.0, 4.0, 6.0, 8.0, 1.0],
        }
    )


def test_contributions_sum_to_pooled_gap():
    df = _frame()
    holdout = 3.0
    table, total = gap.gap_contributions(df, "month", "err", holdout)
    assert total == pytest.approx(df["err"].mean() - holdout)
    assert table["contribution"].sum() == pytest.approx(total)


def test_slice_table_has_expected_counts_and_mae():
    table, _ = gap.gap_contributions(_frame(), "month", "err", 3.0)
    by_month = table.set_index("month")
    assert by_month.loc["2026-10", "n"] == 2
    assert by_month.loc["2026-10", "mae"] == pytest.approx(3.0)
    assert by_month.loc["2026-11", "mae"] == pytest.approx(5.0)
    assert by_month.loc["2026-11", "share_of_rows"] == pytest.approx(3 / 5)


def test_minutes_and_games_buckets():
    mins = pd.Series([10.0, 15.0, 24.9, 25.0, np.nan])
    assert list(gap.minutes_bucket(mins)[:4]) == ["<15 min", "15-24.9 min", "15-24.9 min", "25+ min"]
    gp = pd.Series([0.0, 3.0, 4.0, 9.0, 10.0])
    assert list(gap.games_bucket(gp)) == ["0 (opener)", "1-4", "1-4", "5-9", "10+"]


def test_paired_bootstrap_sign_and_determinism():
    served = np.array([1.0, 1.0, 1.0, 1.0])
    naive = np.array([2.0, 2.0, 2.0, 2.0])
    mean, lo, hi = gap.paired_bootstrap_diff(served, naive)
    assert mean == pytest.approx(-1.0) and hi < 0
    assert gap.paired_bootstrap_diff(served, naive) == (mean, lo, hi)
