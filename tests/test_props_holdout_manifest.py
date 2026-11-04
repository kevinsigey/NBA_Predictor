"""The committed props manifest reports a season holdout, and the post-game bands read that holdout.

The numbers come from docs/props-holdout-mae.json (written by scripts/props_holdout_mae.py). The manifest's
per-stat `mae` is the latest-season holdout; `in_sample_mae` is the shipped pickles scored on their own rows.
"""

import json
from pathlib import Path

import pytest

from nba_predictor.api.routes import _prop_bands

ROOT = Path(__file__).resolve().parents[1]
STATS = ["points", "rebounds", "assists", "threes"]
MANIFEST = json.loads((ROOT / "models" / "player_props_manifest.json").read_text())
HOLDOUT = json.loads((ROOT / "docs" / "props-holdout-mae.json").read_text())
LATEST = HOLDOUT["holdouts"]["latest"]


def test_manifest_declares_its_mae_as_a_holdout_not_in_sample():
    assert MANIFEST["training"]["in_sample_metrics"] is False
    assert MANIFEST["training"]["holdout_season"] == LATEST["holdout_season"]
    assert "holdout" in MANIFEST["training"]["holdout_method"]


@pytest.mark.parametrize("stat", STATS)
def test_manifest_mae_is_the_holdout_mae_and_in_sample_is_kept_separately(stat):
    metrics = MANIFEST["metrics"][stat]
    assert metrics["mae"] == pytest.approx(LATEST["markets"][stat]["holdout_mae"])
    assert metrics["in_sample_mae"] is not None


def test_prop_bands_returns_the_holdout_values():
    bands = _prop_bands(ROOT / "models")

    assert set(bands) == set(STATS)
    for stat in STATS:
        assert bands[stat] == pytest.approx(LATEST["markets"][stat]["holdout_mae"])


def test_holdout_is_strictly_after_its_training_seasons():
    assert LATEST["train_max_date"] < LATEST["holdout_min_date"]
