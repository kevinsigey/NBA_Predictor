"""Production conformal intervals (tracking/conformal_intervals.py) and
their integration into the grading payloads.

The chronology pins are the point, as in the race harnesses: a game's
intervals are calibrated on counted residuals from game days STRICTLY
before its own day. Each test plants exactly the leak it forbids -- a
residual dated the same day, or a later day, with an enormous miss where
a leaky implementation would absorb it -- and asserts the interval does
not move. The fallback tiers are pinned in order, and the aggregates'
coverage is recomputed from the per-game payloads it must agree with:
the coverage counts the same intervals the review showed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nba_predictor.tracking.conformal_intervals import (
    BASIS_HOLDOUT,
    BASIS_ROLLING,
    BASIS_SEASON,
    MIN_RESIDUALS,
    PropIntervalCalibrator,
    interval_for,
)


def _residuals(days: int, per_day: int, *, stat: str = "points", start: int = 1,
               scale: float = 3.0, seed: int = 5) -> list[dict]:
    """`days` game days of counted residuals for one stat, projections
    spread 2..30, misses ~ N(0, scale) -- unless `scale` is callable, in
    which case it maps projection -> miss scale (a growing-volatility
    world for the shape tests)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        date = f"2026-10-{start + d:02d}"
        for _ in range(per_day):
            projected = float(rng.uniform(2.0, 30.0))
            s = scale(projected) if callable(scale) else scale
            rows.append(
                {
                    "game_date": date,
                    "stat": stat,
                    "projected": projected,
                    "abs_resid": abs(float(rng.normal(0.0, s))),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# The fallback chain, tier by tier.
# ---------------------------------------------------------------------------


def test_rolling_tier_when_the_window_is_full():
    rows = _residuals(20, 30)  # 14-day window = 420 residuals
    cal = PropIntervalCalibrator(rows)
    hw, info = cal.half_width("points", 15.0, "2026-10-21")
    assert hw is not None and hw > 0
    assert info == {"basis": BASIS_ROLLING, "n": 420, "window_days": 14}


def test_season_tier_when_recent_days_run_thin():
    # Six fat days (600 residuals) then fourteen thin ones (10/day): on day
    # 21 the trailing window holds 140 < 300 but the season holds 740.
    rows = _residuals(6, 100, start=1) + _residuals(14, 10, start=7)
    cal = PropIntervalCalibrator(rows)
    _hw, info = cal.half_width("points", 15.0, "2026-10-21")
    assert info["basis"] == BASIS_SEASON
    assert info["n"] == 740
    # On day 7 the window is the six fat days alone: rolling, 6 days.
    _hw, info = cal.half_width("points", 15.0, "2026-10-07")
    assert info["basis"] == BASIS_ROLLING
    assert info["window_days"] == 6


def test_holdout_tier_from_the_manifest_parameters():
    cold = {
        "points": {
            "q80": 1.5,
            "sigma": {"edges": [], "scales": [2.0], "floor": 0.5},
            "n_holdout": 1234,
        }
    }
    cal = PropIntervalCalibrator(_residuals(2, 10), cold)
    hw, info = cal.half_width("points", 15.0, "2026-10-03")
    assert info["basis"] == BASIS_HOLDOUT
    assert info["n"] == 1234
    assert hw == pytest.approx(1.5 * 2.0)  # q80 x sigma, the published pair


def test_thin_record_falls_back_to_a_measured_symmetric_quantile():
    rows = _residuals(1, 5)
    cal = PropIntervalCalibrator(rows)  # no cold start anywhere
    hw, info = cal.half_width("points", 15.0, "2026-10-02")
    assert info["basis"] == BASIS_SEASON
    assert info["n"] == 5
    # k = ceil((5 + 1) * 0.8) = 5: the largest of the five residuals.
    assert hw == pytest.approx(max(r["abs_resid"] for r in rows))


def test_no_record_and_no_parameters_means_no_interval():
    cal = PropIntervalCalibrator([])
    hw, info = cal.half_width("points", 15.0, "2026-10-02")
    assert hw is None
    assert info["basis"] is None
    low, high, _ = interval_for(cal, "points", 15.0, "2026-10-02")
    assert low is None and high is None


def test_min_residuals_is_the_tier_boundary():
    rows = _residuals(10, 30)  # exactly 300 prior residuals over 10 days
    cal = PropIntervalCalibrator(rows)
    _hw, info = cal.half_width("points", 15.0, "2026-10-11")
    assert info["n"] == MIN_RESIDUALS
    assert info["basis"] == BASIS_ROLLING  # the window IS the season here
    thinner = PropIntervalCalibrator(rows[:-1])
    _hw, info = thinner.half_width("points", 15.0, "2026-10-11")
    assert info["n"] == MIN_RESIDUALS - 1
    assert info["basis"] == BASIS_SEASON  # tier 4's symmetric fallback


# ---------------------------------------------------------------------------
# Chronology: the leakage pins.
# ---------------------------------------------------------------------------


def test_future_and_same_day_residuals_never_move_an_interval():
    rows = _residuals(20, 30)
    cal = PropIntervalCalibrator(rows)
    hw_before, _ = cal.half_width("points", 15.0, "2026-10-15")

    poisoned = rows + [
        {"game_date": "2026-10-15", "stat": "points", "projected": 15.0, "abs_resid": 1e9},
        {"game_date": "2026-10-16", "stat": "points", "projected": 15.0, "abs_resid": 1e9},
        {"game_date": "2026-12-25", "stat": "points", "projected": 15.0, "abs_resid": 1e9},
    ]
    cal2 = PropIntervalCalibrator(poisoned)
    hw_after, _ = cal2.half_width("points", 15.0, "2026-10-15")
    assert hw_after == hw_before

    # A residual from the day BEFORE does enter the calibration -- the pin
    # is the timing, not blindness to history.
    poisoned_past = rows + [
        {"game_date": "2026-10-14", "stat": "points", "projected": 15.0, "abs_resid": 1e9}
    ]
    cal3 = PropIntervalCalibrator(poisoned_past)
    hw_past, _ = cal3.half_width("points", 15.0, "2026-10-15")
    assert hw_past > hw_before


def test_the_interval_scales_with_the_projection():
    # A world whose misses grow with the line: the star's range must be
    # wider than the scrub's, on the same day, from the same calibration.
    rows = _residuals(20, 40, scale=lambda p: 0.5 + 0.25 * p)
    cal = PropIntervalCalibrator(rows)
    low_hw, _ = cal.half_width("points", 3.0, "2026-10-21")
    high_hw, _ = cal.half_width("points", 28.0, "2026-10-21")
    assert high_hw > 2.0 * low_hw
    assert low_hw > 0  # the sigma floor: a small line still gets a range


# ---------------------------------------------------------------------------
# The manifest cold-start reader.
# ---------------------------------------------------------------------------


def test_cold_start_reader_takes_only_complete_blocks(tmp_path):
    from nba_predictor.tracking.prop_grading import prop_conformal_cold_start

    (tmp_path / "player_props_manifest.json").write_text(json.dumps({
        "metrics": {
            "points": {"mae": 4.6, "conformal": {
                "q80": 1.21,
                "sigma": {"edges": [10.0], "scales": [2.0, 3.0], "floor": 0.4},
                "n_holdout": 900,
            }},
            "rebounds": {"mae": 1.9, "conformal": None},
            "assists": {"mae": 1.35, "conformal": {"q80": 1.1}},  # incomplete
            "threes": {"mae": 0.9},
        },
    }))
    out = prop_conformal_cold_start(tmp_path)
    assert set(out) == {"points"}
    assert out["points"]["q80"] == 1.21
    assert out["points"]["n_holdout"] == 900
    assert prop_conformal_cold_start(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# Integration: the grading payloads carry the same intervals throughout.
# ---------------------------------------------------------------------------


def _seed_store(tmp_path, days: int = 16, players: int = 25):
    """A counted record over `days` game days: one game a day, `players`
    players, all four stats, every pick provably pre-tip. Misses scale
    with the projection, as the real record's do. Returns the db path,
    the schedule, and the raw {(game_id, player, stat): (projected,
    actual)} the store was seeded with."""
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    rng = np.random.default_rng(11)
    schedule = []
    raw = {}
    outcome_rows: list[tuple] = []
    for d in range(days):
        date = f"2026-10-{3 + d:02d}"
        game_id = f"g{d}"
        schedule.append({
            "game_id": game_id, "game_date": date, "home_team": "BOS",
            "away_team": "MIA", "tip_off": f"{date}T19:00:00+00:00",
            "completed": True,
        })
        picks, outcomes = [], []
        for p in range(players):
            for stat, base in (("points", 14.0), ("rebounds", 6.0), ("assists", 4.0), ("threes", 1.6)):
                projected = float(max(0.5, rng.normal(base, base * 0.45)))
                actual = float(max(0.0, projected + rng.normal(0.0, 0.6 + 0.22 * projected)))
                raw[(game_id, f"p{p}", stat)] = (projected, actual)
                picks.append({
                    "game_id": game_id, "player_id": f"p{p}", "stat": stat,
                    "predicted_value": projected,
                    "created_at": f"{date}T12:00:00+00:00", "position": None,
                })
                outcomes.append(
                    (game_id, f"p{p}", stat, actual, f"{date}T23:00:00+00:00")
                )
        store.insert_player_predictions(db_path, picks)
        outcome_rows.extend(outcomes)
    # Outcomes go in as ONE bulk insert: the store's per-row writer pays a
    # connection setup per call (~80ms), which is the right trade for the
    # ingest's trickle and the wrong one for a 1,600-row fixture.
    with store.get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO game_player_outcomes"
            " (game_id, player_id, stat, actual_value, recorded_at)"
            " VALUES (?, ?, ?, ?, ?)",
            outcome_rows,
        )
        conn.commit()  # get_connection closes without committing; the store's own writers commit explicitly
    return db_path, schedule, raw


BANDS = {"points": 4.0, "rebounds": 2.0, "assists": 1.0, "threes": 0.5}


def test_game_payload_rows_carry_the_range_and_the_verdict(tmp_path):
    """Every row's range is EXACTLY what an independent calibrator build
    over the same store produces for that row -- the review, the day
    payload and the surface payloads are one interval, not three."""
    from nba_predictor.tracking.prop_grading import (
        build_game_prop_grading,
        build_interval_calibrator,
    )

    db_path, schedule, raw = _seed_store(tmp_path)
    last = schedule[-1]
    grading = build_game_prop_grading(db_path, schedule, BANDS, last["game_id"])
    assert grading is not None
    assert grading["interval_basis"] is not None
    assert "trailing 14 game days" in grading["interval_basis"]

    calibrator = build_interval_calibrator(db_path, schedule)
    for row in grading["rows"]:
        projected_raw, actual_raw = raw[(last["game_id"], row["player_id"], row["stat"])]
        hw, _info = calibrator.half_width(row["stat"], row["predicted"], last["game_date"])
        assert hw is not None
        # The served low bound floors at zero (the stats are counts); the
        # verdict is computed against the unfloored half-width.
        assert row["interval_low"] == round(max(0.0, row["predicted"] - hw), 1)
        assert row["interval_high"] == round(row["predicted"] + hw, 1)
        assert row["inside_interval"] == (abs(actual_raw - projected_raw) <= hw + 1e-9)


def test_aggregates_coverage_counts_exactly_the_review_rows(tmp_path):
    """The honesty loop, pinned: recompute the aggregates' coverage from
    the per-game payloads (the intervals readers actually saw), over the
    same pre-tip population, and require the same number."""
    from nba_predictor.tracking.prop_grading import (
        build_game_prop_grading,
        build_prop_grading,
    )

    db_path, schedule, _raw = _seed_store(tmp_path)
    payload = build_prop_grading(db_path, schedule, BANDS)
    aggregates = payload["aggregates"]
    assert aggregates["interval_calibration"] is not None

    # One build per game; bucket its rows by stat in a single pass.
    counts: dict[str, list[int]] = {}
    for game in schedule:
        grading = build_game_prop_grading(db_path, schedule, BANDS, game["game_id"])
        if grading is None:
            continue
        for row in grading["rows"]:
            if not row["made_before_tip"] or row["inside_interval"] is None:
                continue
            bucket = counts.setdefault(row["stat"], [0, 0])
            bucket[1] += 1
            bucket[0] += int(row["inside_interval"])

    for stat_entry in aggregates["per_stat"]:
        stat = stat_entry["stat"]
        inside, total = counts.get(stat, [0, 0])
        assert stat_entry["interval_n"] == total
        if total:
            assert stat_entry["interval_coverage"] == pytest.approx(
                round(100.0 * inside / total, 1), abs=0.05
            )
        # Coverage of a conformal 80% interval, on data shaped like the
        # record: it must land in a sane band, not at 0 or 100.
        assert 40.0 < stat_entry["interval_coverage"] < 100.0


def test_a_later_days_outcomes_never_move_an_earlier_games_ranges(tmp_path):
    from nba_predictor.tracking import store
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path, schedule, _raw = _seed_store(tmp_path)
    target = schedule[10]
    before = build_game_prop_grading(db_path, schedule, BANDS, target["game_id"])
    ranges_before = {
        (r["player_id"], r["stat"]): (r["interval_low"], r["interval_high"])
        for r in before["rows"]
    }
    # Poison the FUTURE: the last day's outcomes become absurd.
    last = schedule[-1]
    with store.get_connection(db_path) as conn:
        conn.execute(
            "UPDATE game_player_outcomes SET actual_value = 9999.0 WHERE game_id = ?",
            (last["game_id"],),
        )
    after = build_game_prop_grading(db_path, schedule, BANDS, target["game_id"])
    ranges_after = {
        (r["player_id"], r["stat"]): (r["interval_low"], r["interval_high"])
        for r in after["rows"]
    }
    assert ranges_after == ranges_before


def test_the_first_days_have_no_interval_without_a_cold_start(tmp_path):
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path, schedule, _raw = _seed_store(tmp_path, days=3)
    grading = build_game_prop_grading(db_path, schedule, BANDS, schedule[0]["game_id"])
    assert grading is not None
    assert grading["interval_basis"] is None
    assert all(r["interval_low"] is None for r in grading["rows"])
    assert all(r["inside_interval"] is None for r in grading["rows"])


def test_a_cold_start_gives_the_first_day_a_labelled_range(tmp_path):
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path, schedule, _raw = _seed_store(tmp_path, days=3)
    cold = {
        stat: {
            "q80": 1.3,
            "sigma": {"edges": [], "scales": [mae], "floor": mae * 0.2},
            "n_holdout": 5000,
        }
        for stat, mae in BANDS.items()
    }
    grading = build_game_prop_grading(
        db_path, schedule, BANDS, schedule[0]["game_id"], cold_start=cold
    )
    assert grading is not None
    assert grading["interval_basis"] is not None
    assert "holdout" in grading["interval_basis"]
    assert all(r["interval_low"] is not None for r in grading["rows"])


# ---------------------------------------------------------------------------
# The served surface: /games/{id}/players carries the range, and the
# projection it brackets is the pick's own value, untouched.
# ---------------------------------------------------------------------------


def _client(tmp_path, db_path, schedule):
    import json as _json

    from fastapi.testclient import TestClient

    from nba_predictor.api import deps
    from nba_predictor.api.app import app

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(_json.dumps(schedule))
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []
    app.dependency_overrides[deps.get_models_dir] = lambda: models_dir
    return TestClient(app)


def test_players_route_serves_ranges_bracketing_the_projection(tmp_path):
    from nba_predictor.api.app import app

    db_path, schedule, raw = _seed_store(tmp_path)
    try:
        client = _client(tmp_path, db_path, schedule)
        last = schedule[-1]
        response = client.get(f"/games/{last['game_id']}/players")
        assert response.status_code == 200
        rows = response.json()
        assert rows, "the seeded game must serve its projections"
        for row in rows:
            projected_raw, _actual = raw[(last["game_id"], row["player_id"], row["stat"])]
            # The projection is the pick's own value, byte-identical: the
            # interval is metadata beside it, never a part of it.
            assert row["predicted_value"] == projected_raw
            assert row["interval_low"] is not None and row["interval_high"] is not None
            assert row["interval_low"] <= row["predicted_value"] <= row["interval_high"]
            width = row["interval_high"] - row["interval_low"]
            assert 1.0 < width < 60.0  # sane, in the neighbourhood of the races' widths

        # The season's first game: no prior record, an empty models dir
        # (no cold start) -- the projection stands without a range.
        first = schedule[0]
        response = client.get(f"/games/{first['game_id']}/players")
        assert response.status_code == 200
        for row in response.json():
            assert row["interval_low"] is None and row["interval_high"] is None
    finally:
        app.dependency_overrides.clear()
