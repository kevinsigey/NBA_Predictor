"""Regression: upcoming player projections must follow the team's CURRENT roster.

Live bug: CHI v TOR showed 2 of 28 predicted players on CHI's side because the
roster estimate was frozen at the last manual ingest (before CHI's preseason
game). Real data types throughout: schedule rows as in the schedule cache, ESPN
box-score rows, trained XGB player models, a real SQLite tracking store.
"""
import numpy as np
import pandas as pd
import pytest

from nba_predictor.pipeline.ingest import to_player_training_frame, train_player_prop_models
from nba_predictor.pipeline.player_props_refresher import refresh_stale_player_props, stale_upcoming_games
from nba_predictor.tracking import store

@pytest.fixture(autouse=True)
def _no_roster_feed(monkeypatch):
    monkeypatch.setattr("nba_predictor.data.rosters.get_rosters", lambda: None)  # never the network; None = box-score roster rule


NOW = pd.Timestamp("2026-10-21T00:00Z")  # 19h before up1 tips


def _box(pid, name, rng):
    return {
        "player_id": pid, "player_name": name, "team": "CHI", "position": "F",
        "minutes": float(rng.integers(20, 36)), "points": float(rng.integers(8, 30)),
        "rebounds": float(rng.integers(2, 10)), "assists": float(rng.integers(1, 8)),
        "fg_made_attempted": "6-12", "three_made_attempted": "2-5", "ft_made_attempted": "2-2",
    }


def _setup(tmp_path):
    rng = np.random.default_rng(3)
    games, boxes = [], {}
    for i in range(12):  # April: the old roster
        gid = f"a{i}"
        games.append({"game_id": gid, "game_date": f"2026-04-{i + 1:02d}", "tip_off": f"2026-04-{i + 1:02d}T23:00Z",
                      "home_team": "CHI", "away_team": "TOR", "completed": True, "home_pts": 100, "away_pts": 99})
        boxes[gid] = [_box("old", "Tre Jones", rng), _box("new", "Matas Buzelis", rng)]
    # The preseason game after the last manual refresh: only the new roster played.
    games.append({"game_id": "pre1", "game_date": "2026-10-07", "tip_off": "2026-10-08T00:00Z",
                  "home_team": "CHI", "away_team": "PHX", "completed": True, "home_pts": 124, "away_pts": 117})
    boxes["pre1"] = [_box("new", "Matas Buzelis", rng), _box("new2", "Josh Giddey", rng)]
    games.append({"game_id": "up1", "game_date": "2026-10-21", "tip_off": "2026-10-21T19:00Z",
                  "home_team": "TOR", "away_team": "CHI", "completed": False, "home_pts": None, "away_pts": None})
    models_dir = tmp_path / "models"
    train_player_prop_models(to_player_training_frame(games, boxes), models_dir, model_version="t", trained_at="2026-10-01T00:00:00")
    db = tmp_path / "t.db"
    store.init_db(db)
    # The stale estimate: written 2026-10-05, before pre1.
    store.insert_player_prediction(db, game_id="up1", player_id="old", stat="points", predicted_value=12.0, created_at="2026-10-05T10:00:00+00:00")
    return games, boxes, models_dir, db


def _players(db):
    return {r["player_id"] for r in store.get_player_predictions_for_game(db, "up1")}


def test_a_team_that_played_since_the_last_estimate_gets_its_current_roster(tmp_path):
    games, boxes, models_dir, db = _setup(tmp_path)

    stored = refresh_stale_player_props(games, db, models_dir, fetch=lambda gs: {g["game_id"]: boxes[g["game_id"]] for g in gs}, now=NOW, history=pd.DataFrame())

    assert stored > 0
    assert {"new", "new2"} <= _players(db)  # was: only the frozen April roster


def test_a_fresh_estimate_is_left_alone(tmp_path):
    games, boxes, models_dir, db = _setup(tmp_path)
    fetch = lambda gs: {g["game_id"]: boxes[g["game_id"]] for g in gs}
    refresh_stale_player_props(games, db, models_dir, fetch=fetch, now=NOW, history=pd.DataFrame())

    def boom(gs):
        raise AssertionError("fetched box scores for a game that was not stale")

    assert refresh_stale_player_props(games, db, models_dir, fetch=boom, now=NOW, history=pd.DataFrame()) == 0


def test_a_game_outside_the_lead_window_is_not_projected_yet(tmp_path):
    """The earliest row is the counted pick, so nothing is recorded days ahead."""
    games, boxes, models_dir, db = _setup(tmp_path)
    assert stale_upcoming_games(games, db, NOW - pd.Timedelta(11, unit='D')) == []
    assert [g["game_id"] for g in stale_upcoming_games(games, db, NOW)] == ["up1"]


def test_a_projection_is_rescored_after_the_refresh_interval_inside_the_window(tmp_path):
    games, boxes, models_dir, db = _setup(tmp_path)
    store.insert_player_prediction(db, game_id="up1", player_id="new", stat="points", predicted_value=9.0,
                                   created_at="2026-10-20T23:00:00+00:00")  # after pre1: not stale by roster
    assert stale_upcoming_games(games, db, NOW) == []
    assert [g["game_id"] for g in stale_upcoming_games(games, db, NOW + pd.Timedelta(7, unit='h'))] == ["up1"]
