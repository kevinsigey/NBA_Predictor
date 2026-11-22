"""Props roster + projection-guard regressions (2026-10-20 live defects).

Live tonight: a traded player (Jaylen Brown, BOS->PHI in summer 2026) was
projected in TWO games on the same day -- DET-BOS (a game his new team is not
in) and NYK-PHI -- because `to_player_scoring_frame` built each team's roster
independently from that team's single last box score, with no notion of a
player's LATEST team. A star who sat his team's last game vanished from the
roster entirely for the same reason, and the "last" game was whichever came
last in list order, not latest by date (`roster_window=5` was declared and
never used).

The shipped Sep-18 points model also fails a perfect-feature probe (fed a
25.0 points_roll it predicts ~5), so `score_upcoming_player_props` now runs
every prediction through a sanity guard keyed to the player's own rolling
average rather than serving a silently collapsed number.
"""

import pandas as pd

# The hand-fitted stand-in models below must be fitted on the CURRENT feature
# width: score_upcoming_player_props deliberately skips a model whose
# n_features_in_ is not the current set's (see the old-feature-set test at the
# bottom of this file), so a hard-coded width silently turns "the model's
# numbers are guarded" tests into "the model was skipped" non-tests.
from nba_predictor.features.prop_model_frame import PROP_FEATURE_COLUMNS


def _box(pid, name, team, pts=20.0, pos="G"):
    return {
        "player_id": pid, "player_name": name, "team": team, "position": pos,
        "minutes": 30.0, "points": float(pts), "rebounds": 5.0, "assists": 5.0,
        "fg_made_attempted": "8-15", "three_made_attempted": "2-5",
        "ft_made_attempted": "2-2",
    }


def _game(gid, date, home, away, completed):
    return {
        "game_id": gid, "game_date": date, "home_team": home, "away_team": away,
        "completed": completed,
        "home_pts": 100 if completed else None,
        "away_pts": 99 if completed else None,
    }


def _upcoming(df):
    return df[df["points"].isna()]


def test_traded_player_is_projected_only_for_his_latest_team():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        _game("bos_last", "2026-04-10", "BOS", "NYK", True),   # star still on BOS
        _game("phi_pre", "2026-10-15", "PHI", "MIA", True),    # star now on PHI
        _game("det_bos", "2026-10-21", "DET", "BOS", False),
        _game("nyk_phi", "2026-10-21", "NYK", "PHI", False),
    ]
    boxes = {
        "bos_last": [_box("star", "Jaylen Brown", "BOS")],
        "phi_pre": [_box("star", "Jaylen Brown", "PHI")],
    }

    df = to_player_scoring_frame(games, boxes)
    star_rows = _upcoming(df)[_upcoming(df)["player_id"] == "star"]

    assert list(star_rows["game_id"]) == ["nyk_phi"]
    assert list(star_rows["team"]) == ["PHI"]


def test_a_player_is_projected_for_at_most_one_game_per_day():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        _game("bos_last", "2026-04-10", "BOS", "NYK", True),
        _game("phi_pre", "2026-10-15", "PHI", "MIA", True),
        _game("det_bos", "2026-10-21", "DET", "BOS", False),
        _game("nyk_phi", "2026-10-21", "NYK", "PHI", False),
    ]
    boxes = {
        "bos_last": [_box("star", "Jaylen Brown", "BOS")],
        "phi_pre": [_box("star", "Jaylen Brown", "PHI")],
    }

    df = to_player_scoring_frame(games, boxes)
    upcoming = _upcoming(df)
    per_player_day = upcoming.groupby(["player_id", "game_date"]).size()
    assert (per_player_day <= 1).all()


def test_star_who_sat_the_last_game_is_still_on_the_roster():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        _game("g_a", "2026-10-13", "DET", "MIA", True),
        _game("g_b", "2026-10-16", "DET", "MIA", True),  # star rested this one
        _game("g_next", "2026-10-21", "DET", "CHI", False),
    ]
    boxes = {
        "g_a": [_box("cade", "Cade Cunningham", "DET", 24.0)],
        "g_b": [_box("bench", "Bench Guy", "DET", 8.0)],
    }

    df = to_player_scoring_frame(games, boxes)
    roster = set(_upcoming(df)[_upcoming(df)["game_id"] == "g_next"]["player_id"])
    assert roster == {"cade", "bench"}


def test_roster_window_bounds_how_far_back_a_player_counts():
    """A player whose only appearance for the team is older than the team's
    last `roster_window` completed games is not on the estimated roster."""
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = [], {}
    for i in range(6):  # team's last 6 completed games; ghost played only in the oldest
        gid = f"c{i}"
        games.append(_game(gid, f"2026-10-{10 + i:02d}", "DET", "MIA", True))
        boxes[gid] = [_box("regular", "Regular", "DET", 12.0)]
        if i == 0:
            boxes[gid].append(_box("ghost", "Ghost", "DET", 6.0))
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))

    df = to_player_scoring_frame(games, boxes, roster_window=5)
    roster = set(_upcoming(df)[_upcoming(df)["game_id"] == "next"]["player_id"])
    assert roster == {"regular"}


def test_last_completed_game_is_chosen_by_date_not_list_order():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        _game("newer", "2026-10-16", "DET", "MIA", True),
        _game("older", "2026-10-10", "DET", "MIA", True),  # listed last, played first
        _game("next", "2026-10-21", "DET", "CHI", False),
    ]
    boxes = {
        "newer": [_box("newguy", "New Guy", "DET", 15.0)],
        "older": [_box("oldguy", "Old Guy", "DET", 15.0)],
    }

    df = to_player_scoring_frame(games, boxes, roster_window=1)
    roster = set(_upcoming(df)[_upcoming(df)["game_id"] == "next"]["player_id"])
    assert roster == {"newguy"}


# ---------------------------------------------------------------------------
# Projection sanity guard
# ---------------------------------------------------------------------------

def test_guard_clamps_a_collapsed_prediction_up_to_the_players_own_average_band():
    from nba_predictor.pipeline.ingest import guard_player_prediction

    # The live failure shape: a 24-points_roll star served 8.9.
    assert guard_player_prediction("points", 8.9, 24.0) == 0.4 * 24.0
    # A sane prediction passes through untouched.
    assert guard_player_prediction("points", 22.0, 24.0) == 22.0


def test_guard_clamps_an_exploded_and_a_negative_prediction():
    from nba_predictor.pipeline.ingest import guard_player_prediction

    assert guard_player_prediction("assists", 40.0, 10.0) == 2.5 * 10.0
    assert guard_player_prediction("points", -3.0, 10.0) == 0.4 * 10.0


def test_guard_has_nothing_to_say_about_a_near_zero_average():
    from nba_predictor.pipeline.ingest import guard_player_prediction

    assert guard_player_prediction("threes", 0.9, 0.1) == 0.9          # a non-shooter's average is noise, not a band


def test_guard_leaves_a_prediction_alone_when_there_is_no_rolling_average():
    from nba_predictor.pipeline.ingest import guard_player_prediction

    assert guard_player_prediction("points", 8.9, float("nan")) == 8.9
    assert guard_player_prediction("points", 8.9, None) == 8.9


def test_a_collapsed_model_publishes_nothing(tmp_path):
    """End to end: a model that returns a collapsed number for a star does not reach the tracking store at all
    (every projection needed a clamp, far above MAX_CLAMP_SHARE): a pile of "0.4 x average" is not a model."""
    import joblib

    from nba_predictor.pipeline import ingest
    from nba_predictor.tracking import store

    from sklearn.dummy import DummyRegressor

    games, boxes = [], {}
    for i in range(3):
        gid = f"c{i}"
        games.append(_game(gid, f"2026-10-1{i}", "DET", "MIA", True))
        boxes[gid] = [_box("star", "Star", "DET", 25.0)]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for stat in ingest.PLAYER_STAT_TARGET_COLUMNS:
        collapsed = DummyRegressor(strategy="constant", constant=1.0)
        collapsed.fit([[0.0] * len(PROP_FEATURE_COLUMNS)], [1.0])
        joblib.dump(collapsed, models_dir / f"player_{stat}_model.pkl")

    db_path = tmp_path / "t.db"
    store.init_db(db_path)
    stored = ingest.score_upcoming_player_props(games, boxes, models_dir, db_path, "v-test")
    assert stored == 0
    assert store.get_player_predictions_for_game(db_path, "next") == []


def test_clamps_on_deep_bench_players_alone_do_not_block_publication(tmp_path):
    """A 6-minute player's rolling average is noise: a model that misses it must not take the whole slate down."""
    import joblib
    from sklearn.dummy import DummyRegressor

    from nba_predictor.pipeline import ingest
    from nba_predictor.tracking import store

    games, boxes = [], {}
    for i in range(3):
        gid = f"c{i}"
        games.append(_game(gid, f"2026-10-1{i}", "DET", "MIA", True))
        bench = _box("scrub", "Scrub", "DET", 2.0)
        bench["minutes"] = 6.0
        boxes[gid] = [bench]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for stat in ingest.PLAYER_STAT_TARGET_COLUMNS:
        m = DummyRegressor(strategy="constant", constant=30.0)
        m.fit([[0.0] * len(PROP_FEATURE_COLUMNS)], [30.0])
        joblib.dump(m, models_dir / f"player_{stat}_model.pkl")

    db_path = tmp_path / "t.db"
    store.init_db(db_path)
    assert ingest.score_upcoming_player_props(games, boxes, models_dir, db_path, "v-test") > 0


def test_opening_week_serves_last_seasons_average_not_the_models_window(tmp_path):
    """Week one of the new season: the number is last season's average (the model's constant 8 is replaced)."""
    import joblib
    from sklearn.dummy import DummyRegressor

    from nba_predictor.pipeline import ingest
    from nba_predictor.tracking import store

    games, boxes = [], {}
    for i in range(3):
        gid = f"a{i}"
        games.append(_game(gid, f"2026-04-1{i}", "DET", "MIA", True))
        boxes[gid] = [_box("star", "Star", "DET", 25.0)]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for stat in ingest.PLAYER_STAT_TARGET_COLUMNS:
        m = DummyRegressor(strategy="constant", constant=20.0)
        m.fit([[0.0] * len(PROP_FEATURE_COLUMNS)], [20.0])
        joblib.dump(m, models_dir / f"player_{stat}_model.pkl")

    db_path = tmp_path / "t.db"
    store.init_db(db_path)
    assert ingest.score_upcoming_player_props(games, boxes, models_dir, db_path, "v-test") > 0
    points = [r["predicted_value"] for r in store.get_player_predictions_for_game(db_path, "next") if r["stat"] == "points"]
    assert points == [25.0]


# ---------------------------------------------------------------------------
# Current rosters (ESPN) are the authority; Out players get no projection
# ---------------------------------------------------------------------------

def _roster(*players):
    return [{"player_id": pid, "player_name": name, "position": "G", "injury_status": status} for pid, name, status in players]


def _hist():
    """Brown played every game for BOS last season; BOS's last box score still has him."""
    games = [
        _game("g_old1", "2026-04-08", "BOS", "NYK", True),
        _game("g_old2", "2026-04-10", "BOS", "NYK", True),
        _game("det_bos", "2026-10-21", "DET", "BOS", False),
        _game("nyk_phi", "2026-10-21", "NYK", "PHI", False),
    ]
    boxes = {
        "g_old1": [_box("brown", "Jaylen Brown", "BOS"), _box("tatum", "Jayson Tatum", "BOS")],
        "g_old2": [_box("brown", "Jaylen Brown", "BOS"), _box("tatum", "Jayson Tatum", "BOS")],
    }
    return games, boxes


def test_the_current_roster_beats_where_the_box_score_last_saw_a_player():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = _hist()
    rosters = {"BOS": _roster(("tatum", "Jayson Tatum", "")), "PHI": _roster(("brown", "Jaylen Brown", "")), "DET": [], "NYK": []}
    df = _upcoming(to_player_scoring_frame(games, boxes, rosters=rosters))
    assert {(r.player_id, r.game_id) for r in df.itertuples()} == {("tatum", "det_bos"), ("brown", "nyk_phi")}


def test_a_player_reported_out_gets_no_projection_but_a_questionable_one_does():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = _hist()
    rosters = {"BOS": _roster(("tatum", "Jayson Tatum", "Out"), ("brown", "Jaylen Brown", "Day-To-Day")), "PHI": [], "DET": [], "NYK": []}
    df = _upcoming(to_player_scoring_frame(games, boxes, rosters=rosters))
    assert set(df.player_id) == {"brown"}


def test_a_player_not_on_any_roster_is_not_projected_even_with_box_score_history():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = _hist()
    df = _upcoming(to_player_scoring_frame(games, boxes, rosters={"BOS": [], "PHI": [], "DET": [], "NYK": []}))
    assert len(df) == 0


def test_no_roster_feed_falls_back_to_the_box_score_rule():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = _hist()
    df = _upcoming(to_player_scoring_frame(games, boxes, rosters=None))
    assert set(df.player_id) == {"brown", "tatum"}


def test_earlier_seasons_feed_the_rolling_history_without_creating_projections():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = _hist()
    older = pd.DataFrame([{"player_id": "brown", "player_name": "Jaylen Brown", "team": "BOS", "game_id": "h1", "game_date": "2025-12-01",
                           "points": 25.0, "rebounds": 6.0, "assists": 3.0, "fg3m": 2.0, "minutes": 34.0, "position": "G"}])
    df = to_player_scoring_frame(games, boxes, rosters={"BOS": [], "PHI": _roster(("brown", "Jaylen Brown", "")), "DET": [], "NYK": []}, extra_completed=older)
    assert "h1" in set(df.game_id)
    assert set(_upcoming(df).player_id) == {"brown"}


def test_a_few_clamps_still_publish_the_guarded_values(tmp_path, monkeypatch):
    """One bad projection among many is clamped and published; only a collapsed model is refused."""
    import joblib
    from sklearn.dummy import DummyRegressor

    from nba_predictor.pipeline import ingest
    from nba_predictor.tracking import store

    games, boxes = [], {}
    for i in range(3):
        gid = f"c{i}"
        games.append(_game(gid, f"2026-10-1{i}", "DET", "MIA", True))
        boxes[gid] = [_box(f"p{k}", f"P{k}", "DET", 20.0) for k in range(12)]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))
    models_dir = tmp_path / "models"; models_dir.mkdir()
    for stat in ingest.PLAYER_STAT_TARGET_COLUMNS:
        sane = DummyRegressor(strategy="constant", constant=20.0 if stat == "points" else 5.0)
        sane.fit([[0.0] * len(PROP_FEATURE_COLUMNS)], [1.0])
        joblib.dump(sane, models_dir / f"player_{stat}_model.pkl")
    # twelve players x four stats = 48 projections; make exactly one of them need a clamp by giving one player a tiny average
    boxes["c2"][0] = _box("p0", "P0", "DET", 2.0)
    boxes["c1"][0] = _box("p0", "P0", "DET", 2.0)
    boxes["c0"][0] = _box("p0", "P0", "DET", 2.0)
    db_path = tmp_path / "t.db"; store.init_db(db_path)
    stored = ingest.score_upcoming_player_props(games, boxes, models_dir, db_path, "v-test")
    assert stored == 48
    by = {(r["player_id"], r["stat"]): r["predicted_value"] for r in store.get_player_predictions_for_game(db_path, "next")}
    assert by[("p0", "points")] == 2.5 * 2.0                      # 20 against a 2.0 average: clamped to the band's top
    assert by[("p1", "points")] == 20.0


# ---------------------------------------------------------------------------
# Preseason minutes are not regular-season minutes
# ---------------------------------------------------------------------------

def test_preseason_games_do_not_drag_a_stars_rolling_average_down():
    from nba_predictor.features.player_stats import build_player_feature_frame
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games, boxes = [], {}
    for i, (day, stype, pts) in enumerate([("2026-04-01", 2, 30.0), ("2026-04-03", 2, 30.0), ("2026-04-05", 2, 30.0),
                                           ("2026-10-02", 1, 4.0), ("2026-10-05", 1, 4.0), ("2026-10-08", 1, 4.0)]):
        gid = f"c{i}"
        games.append({**_game(gid, day, "DET", "MIA", True), "season_type": stype})
        boxes[gid] = [_box("star", "Star", "DET", pts)]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))

    scoring = to_player_scoring_frame(games, boxes, rosters={"DET": _roster(("star", "Star", "")), "CHI": [], "MIA": []})
    frame, _ = build_player_feature_frame(scoring)
    upcoming = frame[frame["points"].isna()]
    assert float(upcoming.iloc[0]["points_roll"]) == 30.0


def test_a_model_fitted_on_the_old_feature_set_is_skipped_not_crashed(tmp_path):
    """Right after a deploy the image still holds last night's five-feature pickles; scoring must not die on them."""
    import joblib
    from sklearn.linear_model import LinearRegression

    from nba_predictor.pipeline import ingest
    from nba_predictor.tracking import store

    games, boxes = [], {}
    for i in range(3):
        gid = f"c{i}"
        games.append(_game(gid, f"2026-10-1{i}", "DET", "MIA", True))
        boxes[gid] = [_box("p0", "P0", "DET", 20.0)]
    games.append(_game("next", "2026-10-21", "DET", "CHI", False))
    models_dir = tmp_path / "models"; models_dir.mkdir()
    for stat in ingest.PLAYER_STAT_TARGET_COLUMNS:
        old = LinearRegression().fit([[0.0] * 5, [1.0] * 5], [1.0, 2.0])      # five features: the previous model generation
        joblib.dump(old, models_dir / f"player_{stat}_model.pkl")
    db_path = tmp_path / "t.db"; store.init_db(db_path)
    assert ingest.score_upcoming_player_props(games, boxes, models_dir, db_path, "v-test") == 0
