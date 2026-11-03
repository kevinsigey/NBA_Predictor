"""Player-prop grading: the delta, the grade bands, the section ranking.

The grading judges the COUNTED pick (the earliest recorded projection per
player and stat -- the pick the track record counts) against the recorded
actual, in units of the props model's own holdout MAE for the stat. These
tests pin the arithmetic and the honesty rules: no band means no grade, a
baseline comes only from games recorded BEFORE the graded one, and a pick
that cannot prove it was made before tip-off is graded and labelled, never
dropped and never passed off as pre-tip.
"""

import json
from datetime import datetime, timedelta, timezone

BANDS = {"points": 4.0, "rebounds": 2.0, "assists": 1.0, "threes": 0.5}


def _fresh_stamp() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _game(game_id, day, home="BOS", away="MIA", tip="19:00"):
    return {
        "game_id": game_id,
        "game_date": day,
        "home_team": home,
        "away_team": away,
        "tip_off": f"{day}T{tip}:00+00:00",
        "completed": True,
    }


def _seed(db_path, schedule, picks, outcomes, names=()):
    from nba_predictor.tracking import store

    store.init_db(db_path)
    for pick in picks:
        store.insert_player_prediction(db_path, **pick)
    for outcome in outcomes:
        store.insert_player_outcome(db_path, **outcome)
    if names:
        store.record_player_names(db_path, [{"player_id": pid, "player_name": name} for pid, name in names])


def _pick(game_id, player_id, stat, value, created_at, position=None):
    return {
        "game_id": game_id, "player_id": player_id, "stat": stat,
        "predicted_value": value, "created_at": created_at, "position": position,
    }


def _outcome(game_id, player_id, stat, value, recorded_at="2026-11-02T03:00:00+00:00"):
    return {
        "game_id": game_id, "player_id": player_id, "stat": stat,
        "actual_value": value, "recorded_at": recorded_at,
    }


# ---------------------------------------------------------------------------
# The grade bands
# ---------------------------------------------------------------------------


def test_grade_bands_are_multiples_of_the_stats_own_mae():
    from nba_predictor.tracking.prop_grading import grade_for

    # MAE 4.0: half is 2.0, the error itself 4.0, twice it 8.0.
    assert grade_for(0.0, 4.0) == "On the number"
    assert grade_for(2.0, 4.0) == "On the number"      # exactly half: inside the band
    assert grade_for(2.1, 4.0) == "Close"
    assert grade_for(4.0, 4.0) == "Close"              # exactly the MAE: inside the band
    assert grade_for(4.1, 4.0) == "Off"
    assert grade_for(8.0, 4.0) == "Off"                # exactly twice: inside the band
    assert grade_for(8.1, 4.0) == "Way off"


def test_grade_is_none_without_a_measured_error():
    from nba_predictor.tracking.prop_grading import grade_for

    assert grade_for(3.0, None) is None
    assert grade_for(3.0, 0.0) is None


def test_prop_bands_reads_the_manifests_published_mae(tmp_path):
    from nba_predictor.tracking.prop_grading import prop_bands

    (tmp_path / "player_props_manifest.json").write_text(json.dumps({
        "metrics": {"points": {"mae": 4.557}, "rebounds": {"mae": None}, "assists": {"mae": 1.326}},
    }))
    assert prop_bands(tmp_path) == {"points": 4.557, "assists": 1.326}
    assert prop_bands(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# The graded day
# ---------------------------------------------------------------------------


def _two_day_world(tmp_path):
    """Day 1 (2026-10-30) exists only as baseline history; day 2 (2026-11-01)
    is the graded day. Player A averages 20 points over two prior recorded
    games; the model called 30 and he scored 31."""
    db_path = tmp_path / "tracking.db"
    schedule = [
        _game("g0", "2026-10-28"),
        _game("g1", "2026-10-30"),
        _game("g2", "2026-11-01"),
    ]
    picks = [
        _pick("g2", "A", "points", 30.0, "2026-11-01T12:00:00+00:00"),
        _pick("g2", "B", "points", 10.0, "2026-11-01T12:00:00+00:00"),
    ]
    outcomes = [
        _outcome("g0", "A", "points", 18.0),
        _outcome("g1", "A", "points", 22.0),
        _outcome("g2", "A", "points", 31.0),
        _outcome("g2", "B", "points", 4.0),
    ]
    _seed(db_path, schedule, picks, outcomes, names=(("A", "Player A"), ("B", "Player B")))
    return db_path, schedule


def test_latest_day_is_the_most_recent_day_with_graded_rows(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    payload = build_prop_grading(db_path, schedule, BANDS)

    assert payload["latest_day"]["date"] == "2026-11-01"
    assert payload["latest_day"]["n_graded"] == 2
    games = payload["latest_day"]["games"]
    assert [g["game_id"] for g in games] == ["g2"]
    row_a = next(r for r in games[0]["rows"] if r["player_id"] == "A")
    # delta = actual - projected: beating the projection is positive.
    assert row_a["predicted"] == 30.0
    assert row_a["actual"] == 31.0
    assert row_a["delta"] == 1.0
    # |1.0| <= 0.5 x 4.0 MAE -> on the number.
    assert row_a["grade"] == "On the number"
    assert row_a["player_name"] == "Player A"
    assert row_a["made_before_tip"] is True


def test_biggest_surprises_rank_by_absolute_delta_both_directions(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    payload = build_prop_grading(db_path, schedule, BANDS)

    surprises = payload["biggest_surprises"]
    # B fell 6 short (|delta| 6) outranks A's +1, and the sign is preserved.
    assert [r["player_id"] for r in surprises] == ["B", "A"]
    assert surprises[0]["delta"] == -6.0
    assert surprises[0]["grade"] == "Off"  # 6 > 1x MAE (4), <= 2x MAE (8)


def test_most_confident_ranks_by_departure_from_the_rolling_average(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    payload = build_prop_grading(db_path, schedule, BANDS)

    # A's baseline is his two prior recorded games: (18 + 22) / 2 = 20. The
    # graded game's own 31 never enters it.
    confident = payload["most_confident"]
    assert [r["player_id"] for r in confident] == ["A"]
    assert confident[0]["baseline"] == 20.0
    assert confident[0]["departure"] == 10.0
    # B has no recorded prior game: graded, but unrankable here -- and the
    # payload says how many rows the ranking could cover.
    assert payload["n_with_baseline"] == 1
    day_rows = payload["latest_day"]["games"][0]["rows"]
    row_b = next(r for r in day_rows if r["player_id"] == "B")
    assert row_b["baseline"] is None and row_b["departure"] is None


def test_baseline_uses_only_the_previous_ten_games(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game(f"h{i}", f"2026-10-{10 + i:02d}") for i in range(12)] + [_game("gx", "2026-11-01")]
    outcomes = []
    # Two early 0-point games fall out of the 10-game window; the last ten are all 20s.
    values = [0.0, 0.0] + [20.0] * 10
    for game, value in zip(schedule[:12], values):
        outcomes.append(_outcome(game["game_id"], "A", "points", value))
    outcomes.append(_outcome("gx", "A", "points", 25.0))
    picks = [_pick("gx", "A", "points", 24.0, "2026-11-01T12:00:00+00:00")]
    _seed(db_path, schedule, picks, outcomes)

    payload = build_prop_grading(db_path, schedule, BANDS)
    row = payload["most_confident"][0]
    assert row["baseline"] == 20.0


def test_a_pick_that_cannot_prove_pre_tip_is_graded_and_labelled(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game("g2", "2026-11-01")]  # tips 19:00Z
    picks = [
        _pick("g2", "A", "points", 20.0, "2026-11-01T12:00:00+00:00"),   # before tip
        _pick("g2", "B", "points", 20.0, "2026-11-01T23:30:00+00:00"),   # after tip
    ]
    outcomes = [_outcome("g2", "A", "points", 21.0), _outcome("g2", "B", "points", 21.0)]
    _seed(db_path, schedule, picks, outcomes)

    payload = build_prop_grading(db_path, schedule, BANDS)
    assert payload["latest_day"]["n_graded"] == 2
    assert payload["latest_day"]["n_pre_tip"] == 1
    rows = {r["player_id"]: r for r in payload["latest_day"]["games"][0]["rows"]}
    assert rows["A"]["made_before_tip"] is True
    assert rows["B"]["made_before_tip"] is False


def test_the_counted_pick_is_the_earliest_recorded(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game("g2", "2026-11-01")]
    picks = [
        _pick("g2", "A", "points", 20.0, "2026-11-01T12:00:00+00:00"),
        _pick("g2", "A", "points", 99.0, "2026-11-01T13:00:00+00:00"),  # a rerun: history, not the pick
    ]
    outcomes = [_outcome("g2", "A", "points", 22.0)]
    _seed(db_path, schedule, picks, outcomes)

    payload = build_prop_grading(db_path, schedule, BANDS)
    rows = payload["latest_day"]["games"][0]["rows"]
    assert len(rows) == 1
    assert rows[0]["predicted"] == 20.0


def test_no_bands_means_rows_are_served_ungraded(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    payload = build_prop_grading(db_path, schedule, None)
    assert payload["bands"] is None
    assert all(r["grade"] is None for r in payload["biggest_surprises"])


def test_a_store_without_props_tables_grades_nothing(tmp_path):
    """A DB whose props tables were never created holds no recorded props:
    the grading is empty and building it must not raise. The public snapshot
    builds against such a path whenever init_db did not run (the staleness
    tests pin exactly that), and it must still publish."""
    from nba_predictor.tracking.prop_grading import build_prop_grading

    payload = build_prop_grading(tmp_path / "never-initialised.db", [], BANDS)
    assert payload["latest_day"] is None
    assert payload["most_confident"] == []
    assert payload["biggest_surprises"] == []
    assert payload["n_with_baseline"] == 0


def test_empty_store_grades_nothing(tmp_path):
    from nba_predictor.tracking import store
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    payload = build_prop_grading(db_path, [], BANDS)
    assert payload["latest_day"] is None
    assert payload["most_confident"] == []
    assert payload["biggest_surprises"] == []


# ---------------------------------------------------------------------------
# The aggregates: the model's tracking over the whole counted record
# ---------------------------------------------------------------------------


def test_aggregates_cover_the_counted_record_with_mae_bias_and_band_shares(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    aggregates = build_prop_grading(db_path, schedule, BANDS)["aggregates"]

    # Both graded rows are points, both pre-tip: A +1.0, B -6.0.
    (points,) = aggregates["per_stat"]
    assert points["stat"] == "points"
    assert points["n"] == 2
    assert points["mae"] == 3.5                       # (1 + 6) / 2
    assert points["bias"] == -2.5                     # (1 - 6) / 2: projections ran high
    assert points["holdout_mae"] == 4.0               # the manifest band, as the reference
    assert points["band_n"] == 2
    shares = {s["grade"]: s for s in points["bands"]}
    assert shares["On the number"] == {"grade": "On the number", "n": 1, "pct": 50.0}
    assert shares["Off"] == {"grade": "Off", "n": 1, "pct": 50.0}
    assert shares["Close"]["pct"] == 0.0 and shares["Way off"]["pct"] == 0.0
    # The window is stated, and the daily trend carries the same MAE.
    assert aggregates["from_date"] == aggregates["to_date"] == "2026-11-01"
    assert aggregates["n_days"] == 1 and aggregates["total"] == 2
    assert aggregates["daily"] == [{"date": "2026-11-01", "stats": {"points": {"n": 2, "mae": 3.5}}}]


def test_aggregates_count_only_rows_that_prove_they_were_made_before_tip(tmp_path):
    """The day payload grades the post-tip row and labels it; the model's
    tracking may not count it -- an aggregate over answers seen in advance
    would grade the model on a test it had already read."""
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game("g2", "2026-11-01")]
    picks = [
        _pick("g2", "A", "points", 20.0, "2026-11-01T12:00:00+00:00"),   # before tip
        _pick("g2", "B", "points", 20.0, "2026-11-01T23:30:00+00:00"),   # after tip
    ]
    outcomes = [_outcome("g2", "A", "points", 21.0), _outcome("g2", "B", "points", 29.0)]
    _seed(db_path, schedule, picks, outcomes)

    aggregates = build_prop_grading(db_path, schedule, BANDS)["aggregates"]
    (points,) = aggregates["per_stat"]
    assert aggregates["total"] == 1
    assert points["n"] == 1 and points["mae"] == 1.0  # only A's +1.0; B's +9.0 is not the model's record


def test_aggregates_without_a_band_for_a_stat_carry_no_shares(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    aggregates = build_prop_grading(db_path, schedule, {"rebounds": 2.0})["aggregates"]
    (points,) = aggregates["per_stat"]
    assert points["holdout_mae"] is None
    assert points["band_n"] == 0 and points["bands"] == []
    assert points["mae"] == 3.5  # the miss is still measured; only the judging is withheld


def test_aggregates_daily_trend_spans_every_graded_day(tmp_path):
    from nba_predictor.tracking.prop_grading import build_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game("g1", "2026-10-30"), _game("g2", "2026-11-01")]
    picks = [
        _pick("g1", "A", "points", 20.0, "2026-10-30T12:00:00+00:00"),
        _pick("g2", "A", "points", 20.0, "2026-11-01T12:00:00+00:00"),
    ]
    outcomes = [_outcome("g1", "A", "points", 22.0), _outcome("g2", "A", "points", 26.0)]
    _seed(db_path, schedule, picks, outcomes)

    aggregates = build_prop_grading(db_path, schedule, BANDS)["aggregates"]
    assert aggregates["from_date"] == "2026-10-30" and aggregates["to_date"] == "2026-11-01"
    assert aggregates["n_days"] == 2
    assert [d["date"] for d in aggregates["daily"]] == ["2026-10-30", "2026-11-01"]
    assert aggregates["daily"][0]["stats"]["points"]["mae"] == 2.0
    assert aggregates["daily"][1]["stats"]["points"]["mae"] == 6.0


# ---------------------------------------------------------------------------
# One game's grading: the post-game review's payload
# ---------------------------------------------------------------------------


def test_game_grading_returns_the_games_complete_rows(tmp_path):
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    grading = build_game_prop_grading(db_path, schedule, BANDS, "g2")

    assert grading["game_date"] == "2026-11-01"
    assert grading["n_graded"] == 2 and grading["n_pre_tip"] == 2
    assert [r["player_id"] for r in grading["rows"]] == ["A", "B"]  # by player, as the day payload orders them
    # The game's biggest surprise is B's -6.0, ahead of A's +1.0.
    assert [r["player_id"] for r in grading["biggest_surprises"]] == ["B", "A"]
    assert grading["bands"] == BANDS


def test_game_grading_labels_the_post_tip_row_and_keeps_it(tmp_path):
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path = tmp_path / "tracking.db"
    schedule = [_game("g2", "2026-11-01")]
    picks = [
        _pick("g2", "A", "points", 20.0, "2026-11-01T12:00:00+00:00"),
        _pick("g2", "B", "points", 20.0, "2026-11-01T23:30:00+00:00"),
    ]
    outcomes = [_outcome("g2", "A", "points", 21.0), _outcome("g2", "B", "points", 21.0)]
    _seed(db_path, schedule, picks, outcomes)

    grading = build_game_prop_grading(db_path, schedule, BANDS, "g2")
    assert grading["n_graded"] == 2 and grading["n_pre_tip"] == 1
    rows = {r["player_id"]: r for r in grading["rows"]}
    assert rows["B"]["made_before_tip"] is False and rows["B"]["grade"] is not None


def test_game_grading_is_none_when_the_game_has_no_counted_rows(tmp_path):
    from nba_predictor.tracking.prop_grading import build_game_prop_grading

    db_path, schedule = _two_day_world(tmp_path)
    # g1 was played and has recorded outcomes, but nobody's projection was
    # counted for it; an unknown id is the same absence.
    assert build_game_prop_grading(db_path, schedule, BANDS, "g1") is None
    assert build_game_prop_grading(db_path, schedule, BANDS, "nope") is None


def test_game_grading_route_serves_the_game_and_404s_the_absence(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor import config
    from nba_predictor.api import deps
    from nba_predictor.api.app import app
    from nba_predictor.tracking import store

    schedule = [_game("g2", "2026-11-01")]
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="A", stat="points",
        predicted_value=20.0, created_at="2026-11-01T12:00:00+00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g2", player_id="A", stat="points",
        actual_value=26.0, recorded_at="2026-11-02T03:00:00+00:00",
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sched_dir = tmp_path / "cache" / "schedule"
    sched_dir.mkdir(parents=True)
    (sched_dir / "games.json").write_text(json.dumps(schedule))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "player_props_manifest.json").write_text(json.dumps({"metrics": {"points": {"mae": 4.0}}}))
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_models_dir] = lambda: models_dir
    try:
        client = TestClient(app)
        body = client.get("/hub/prop-grading/games/g2").json()
        assert body["n_graded"] == 1 and body["rows"][0]["delta"] == 6.0
        assert body["rows"][0]["grade"] == "Off"
        assert client.get("/hub/prop-grading/games/nope").status_code == 404
        # The day payload carries the aggregates over the same population.
        day = client.get("/hub/prop-grading").json()
        assert day["aggregates"]["total"] == 1
        assert day["aggregates"]["per_stat"][0]["mae"] == 6.0
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# The snapshot carries the payload
# ---------------------------------------------------------------------------


def test_generate_snapshot_includes_prop_grading(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    schedule_path = tmp_path / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(json.dumps([_game("g2", "2026-11-01")]))
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    for name in ("teams", "players", "rankings", "standings"):
        (hub_dir / f"{name}.json").write_text("[]")
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(), "models": [], "metrics": {}}))
    (tmp_path / "models" / "player_props_manifest.json").write_text(json.dumps({"metrics": {"points": {"mae": 4.0}}}))

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="A", stat="points",
        predicted_value=20.0, created_at="2026-11-01T12:00:00+00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g2", player_id="A", stat="points",
        actual_value=26.0, recorded_at="2026-11-02T03:00:00+00:00",
    )

    snapshot = generate_snapshot(schedule_path, hub_dir, manifest_path, db_path)
    grading = snapshot["prop_grading"]
    assert grading["bands"] == {"points": 4.0}
    assert grading["latest_day"]["date"] == "2026-11-01"
    row = grading["biggest_surprises"][0]
    assert row["delta"] == 6.0 and row["grade"] == "Off"
