"""The props endpoint: an in-sample MAE on every row, and the gate applied.

/games/{id}/players stays a bare JSON array of PlayerPropOut -- the frontend
client already types it that way (frontend/src/api/client.ts getGamePlayers),
so the out entries get their own endpoint rather than a wrapper this backend
PR cannot add without rewriting the frontend it is not allowed to touch.

Two invariants asserted here:

  * no probability is served anywhere on the row. NBA's model is an
    XGBRegressor returning a raw point total; there is no calibrated
    probability to show, and the phase forbids inventing one. Every row is a
    projection.
  * an out player's row is ABSENT from the ranking, not flagged in place, and
    appears exactly once below it with a source and a date.
"""

import json

import pytest
from fastapi.testclient import TestClient

from nba_predictor.api.app import app
from nba_predictor.api import deps
from nba_predictor.tracking import store


SCHEDULE = [
    {
        "game_id": "g1",
        "game_date": "2026-02-16",
        "home_team": "BOS",
        "away_team": "MIA",
        "tip_off": "2026-02-16T00:30Z",
    },
    {
        "game_id": "g2",
        "game_date": "2026-11-01",
        "home_team": "BOS",
        "away_team": "MIA",
        "tip_off": "2026-11-01T00:30Z",
    },
]

PRE_TIP = "2026-02-16T00:00:00+00:00"  # before g1's tip-off
POST_TIP = "2026-09-20T08:00:00+00:00"  # after it: a backtest row
UPCOMING = "2026-11-01T00:00:00+00:00"  # before g2's tip-off


# Holds the pristine feed function while a test has it replaced, so the fixture
# can put it back. A one-slot list rather than a global scalar so no test has to
# know about another test's patch.
_PATCHED_FEED = [None]


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))

    hub_dir = tmp_path / "data" / "cache" / "hub"
    hub_dir.mkdir(parents=True)
    (hub_dir / "players.json").write_text(
        json.dumps(
            [
                {"player_id": "5105571", "player_name": "Jayson Tatum"},
                {"player_id": "3934672", "player_name": "Nikola Jokic"},
                {"player_id": "203999", "player_name": "Someone Healthy"},
            ]
        )
    )

    from nba_predictor import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    # The gate's feed is a dependency, so no test here can reach the network.
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    yield TestClient(app)
    app.dependency_overrides.clear()
    # Restore the feed function the feed-down tests replace, so a test that makes
    # the feed unreadable cannot hand the next test a permanently broken feed
    # (which surfaces as an unexplained 503 far from its cause). A direct
    # assignment is undone here, not through monkeypatch, because monkeypatch's
    # own teardown would then re-apply the broken value.
    from nba_predictor.data import espn as espn_module

    if _PATCHED_FEED[0] is not None:
        espn_module.get_injuries, _PATCHED_FEED[0] = _PATCHED_FEED[0], None


def _injuries(*rows):
    app.dependency_overrides[deps.get_injury_report] = lambda: list(rows)


def _unreadable_feed():
    """Make the real dependency run against a feed that cannot be read.

    Overriding the dependency is not how the real feed fails -- the real failure
    is inside deps.get_injury_report, which is what has to convert it into a 503
    rather than serve an empty list. The original is stashed in ``_PATCHED_FEED``
    and put back by the client fixture's teardown.
    """
    def _boom():
        raise RuntimeError("injury feed unreachable")

    import nba_predictor.data.espn as espn_module

    _PATCHED_FEED[0] = espn_module.get_injuries
    espn_module.get_injuries = _boom
    app.dependency_overrides.pop(deps.get_injury_report, None)


def _get_unoverridden(client, path):
    """GET with the injury dependency override removed, restoring it after.

    The feed-down state is the one state that cannot be produced by a stub
    override, so these helpers are careful to put the fixture's overrides back:
    a leaked override would silently convert a 503 into a 200 for the next test.
    """
    saved = app.dependency_overrides.pop(deps.get_injury_report, None)
    try:
        return client.get(path)
    finally:
        if saved is not None:
            app.dependency_overrides[deps.get_injury_report] = saved


def _seed_resolved(db_path):
    """g1, resolved before tip-off on two players.

    points: 5105571 off by 4 and 3934672 off by 2 -> mean 3.
    threes: one resolved row, off by 1.
    rebounds and assists: predicted, never resolved.
    """
    for player_id, predicted, actual in (("5105571", 20.0, 24.0), ("3934672", 10.0, 12.0)):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id=player_id, stat="points",
            predicted_value=predicted, created_at=PRE_TIP,
        )
        store.insert_player_outcome(
            db_path, game_id="g1", player_id=player_id, stat="points",
            actual_value=actual, recorded_at="2026-02-17T00:00:00+00:00",
        )
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="5105571", stat="threes",
        predicted_value=3.0, created_at=PRE_TIP,
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="threes",
        actual_value=4.0, recorded_at="2026-02-17T00:00:00+00:00",
    )


def _predict(db_path, player_id, stat, value=25.0):
    store.insert_player_prediction(
        db_path, game_id="g2", player_id=player_id, stat=stat,
        predicted_value=value, created_at=UPCOMING,
    )


def test_a_prop_row_carries_the_mae_for_its_own_stat(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    _predict(db_path, "203999", "points")

    rows = client.get("/games/g2/players").json()

    assert [(r["stat"], r["mae"]) for r in rows] == [("points", pytest.approx(3.0))]


def test_each_stat_carries_its_own_error_estimate(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    for stat in ("points", "threes", "rebounds"):
        _predict(db_path, "203999", stat)

    by_stat = {r["stat"]: r["mae"] for r in client.get("/games/g2/players").json()}

    assert by_stat == {
        "points": pytest.approx(3.0),   # two resolved rows, mean of 4 and 2
        "threes": pytest.approx(1.0),   # one resolved row
        "rebounds": None,               # predicted, never resolved
    }


def test_a_stat_with_no_resolved_rows_is_served_as_null_not_zero(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _predict(db_path, "3934672", "rebounds")

    row = client.get("/games/g2/players").json()[0]

    assert row["mae"] is None
    assert row["mae"] != 0.0


def test_the_mae_does_not_change_when_an_unresolved_prediction_is_added(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    _seed_resolved(db_path)
    _predict(db_path, "203999", "points")
    before = client.get("/games/g2/players").json()[0]["mae"]

    # A wildly wrong prediction with no actual behind it must move nothing.
    _predict(db_path, "203999", "points", value=99.0)

    assert client.get("/games/g2/players").json()[0]["mae"] == before == pytest.approx(3.0)


def test_a_backtest_row_graded_after_the_fact_reaches_the_mae_and_not_the_pre_tip_one(client, tmp_path):
    """The old rule, stated so the reversal is visible: "Judged nowhere."

    Before 2026-10-01 (predictor-hub #66) this test asserted
    `mae is None` -- a prediction written after the game was played, scored
    against that game's result, reached no error estimate at all. That was the
    rule arguing that a look-forward row must not be judged, on the grounds
    that CFB's track record did the same. CFB reversed it too (CFB #27),
    because the exclusion was not accuracy: it meant a season of projections
    could sit on the page with no error estimate at all, since every pick in it
    came from a rerun.

    Now: the row is a recorded pick, so it counts -- `mae` is |50 - 10| = 40 --
    and it is never presented as a pre-game one, so `mae_pre_tip` is None and
    `mae_n` says 1. The look-forward concern is answered by the label, not by
    the count.
    """
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="5105571", stat="points",
        predicted_value=50.0, created_at=POST_TIP,
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="points",
        actual_value=10.0, recorded_at="2026-02-17T00:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="203999", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    row = client.get("/games/g2/players").json()[0]

    assert row["mae"] == pytest.approx(40.0), "a recorded pick was dropped from the record"
    assert row["mae_n"] == 1
    # And it is not dressed up as a pre-game pick: no pre-tip subset exists.
    assert row["mae_pre_tip"] is None
    assert row["mae_n_pre_tip"] == 0


def test_the_counted_prop_is_the_earliest_recorded_one_not_the_latest_pre_tip_row(client, tmp_path):
    """The old rule was "only the LATEST pre-tip prediction is graded".

    Two pre-tip snapshots of one player prop: the first (2026-02-16) predicted
    1.0, a rerun the next day predicted 2.0, and the actual was 20.0. Under the
    old rule the rerun was graded and the MAE was |2 - 20| = 18.0 -- the model
    was measured on its second opinion, so a model change silently restated the
    record.

    Under the current rule the EARLIEST recorded pick is the counted one, so it
    is |1 - 20| = 19.0, graded once. A rule that counted both would give
    (19 + 18) / 2 = 18.5, which is the error this test exists to make
    impossible. The rerun is history: still in the table, never graded.
    """
    db_path = tmp_path / "tracking.db"
    for created_at, predicted in ((PRE_TIP, 1.0), ("2026-02-17T00:00:00+00:00", 2.0)):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id="5105571", stat="points",
            predicted_value=predicted, created_at=created_at,
        )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="5105571", stat="points",
        actual_value=20.0, recorded_at="2026-02-18T00:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="203999", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    row = client.get("/games/g2/players").json()[0]

    # |1 - 20| = 19, graded once. The old rule said 18.0; counting both says 18.5.
    assert row["mae"] == pytest.approx(19.0)
    assert row["mae_n"] == 1
    # Both snapshots were pre-tip, so the secondary figure is the same number
    # over the same one row: the two figures agree only when nothing was late.
    assert row["mae_pre_tip"] == pytest.approx(19.0)
    assert row["mae_n_pre_tip"] == 1


def test_an_out_players_row_is_absent_from_the_ranking(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "3934672"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()

    assert [r["player_id"] for r in rows] == ["3934672"]


def test_the_out_player_is_surfaced_once_with_a_source_and_a_date(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for stat in ("points", "rebounds", "assists", "threes"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id="5105571", stat=stat,
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    out = client.get("/games/g2/players/out").json()

    assert client.get("/games/g2/players").json() == []
    assert len(out) == 1  # four stat rows, one player
    assert out[0]["player_name"] == "Jayson Tatum"
    assert out[0]["team"] == "BOS"
    assert out[0]["status"] == "Out"
    assert "ESPN" in out[0]["source"]
    assert out[0]["dated"] == "2026-09-21T19:50Z"


def test_an_unresolvable_injury_removes_nobody(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "3934672"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    # Blank id: the report could not be resolved to a person, so it removes
    # nobody. A false removal is worse than a missed one.
    _injuries({"team": "BOS", "player_id": "", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()

    assert sorted(r["player_id"] for r in rows) == ["3934672", "5105571"]
    assert client.get("/games/g2/players/out").json() == []


def test_an_injury_for_someone_not_in_this_game_removes_nobody(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "9999999", "player_name": "LeBron James",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/out").json() == []


def test_a_day_to_day_player_stays_in_the_ranking(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/out").json() == []


def test_the_out_endpoint_404s_an_unknown_game(client):
    assert client.get("/games/nope/players/out").status_code == 404


def test_a_row_serves_no_probability_field(client, tmp_path):
    # There is no calibrated probability for this model to serve, so there is
    # no field for one. predict_double_double_probability exists in
    # player_props.py and has zero callers; wiring it up is not this phase.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="3934672", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )

    row = client.get("/games/g2/players").json()[0]

    # The exact set, and it is still exactly a projection plus its error
    # estimate. `mae_pre_tip` / `mae_n` / `mae_n_pre_tip` are the two figures
    # and their counts (predictor-hub #66): `mae` is the headline over every
    # counted pick, `mae_pre_tip` the same estimate over the picks made before
    # tip-off. None of them is a probability, and none of them is a
    # recommendation.
    assert set(row) == {
        "player_id", "player_name", "stat", "predicted_value",
        "actual_value", "rebuilt", "mae", "mae_pre_tip", "mae_n", "mae_n_pre_tip",
        # The team the player plays for, so the box score can place a player on
        # a side for a game that has not been played yet (#119). It is the
        # player's own team, not a status and not a probability.
        "team",
        # The projection's 80% conformal range (the conformal ship): two
        # bounds in the stat's own unit, calibrated on the counted record.
        # A measured range around the number -- still not a probability,
        # and the token scan below still applies to them.
        "interval_low", "interval_high",
    }
    assert not any(
        token in field.lower() for field in row for token in ("prob", "pct", "percent", "chance", "odds")
    )


def test_the_out_entry_serves_no_probability_field(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    entry = client.get("/games/g2/players/out").json()[0]

    assert set(entry) == {"player_id", "player_name", "team", "status", "source", "dated"}


def test_the_props_endpoint_refuses_to_serve_ranked_rows_when_the_feed_fails(tmp_path, monkeypatch):
    # Decision 6: rows do not ship without the gate. A feed we cannot read is
    # not a gate that came back empty, and the two must not look alike.
    from nba_predictor.api import deps as deps_module

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))
    _predict(db_path, "203999", "points")

    def _boom():
        raise RuntimeError("injury feed unreachable")

    # Overriding the dependency is not how the real feed fails -- the real
    # failure is inside deps.get_injury_report, which is what must convert it
    # into a 503 rather than a 500 or a silently ungated ranking.
    monkeypatch.setattr("nba_predictor.data.espn.get_injuries", _boom)
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    # Drop any override from the fixture so the real dependency runs.
    app.dependency_overrides.pop(deps.get_injury_report, None)
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/games/g2/players")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "availability" in response.json()["detail"].lower()


def test_an_empty_injury_report_is_served_normally_not_refused(tmp_path, monkeypatch):
    # An empty report and an unreadable feed are different facts: the first
    # means nobody is out, the second means nobody checked.
    monkeypatch.setattr("nba_predictor.data.espn.get_injuries", lambda: [])
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))
    _predict(db_path, "203999", "points")

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides.pop(deps.get_injury_report, None)
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/games/g2/players")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(response.json()) == 1


# ---------------------------------------------------------------------------
# DOUBTFUL on the wire.
#
# Placement, decided from the code rather than taste: beside the out entries,
# on their OWN feed, and not on the prop row.
#
#   * Not on the row. /games/{id}/players is a bare PlayerProp[] the frontend
#     already types (client.ts getGamePlayers). Widening the row would make the
#     ranking carry feed data with a 15-minute TTL and an external attribution,
#     and one player has up to four stat rows -- the same status repeated four
#     times on a payload that is deliberately a projection and nothing else.
#   * Not on the OUT feed either, and this is the sharp one: TopCalls.tsx
#     (NBA#22) builds `outIds` from every row of the out feed and filters the
#     ranking by it. Putting doubtful players on that feed would make the
#     shipped frontend delete all 52 of them -- reintroducing this very defect
#     from a backend change no frontend PR had asked for. The out feed is a
#     removal instruction; a doubtful player must never be one.
#
# So: a third feed, /games/{id}/players/doubtful. The frontend attaches the
# note by joining on player_id, which is exact for the same reason the gate is.
# ---------------------------------------------------------------------------


def test_a_day_to_day_player_stays_in_the_ranking_and_carries_the_status(client, tmp_path):
    # The headline, asserted from both ends at once: still ranked, and named on
    # the doubtful feed with its source and date. Neither half is optional --
    # ranking without the status is the bug, and the status without the ranking
    # would be removal wearing a different hat.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()
    doubtful = client.get("/games/g2/players/doubtful").json()

    assert [r["player_id"] for r in rows] == ["5105571"]
    assert rows[0]["predicted_value"] == 25.0  # the projection is untouched
    assert len(doubtful) == 1
    assert doubtful[0]["player_id"] == "5105571"
    assert doubtful[0]["status"] == "Day-To-Day"
    assert "ESPN" in doubtful[0]["source"]
    assert doubtful[0]["dated"] == "2026-09-21T19:50Z"
    # And it is emphatically not on the out feed, which is a removal order.
    assert client.get("/games/g2/players/out").json() == []


def test_a_day_to_day_player_with_four_stat_rows_is_ranked_and_named_once(client, tmp_path):
    # Per player, not per row: the feed is keyed by athlete id, so four stat
    # rows yield one note. Four copies of one player's status is noise.
    db_path = tmp_path / "tracking.db"
    for stat in ("points", "rebounds", "assists", "threes"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id="5105571", stat=stat,
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    rows = client.get("/games/g2/players").json()
    doubtful = client.get("/games/g2/players/doubtful").json()

    assert len(rows) == 4  # every stat row still ranked
    assert len(doubtful) == 1  # one player, one note


def test_a_day_to_day_flag_does_not_reorder_or_devalue_the_ranking(client, tmp_path):
    # "may stay ranked, lower" -- but no demotion is implementable here, and
    # inventing one would be a coefficient ESPN does not publish. The ranking
    # this player gets is byte-identical to an unflagged player's.
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "203999"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    rows = {r["player_id"]: r for r in client.get("/games/g2/players").json()}

    assert set(rows) == {"5105571", "203999"}
    # Identical on every model-relevant field. player_id/player_name differ
    # because these are two different people; the flag changes nothing a reader
    # would weigh as a projection.
    def _projection_fields(row):
        return {k: v for k, v in row.items() if k not in ("player_id", "player_name")}

    assert _projection_fields(rows["5105571"]) == _projection_fields(rows["203999"])


def test_a_plain_out_player_is_on_neither_the_ranking_nor_the_doubtful_feed(client, tmp_path):
    # Removal is reserved for Out, and an out player is not also "doubtful":
    # one player, one feed, or the UI would show them both removed and flagged.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})

    assert client.get("/games/g2/players").json() == []
    assert client.get("/games/g2/players/out").json()[0]["player_id"] == "5105571"
    assert client.get("/games/g2/players/doubtful").json() == []


def test_the_three_availability_stays_stay_distinguishable(client, tmp_path):
    """out / report-listed-nobody / route-unreadable, as three different facts.

    The frontend already words these apart and must not have to guess which it
    is holding. Each is asserted on its own, and the third is a refusal rather
    than an empty list -- an empty list on an unreadable feed is the
    "checked and clear" lie this whole change exists to stop.
    """
    db_path = tmp_path / "tracking.db"
    for player_id in ("5105571", "203999"):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=player_id, stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )

    # 1. the report names somebody out: removed, and named on the out feed
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Out", "dated": "2026-09-21T19:50Z"})
    assert client.get("/games/g2/players/out").json() != []
    assert client.get("/games/g2/players/doubtful").json() == []

    # 2. the report was read and names nobody: empty, but served as empty
    _injuries()
    assert client.get("/games/g2/players/out").status_code == 200
    assert client.get("/games/g2/players/out").json() == []
    assert client.get("/games/g2/players/doubtful").json() == []
    assert len(client.get("/games/g2/players").json()) == 2  # ranked, nothing flagged

    # 3. the feed could not be read at all: refused, never an empty list
    _unreadable_feed()
    for path in ("/games/g2/players", "/games/g2/players/out", "/games/g2/players/doubtful"):
        response = _get_unoverridden(client, path)
        assert response.status_code == 503, path
        assert "availability" in response.json()["detail"].lower()


def test_the_props_row_carries_no_availability_field(client, tmp_path):
    # The row stays a bare projection: adding the status here is what the
    # placement decision refuses, so it is asserted rather than left to review.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    row = client.get("/games/g2/players").json()[0]

    assert set(row) == {
        "player_id", "player_name", "stat", "predicted_value",
        "actual_value", "rebuilt", "mae", "mae_pre_tip", "mae_n", "mae_n_pre_tip",
        # The team the player plays for, so the box score can place a player on
        # a side for a game that has not been played yet (#119). It is the
        # player's own team, not a status and not a probability.
        "team",
        # The projection's 80% conformal range (the conformal ship): the
        # model's measured spread for the line, in the stat's own unit.
        # Not an availability signal of any kind.
        "interval_low", "interval_high",
    }
    assert not any(
        token in field.lower() for field in row
        for token in ("doubt", "status", "injur", "avail", "day", "prob")
    )


def test_the_doubtful_entry_serves_only_provenance_and_no_verdict(client, tmp_path):
    # Exactly the out entry's six fields. No probability, no coefficient, no
    # recommendation field: the note states availability and nothing else.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "5105571", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    entry = client.get("/games/g2/players/doubtful").json()[0]

    assert set(entry) == {"player_id", "player_name", "team", "status", "source", "dated"}
    assert not any(
        token in field.lower() for field in entry
        for token in ("prob", "pct", "percent", "chance", "odds", "coef", "avoid", "fade", "bet")
    )


def test_the_doubtful_endpoint_404s_an_unknown_game(client):
    assert client.get("/games/nope/players/doubtful").status_code == 404


def test_an_unresolvable_doubtful_injury_flags_nobody(client, tmp_path):
    # The removal asymmetry applies to flagging too: an entry that cannot be
    # resolved to a person by exact id resolves to nobody, rather than falling
    # back to the name and flagging the wrong player.
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "BOS", "player_id": "", "player_name": "Jayson Tatum",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/doubtful").json() == []


def test_a_doubtful_injury_for_someone_not_in_this_game_flags_nobody(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.insert_player_prediction(
        db_path, game_id="g2", player_id="5105571", stat="points",
        predicted_value=25.0, created_at=UPCOMING,
    )
    _injuries({"team": "LAL", "player_id": "9999999", "player_name": "LeBron James",
               "status": "Day-To-Day", "dated": "2026-09-21T19:50Z"})

    assert len(client.get("/games/g2/players").json()) == 1
    assert client.get("/games/g2/players/doubtful").json() == []


def test_the_live_shaped_report_renders_52_doubtful_notes_over_52_ranked_rows(client, tmp_path):
    """The count this change is for, from the fixture and not from a hand-pick.

    Shaped like the live feed measured 2026-10-01 -- 65 entries, 52 Day-To-Day
    and 13 Out -- with every one of the 65 ranked for this game. The endpoint
    reports what the feed actually contains: 13 removed and named on the out
    feed, 52 still ranked AND named on the doubtful feed. Before this change
    those 52 appeared on no feed at all.
    """
    db_path = tmp_path / "tracking.db"
    for i in range(65):
        store.insert_player_prediction(
            db_path, game_id="g2", player_id=str(1000 + i), stat="points",
            predicted_value=25.0, created_at=UPCOMING,
        )
    _injuries(*(
        {"team": "BOS", "player_id": str(1000 + i), "player_name": f"Player {i}",
         "status": "Day-To-Day" if i < 52 else "Out", "dated": "2026-09-21T19:50Z"}
        for i in range(65)
    ))

    rows = client.get("/games/g2/players").json()
    out = client.get("/games/g2/players/out").json()
    doubtful = client.get("/games/g2/players/doubtful").json()

    assert len(rows) == 52           # the 13 out players are gone
    assert len(out) == 13
    assert len(doubtful) == 52       # and the 52 that stayed are all named
    assert {r["player_id"] for r in rows} == {d["player_id"] for d in doubtful}
    assert not {r["player_id"] for r in rows} & {o["player_id"] for o in out}
    assert {d["status"] for d in doubtful} == {"Day-To-Day"}


# --- names: a player outside the hub file still gets a name -----------------

def _box_row(pid, name):
    # shaped like data/espn.py::get_player_boxscore rows
    return {"player_id": pid, "player_name": name, "team": "BOS", "position": "G", "minutes": 20.0,
            "points": 5.0, "rebounds": 1.0, "assists": 1.0}


def test_a_player_missing_from_the_hub_is_named_from_the_persisted_map(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.record_player_names(db_path, [_box_row("3137259", "Some Bench Guy")])
    for pid in ("3137259", "9999999"):
        _predict(db_path, pid, "points")

    names = {r["player_id"]: r["player_name"] for r in client.get("/games/g2/players").json()}

    assert names["3137259"] == "Some Bench Guy"  # not in hub players.json
    assert names["9999999"] == "9999999"  # unknown everywhere: the id, last resort


def test_the_hub_name_wins_over_the_persisted_one(client, tmp_path):
    db_path = tmp_path / "tracking.db"
    store.record_player_names(db_path, [_box_row("5105571", "J. Tatum")])
    _predict(db_path, "5105571", "points")

    assert client.get("/games/g2/players").json()[0]["player_name"] == "Jayson Tatum"


def test_an_empty_display_name_never_overwrites_a_known_name(tmp_path):
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.record_player_names(db_path, [_box_row("1", "Known Name")])
    store.record_player_names(db_path, [_box_row("1", ""), _box_row("1", "  "), _box_row("2", "")])

    assert store.load_player_names(db_path) == {"1": "Known Name"}


def test_the_refresher_rebuilds_names_on_a_fresh_disk(tmp_path):
    from nba_predictor.pipeline.player_props_refresher import rebuild_player_names

    db_path = tmp_path / "fresh.db"
    store.init_db(db_path)
    sched = [{"game_id": "x", "game_date": "2026-10-01", "completed": True}]
    n = rebuild_player_names(sched, db_path, fetch=lambda gs: {"x": [_box_row("42", "Rebuilt Name")]},
                             now=__import__("pandas").Timestamp("2026-10-10T00:00Z"))

    assert n == 1
    assert store.load_player_names(db_path) == {"42": "Rebuilt Name"}


def test_backfill_reads_cached_espn_box_files(tmp_path):
    from nba_predictor.pipeline.ingest import backfill_names_from_espn_cache

    cache = tmp_path / "espn"
    cache.mkdir()
    (cache / "player_boxscore_1.json").write_text(json.dumps({"rows": [_box_row("7", "Cached Guy"), _box_row("8", "")]}))
    db_path = tmp_path / "bf.db"

    backfill_names_from_espn_cache(db_path, cache)

    assert store.load_player_names(db_path) == {"7": "Cached Guy"}
