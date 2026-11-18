import json

import pytest


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """This module installs app-level dependency overrides and, before this
    fixture, never removed them -- they leaked into whatever test ran next. The
    players routes also depend on the availability gate, whose feed is ESPN's
    live injury report: with no override, every /games/{id}/players test here
    reached the real network and answered 503 when it was blocked. The gate's
    feed is a dependency precisely so it can be substituted (see
    api/deps.get_injury_report), so an empty report is substituted: no test in
    this repo touches the network.
    """
    from nba_predictor.api.app import app

    yield
    app.dependency_overrides.clear()


def _client_with_overrides(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]'
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    client = TestClient(app)
    return client, db_path


def test_list_games_requires_date_query_param(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games")
    assert response.status_code == 422


def test_list_games_returns_games_for_date_with_null_prediction(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games", params={"date": "2026-11-01"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "g1"
    assert body[0]["prediction"] is None


def test_list_games_includes_latest_prediction_when_present(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T12:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=3.5, predicted_total=224.5,
    )

    response = client.get("/games", params={"date": "2026-11-01"})
    body = response.json()
    assert body[0]["prediction"]["home_win_probability"] == 0.62


def test_get_game_detail_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist")
    assert response.status_code == 404


def test_get_game_detail_includes_markets(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.62,
        market_probability=0.55, edge=0.07, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1")
    assert response.status_code == 200
    body = response.json()
    assert body["home_team"] == "BOS"
    assert len(body["markets"]) == 1
    assert body["markets"][0]["market"] == "h2h"


def test_get_game_players_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist/players")
    assert response.status_code == 404


def test_get_game_players_returns_tracked_props(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["stat"] == "points"
    assert body[0]["predicted_value"] == 27.5


def test_get_game_players_uses_real_player_name_from_hub_cache(tmp_path, monkeypatch):
    import json

    from nba_predictor.tracking import store
    from nba_predictor import config

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    hub_dir = tmp_path / "data" / "cache" / "hub"
    hub_dir.mkdir(parents=True)
    (hub_dir / "players.json").write_text(json.dumps([{"player_id": "203999", "player_name": "Nikola Jokic"}]))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    assert response.json()[0]["player_name"] == "Nikola Jokic"


def test_get_game_players_falls_back_to_id_when_name_unknown(tmp_path, monkeypatch):
    from nba_predictor.tracking import store
    from nba_predictor import config

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "no-such-data-dir")

    response = client.get("/games/g1/players")
    assert response.json()[0]["player_name"] == "203999"


def _client_with_week_schedule(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        json.dumps(
            [
                {"game_id": "g1", "game_date": "2026-02-16", "home_team": "BOS", "away_team": "MIA"},
                {"game_id": "g2", "game_date": "2026-02-19", "home_team": "LAL", "away_team": "GSW"},
                {"game_id": "g3", "game_date": "2026-02-23", "home_team": "DEN", "away_team": "PHX"},
            ]
        )
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    return TestClient(app)


def test_list_games_for_week_requires_start_param(tmp_path, monkeypatch):
    client = _client_with_week_schedule(tmp_path, monkeypatch)
    response = client.get("/games/week")
    assert response.status_code == 422


def test_list_games_for_week_returns_only_games_in_window(tmp_path, monkeypatch):
    client = _client_with_week_schedule(tmp_path, monkeypatch)
    response = client.get("/games/week", params={"start": "2026-02-16"})

    assert response.status_code == 200
    body = response.json()
    assert [g["game_id"] for g in body] == ["g1", "g2"]


def test_season_first_week_uses_the_injected_today(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([
        {"game_id": "g0", "game_date": "2025-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]))
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_today] = lambda: "2026-10-25"

    client = TestClient(app)
    response = client.get("/season/first-week")

    assert response.status_code == 200
    from nba_predictor.services.schedule_repository import monday_of
    assert response.json() == {"first_week_start": monday_of("2026-10-25")}


def test_get_game_players_includes_actual_value_when_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="203999", stat="points",
        actual_value=24.0, recorded_at="2026-11-01T22:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    assert response.json()[0]["actual_value"] == 24.0


def test_get_game_players_actual_value_is_null_when_not_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.json()[0]["actual_value"] is None


def _client_with_final(tmp_path):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-03-01", "tip_off": "2026-03-02T00:30Z", "home_team": "BOS",'
        ' "away_team": "MIA", "completed": true, "home_pts": 110, "away_pts": 100}]'
    )
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []
    return TestClient(app), db_path


def test_final_game_shows_the_pre_tip_pick_not_a_later_rebuild(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T10:00:00+00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.3, predicted_margin=-2.0, predicted_total=220.0,
    )

    game = client.get("/games", params={"date": "2026-03-01"}).json()[0]

    assert game["prediction"]["home_win_probability"] == 0.7
    assert game["rebuilt"] is False
    assert game["tip_off"] == "2026-03-02T00:30Z"


def test_final_game_with_only_a_backtest_pick_is_marked_rebuilt(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.3, predicted_margin=-2.0, predicted_total=220.0,
    )

    game = client.get("/games/week", params={"start": "2026-02-23"}).json()[0]
    detail = client.get("/games/g1").json()

    assert game["rebuilt"] is True
    assert game["prediction"]["home_win_probability"] == 0.3
    assert detail["rebuilt"] is True


def test_player_props_are_one_per_player_stat_and_flag_rebuilt_rows(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p1", stat="points", predicted_value=20.0, created_at="2026-03-01T10:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p1", stat="points", predicted_value=31.0, created_at="2026-09-20T08:00:00+00:00",
    )
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p2", stat="points", predicted_value=12.0, created_at="2026-09-20T08:00:00+00:00",
    )

    props = {p["player_id"]: p for p in client.get("/games/g1/players").json()}

    assert len(props) == 2
    assert props["p1"]["predicted_value"] == 20.0 and props["p1"]["rebuilt"] is False
    assert props["p2"]["rebuilt"] is True


def test_market_rows_made_after_tip_off_are_flagged_rebuilt(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    for created_at in ("2026-03-01T10:00:00+00:00", "2026-03-02T01:00:00+00:00"):
        store.insert_market_prediction(
            db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.6,
            market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-120, created_at=created_at,
        )

    markets = client.get("/games/g1").json()["markets"]

    assert sorted(m["rebuilt"] for m in markets) == [False, True]


def test_the_out_list_names_only_the_two_teams(tmp_path, monkeypatch):
    """A third team's Out player is not this game's news.

    The report is league-wide, so a game between two teams used to list a
    player who was playing for neither of them.
    """
    from nba_predictor.api import deps
    from nba_predictor.api.app import app as fastapi_app
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    for player_id in ("p1", "p2", "p3"):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id=player_id, stat="points",
            predicted_value=20.0, created_at="2026-11-01T12:00:00",
        )
    fastapi_app.dependency_overrides[deps.get_injury_report] = lambda: [
        {"team": "BOS", "player_id": "p1", "player_name": "Home Out", "status": "Out", "dated": "2026-11-01"},
        {"team": "MIA", "player_id": "p2", "player_name": "Away Out", "status": "Out", "dated": "2026-11-01"},
        {"team": "SAS", "player_id": "p3", "player_name": "Tobias Harris", "status": "Out", "dated": "2026-11-01"},
    ]

    body = client.get("/games/g1/players/out").json()

    assert {row["team"] for row in body} == {"BOS", "MIA"}, (
        "the out list names a team that is not playing in this game"
    )
    assert len(body) == 2


def test_the_game_detail_counts_only_the_two_teams_outs(tmp_path, monkeypatch):
    """The availability sentence counts this game's players, not the league's."""
    from nba_predictor.api import deps
    from nba_predictor.api.app import app as fastapi_app
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    for player_id in ("p1", "p2", "p3"):
        store.insert_player_prediction(
            db_path, game_id="g1", player_id=player_id, stat="points",
            predicted_value=20.0, created_at="2026-11-01T12:00:00",
        )
    fastapi_app.dependency_overrides[deps.get_injury_report_best_effort] = lambda: [
        {"team": "BOS", "player_id": "p1", "player_name": "Home Out", "status": "Out", "dated": "2026-11-01"},
        {"team": "SAS", "player_id": "p3", "player_name": "Tobias Harris", "status": "Out", "dated": "2026-11-01"},
    ]

    body = client.get("/games/g1").json()

    assert body["injury_summary"] == "Checked against ESPN availability — 1 Out, 0 Day-to-Day", (
        "the availability sentence counts a player from another team"
    )


def test_the_game_detail_serves_the_props_bands(tmp_path, monkeypatch):
    """The band is the props model's own error, so it moves with each retrain."""
    from nba_predictor.api import deps
    from nba_predictor.api.app import app as fastapi_app

    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "player_props_manifest.json").write_text(
        json.dumps({"metrics": {"points": {"mae": 4.6}, "rebounds": {"mae": 1.9}}}),
    )
    fastapi_app.dependency_overrides[deps.get_models_dir] = lambda: models_dir

    body = client.get("/games/g1").json()

    assert body["prop_bands"] == {"points": 4.6, "rebounds": 1.9}


def test_the_props_bands_are_null_without_a_manifest(tmp_path, monkeypatch):
    """No manifest means no band. A made-up default would be a number nobody
    measured, and the review would judge players against it."""
    from nba_predictor.api import deps
    from nba_predictor.api.app import app as fastapi_app

    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    fastapi_app.dependency_overrides[deps.get_models_dir] = lambda: tmp_path / "no-models"

    body = client.get("/games/g1").json()

    assert body["prop_bands"] is None


def test_a_game_with_quoted_lines_gets_a_cover_and_an_over_chance(tmp_path):
    """The predictions table has no lines, so the API never priced a cover/over chance. The lines come from the odds rows."""
    from nba_predictor.api.routes import _prediction_out
    from nba_predictor.tracking import store

    db = tmp_path / "t.db"
    store.init_db(db)
    game = {"game_id": "g1", "home_team": "BOS", "away_team": "MIA", "game_date": "2099-01-01", "tip_off": "2099-01-01T23:00Z", "completed": False}
    store.insert_prediction(db, game_id="g1", created_at="2098-12-31T12:00:00+00:00", model_version="v", home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0)
    quote = dict(game_id="g1", model_probability=0.5, market_probability=0.5, edge=0.0, american_odds=-110, created_at="2098-12-31T13:00:00+00:00")
    for book in ("A", "B", "C"):
        store.insert_market_prediction(db, market="total", selection="over", point=224.5 if book != "C" else 225.5, bookmaker=book, **quote)
        store.insert_market_prediction(db, market="total", selection="under", point=224.5 if book != "C" else 225.5, bookmaker=book, **quote)
        store.insert_market_prediction(db, market="spread", selection="BOS", point=-3.5, bookmaker=book, **quote)
        store.insert_market_prediction(db, market="spread", selection="MIA", point=3.5, bookmaker=book, **quote)
    manifest = {"metrics": {"margin": {"residual_sigma": 13.0}, "total": {"residual_sigma": 18.0}}}

    pred, _ = _prediction_out(db, game, manifest)

    assert pred.market_point == -3.5 and pred.market_total_point == 224.5      # the line most books agree on
    # home must win by more than 3.5: P(margin 5 > 3.5, sigma 13) just above 50%; over 224.5 with a 230 total: ~62%
    assert 0.53 < pred.cover_prob_spread < 0.55
    assert 0.60 < pred.cover_prob_total < 0.64


def test_no_quoted_lines_means_no_cover_or_over_chance(tmp_path):
    from nba_predictor.api.routes import _prediction_out
    from nba_predictor.tracking import store

    db = tmp_path / "t.db"
    store.init_db(db)
    game = {"game_id": "g1", "home_team": "BOS", "away_team": "MIA", "game_date": "2099-01-01", "tip_off": "2099-01-01T23:00Z", "completed": False}
    store.insert_prediction(db, game_id="g1", created_at="2098-12-31T12:00:00+00:00", model_version="v", home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0)
    pred, _ = _prediction_out(db, game, {"metrics": {"margin": {"residual_sigma": 13.0}, "total": {"residual_sigma": 18.0}}})
    assert pred.market_point is None and pred.cover_prob_spread is None and pred.cover_prob_total is None


def test_market_home_win_is_the_latest_pre_tip_moneyline_and_never_defaulted(tmp_path):
    from nba_predictor.api.routes import _prediction_out
    from nba_predictor.tracking import store

    db = tmp_path / "t.db"
    store.init_db(db)
    game = {"game_id": "g1", "home_team": "BOS", "away_team": "MIA", "game_date": "2099-01-01", "tip_off": "2099-01-01T23:00Z", "completed": False}
    store.insert_prediction(db, game_id="g1", created_at="2098-12-31T12:00:00+00:00", model_version="v", home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0)
    manifest = {"metrics": {}}
    assert _prediction_out(db, game, manifest)[0].market_home_win_probability is None
    for at, p in (("2098-12-31T13:00:00+00:00", 0.55), ("2099-01-01T01:00:00+00:00", 0.58), ("2099-01-02T01:00:00+00:00", 0.99)):
        store.insert_market_prediction(db, game_id="g1", market="h2h", selection="BOS", model_probability=0.62, market_probability=p, edge=0.0, bookmaker="K", american_odds=None, point=None, created_at=at)
    assert _prediction_out(db, game, manifest)[0].market_home_win_probability == 0.58  # the post-tip 0.99 is ignored
