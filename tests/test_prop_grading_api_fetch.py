"""The CI snapshot build has no tracking DB, so an empty prop grading is
filled from the live API -- the same remedy the pick history already uses.

Before this, every published snapshot carried the empty grading forever:
correct bands and the counting text over no days, because generate_snapshot
graded a store CI never had. The local grading always wins when the build
has one; the fetch fills a gap and fails soft.
"""
import io
import json
from datetime import datetime, timedelta, timezone

from nba_predictor import public_snapshot as ps


def _fresh_stamp() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _api(payload):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda req, timeout=None: Resp(json.dumps(payload).encode())


def _grading_payload(date="2026-11-01"):
    return {
        "bands": {"points": 4.0},
        "counting": "counting text",
        "confidence_basis": "confidence text",
        "grade_basis": "grade text",
        "latest_day": {"date": date, "n_graded": 2, "n_pre_tip": 2, "games": []},
        "most_confident": [{"player_id": "A"}],
        "biggest_surprises": [{"player_id": "A"}],
        "n_with_baseline": 1,
        "served_stale_seconds": 42.0,
    }


# ---------------------------------------------------------------------------
# The fetch itself
# ---------------------------------------------------------------------------


def test_fetch_prop_grading_returns_the_served_payload_without_its_cache_stamp(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _api(_grading_payload()))
    grading = ps.fetch_prop_grading("https://api.test")
    assert grading["latest_day"]["date"] == "2026-11-01"
    assert grading["bands"] == {"points": 4.0}
    # The stale stamp describes the API process's cache, not this copy.
    assert "served_stale_seconds" not in grading


def test_fetch_prop_grading_returns_none_when_the_api_has_no_graded_day(monkeypatch):
    payload = _grading_payload()
    payload["latest_day"] = None
    payload["most_confident"] = []
    payload["biggest_surprises"] = []
    monkeypatch.setattr("urllib.request.urlopen", _api(payload))
    assert ps.fetch_prop_grading("https://api.test") is None


def test_fetch_prop_grading_returns_none_for_a_payload_that_is_not_the_grading(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _api({"detail": "not found"}))
    assert ps.fetch_prop_grading("https://api.test") is None


# ---------------------------------------------------------------------------
# The fill rule: local wins, the fetch only fills an empty grading
# ---------------------------------------------------------------------------


def test_fill_replaces_an_empty_local_grading(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _api(_grading_payload()))
    snapshot = {"prop_grading": {**_grading_payload(), "latest_day": None, "most_confident": [], "biggest_surprises": []}}
    assert ps.fill_prop_grading_from_api(snapshot, "https://api.test") is True
    assert snapshot["prop_grading"]["latest_day"]["date"] == "2026-11-01"


def test_fill_never_overrules_a_local_grading(monkeypatch):
    def forbidden(req, timeout=None):
        raise AssertionError("the API must not be called when the local grading has a day")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    local = _grading_payload(date="2026-10-31")
    snapshot = {"prop_grading": local}
    assert ps.fill_prop_grading_from_api(snapshot, "https://api.test") is False
    assert snapshot["prop_grading"] is local


# ---------------------------------------------------------------------------
# main(): the CI shape end to end
# ---------------------------------------------------------------------------


def _route_api(grading=None, fail_grading=False):
    def fetch(req, timeout=None):
        if "prop-grading" in req.full_url:
            if fail_grading:
                raise OSError("connection refused")
            return _api(grading if grading is not None else _grading_payload())(req)
        return _api({"rows": []})(req)

    return fetch


def _point_main_at(tmp_path, monkeypatch):
    from nba_predictor import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])
    )
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(), "models": [], "metrics": {}})
    )
    monkeypatch.setenv("PICKS_API", "https://api.test")


def _written_grading(tmp_path):
    snapshot = json.loads((tmp_path / "public_snapshot.json").read_text())
    return snapshot["prop_grading"]


def test_main_fills_an_empty_grading_from_the_api(tmp_path, monkeypatch):
    from nba_predictor.public_snapshot import main

    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", _route_api())
    main()
    grading = _written_grading(tmp_path)
    assert grading["latest_day"]["date"] == "2026-11-01"
    assert grading["most_confident"] == [{"player_id": "A"}]
    assert "served_stale_seconds" not in grading


def test_main_keeps_the_local_grading_when_the_store_has_one(tmp_path, monkeypatch):
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    _point_main_at(tmp_path, monkeypatch)
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="A", stat="points",
        predicted_value=20.0, created_at="2026-11-01T12:00:00+00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="A", stat="points",
        actual_value=26.0, recorded_at="2026-11-02T03:00:00+00:00",
    )
    # The API would hand back a different day; the local measurement wins.
    monkeypatch.setattr("urllib.request.urlopen", _route_api(grading=_grading_payload(date="2026-10-30")))
    main()
    grading = _written_grading(tmp_path)
    assert grading["latest_day"]["date"] == "2026-11-01"
    assert grading["biggest_surprises"][0]["delta"] == 6.0


def test_main_publishes_the_local_empty_when_the_grading_fetch_fails(tmp_path, monkeypatch):
    from nba_predictor.public_snapshot import main

    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", _route_api(fail_grading=True))
    main()  # fail soft: the snapshot still publishes, the grading stays local
    grading = _written_grading(tmp_path)
    assert grading["latest_day"] is None
    assert grading["most_confident"] == []
