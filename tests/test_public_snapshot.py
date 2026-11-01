import gzip
import json
from datetime import datetime, timedelta, timezone


def _fresh_stamp() -> str:
    """A recent, parseable `trained_at`.

    These fixtures used to carry `trained_at: "t"`, which the staleness gate
    (spec section 10) now refuses on purpose: an unparseable stamp is
    indistinguishable from a fresh one.
    """
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_generate_snapshot_bundles_all_sources(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    schedule_path = tmp_path / "schedule" / "games.json"
    _write_json(schedule_path, [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])

    hub_dir = tmp_path / "hub"
    _write_json(hub_dir / "teams.json", [{"abbreviation": "BOS"}])
    _write_json(hub_dir / "players.json", [])
    _write_json(hub_dir / "rankings.json", [])
    _write_json(hub_dir / "standings.json", [])

    manifest_path = tmp_path / "models" / "manifest.json"
    _write_json(manifest_path, {"model_version": "v1", "trained_at": _fresh_stamp(), "models": [], "metrics": {}})

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DK", american_odds=-130, created_at="2026-11-01T00:00:00",
    )

    snapshot = generate_snapshot(schedule_path, hub_dir, manifest_path, db_path)

    assert snapshot["schedule"][0]["game_id"] == "g1"
    assert snapshot["hub"]["teams"][0]["abbreviation"] == "BOS"
    assert snapshot["hub"]["players"] == []
    assert snapshot["manifest"]["model_version"] == "v1"
    assert snapshot["track_record"][0]["market"] == "h2h"
    assert "generated_at" in snapshot


def test_generate_snapshot_manifest_none_when_missing(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    snapshot = generate_snapshot(
        tmp_path / "schedule" / "games.json", tmp_path / "hub", tmp_path / "models" / "manifest.json", db_path
    )
    assert snapshot["manifest"] is None


def test_write_snapshot_creates_valid_json(tmp_path):
    from nba_predictor.public_snapshot import write_snapshot

    output_path = tmp_path / "public_snapshot.json"
    write_snapshot({"generated_at": "2026-11-01T00:00:00", "schedule": []}, output_path)

    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded["schedule"] == []


def test_main_writes_to_configured_output_path(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    store.init_db(tmp_path / "tracking.db")

    main()

    assert (tmp_path / "public_snapshot.json").exists()


def test_main_initializes_tracking_db_when_it_does_not_exist_yet(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    main()  # must not raise even though tracking.db has never been created

    assert (tmp_path / "tracking.db").exists()
    assert (tmp_path / "public_snapshot.json").exists()                        


def test_main_appends_each_pick_to_history(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])
    )

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-18T12:00:00+00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=3.0, predicted_total=220.0,
    )
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(),
                    "models": [], "metrics": {"total": {"residual_sigma": 15.0},
                                              "margin": {"residual_sigma": 12.0}}})
    )

    main()

    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == "g1"
    assert row["game_date"] == "2026-11-01"
    assert row["home_team"] == "BOS" and row["away_team"] == "MIA"
    assert row["home_win_prob"] == 0.60
    assert row["predicted_margin"] == 3.0
    assert row["predicted_total"] == 220.0
    assert row["total_sigma"] == 15.0
    # The margin's sigma, so the ledger can price a spread rung: a spread
    # market needs P(margin > line) and the margin's residual sigma is what
    # that is computed from.
    assert row["margin_sigma"] == 12.0
    assert row["model_version"] == "v1"
    assert row["created_at"] == "2026-10-18T12:00:00+00:00"


def test_the_pick_history_carries_the_margin_sigma_the_manifest_publishes(tmp_path, monkeypatch):
    """The manifest publishes `metrics.margin.residual_sigma`; the feed must carry it.

    Without it the ledger cannot price a spread rung at all, because a spread
    needs P(margin > line) and that probability is computed from the margin's
    own sigma. A missing sigma is not a zero and not a guess.
    """
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])
    )
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-18T12:00:00+00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=3.0, predicted_total=220.0,
    )
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(),
                    "models": [], "metrics": {"total": {"residual_sigma": 15.0},
                                              "margin": {"residual_sigma": 12.5}}})
    )

    main()

    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert rows[0]["margin_sigma"] == 12.5


def test_a_manifest_with_no_margin_sigma_leaves_it_absent(tmp_path, monkeypatch):
    """A model that publishes no margin sigma leaves the field out.

    Not zero: zero would be a claim that the margin is certain, which is a
    different claim from "we do not know".
    """
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])
    )
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-18T12:00:00+00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=3.0, predicted_total=220.0,
    )
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(),
                    "models": [], "metrics": {"total": {"residual_sigma": 15.0}}})
    )

    main()

    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert "margin_sigma" not in rows[0] or rows[0]["margin_sigma"] is None


def test_pick_history_keeps_every_version_and_never_overwrites(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    schedule_path = tmp_path / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(
        json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])
    )

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"model_version": "v1", "trained_at": _fresh_stamp(),
                    "models": [], "metrics": {"total": {"residual_sigma": 15.0}}})
    )

    store.insert_prediction(db_path, game_id="g1", created_at="2026-10-18T12:00:00+00:00",
                            model_version="v1", home_win_prob=0.60, predicted_margin=3.0, predicted_total=220.0)
    main()
    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert len(rows) == 1
    first = rows[0]["created_at"], rows[0]["home_win_prob"]

    store.insert_prediction(db_path, game_id="g1", created_at="2026-10-19T12:00:00+00:00",
                            model_version="v2", home_win_prob=0.55, predicted_margin=2.0, predicted_total=218.0)
    main()
    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert len(rows) == 2
    assert rows[0]["created_at"] == first[0] and rows[0]["home_win_prob"] == first[1]
    assert rows[1]["created_at"] == "2026-10-19T12:00:00+00:00"
    assert rows[1]["home_win_prob"] == 0.55

    main()
    rows = [json.loads(line) for line in gzip.open(tmp_path / "picks_history.jsonl.gz", "rt")]
    assert len(rows) == 2
