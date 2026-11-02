"""The track-record summary no longer inlines the per-pick disclosure.

Measured 2026-11-01: GET /hub/track-record was 14,367,011 bytes on the live
tracker -- ~68k per-pick rows (game_outcome alone: 39,454) inlined into a
summary whose page defers *rendering* the every-pick table until its fold
opens, but downloaded all of it up front. On a phone that was the worst
payload in the estate.

The split:

  1. `compute_track_record(..., include_picks=False)` -- what the summary
     route serves -- returns identical aggregates with `per_pick` empty and
     a bounded `recent_picks` (the latest counted pre-tip picks, the only
     rows the summary page renders) lifted out of the disclosure.
  2. `compute_track_record_picks` pages the disclosure itself: the same
     rows from the same builder, newest stamp first, with `market` and
     `game_ids` filters and a `total` that counts the filtered set.
  3. The default builder is unchanged: every existing caller and test that
     reads `per_pick` off `compute_track_record(db, schedule)` still can.
"""
import json

from nba_predictor.services import hub_service
from nba_predictor.tracking import store


def _completed(game_id, home, away, home_pts, away_pts, **over):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts, **over,
    }


def _db(tmp_path):
    db = tmp_path / "tracking.db"
    store.init_db(db)
    return db


def _outcome(db, game_id, prob, at, *, version="v1"):
    return store.insert_prediction(
        db, game_id=game_id, created_at=at, model_version=version,
        home_win_prob=prob, predicted_margin=1.0, predicted_total=220.0,
    )


def _row(records, market):
    return next(r for r in records if r.market == market)


def _seeded_db(tmp_path):
    """Three games: a counted pre-tip pick, a rerun of it, a rebuilt pick,
    and one h2h market row -- the shapes the split has to keep straight."""
    db = _db(tmp_path)
    _outcome(db, "g1", 0.7, "2026-03-01T09:00:00+00:00")
    _outcome(db, "g1", 0.6, "2026-03-01T10:00:00+00:00", version="v2")  # rerun, not counted
    _outcome(db, "g2", 0.4, "2026-09-20T08:00:00+00:00", version="v2")  # after tip, counted
    store.insert_market_prediction(
        db, game_id="g1", market="h2h", selection="BOS", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-120,
        created_at="2026-03-01T09:30:00+00:00",
    )
    schedule = [
        _completed("g1", "BOS", "MIA", 110, 100),
        _completed("g2", "LAL", "GSW", 95, 105),
    ]
    return db, schedule


# --- the summary split ---------------------------------------------------------

def test_summary_drops_per_pick_but_keeps_identical_aggregates(tmp_path):
    db, schedule = _seeded_db(tmp_path)

    full = hub_service.compute_track_record(db, schedule)
    summary = hub_service.compute_track_record(db, schedule, include_picks=False)

    assert [r.market for r in summary] == [r.market for r in full]
    for slim, whole in zip(summary, full):
        assert slim.per_pick == []
        assert (slim.total_predictions, slim.correct_predictions, slim.hit_rate) == (
            whole.total_predictions, whole.correct_predictions, whole.hit_rate,
        )
        assert (slim.pre_tip.total_predictions, slim.pre_tip.correct_predictions) == (
            whole.pre_tip.total_predictions, whole.pre_tip.correct_predictions,
        )
        assert slim.n_rebuilt == whole.n_rebuilt
        assert slim.weekly == whole.weekly


def test_summary_recent_picks_are_the_counted_pre_tip_rows_newest_first(tmp_path):
    db, schedule = _seeded_db(tmp_path)

    games = _row(hub_service.compute_track_record(db, schedule, include_picks=False), "game_outcome")

    # g1's counted pick is in; its rerun (counted=False) and g2's after-tip
    # pick (made_before_tip=False) are not -- exactly the population the
    # page's "Latest counted picks" block has always rendered.
    assert [p.game_id for p in games.recent_picks] == ["g1"]
    assert games.recent_picks[0].counted is True
    assert games.recent_picks[0].made_before_tip is True


def test_recent_picks_are_bounded_per_market(tmp_path):
    db = _db(tmp_path)
    schedule = []
    for i in range(hub_service.RECENT_PICKS_PER_MARKET + 6):
        gid = f"g{i:02d}"
        _outcome(db, gid, 0.7, f"2026-03-01T09:{i:02d}:00+00:00")
        schedule.append(_completed(gid, "BOS", "MIA", 110, 100))

    games = _row(hub_service.compute_track_record(db, schedule, include_picks=False), "game_outcome")

    assert len(games.recent_picks) == hub_service.RECENT_PICKS_PER_MARKET
    stamps = [p.created_at for p in games.recent_picks]
    assert stamps == sorted(stamps, reverse=True), "newest first"


# --- the paged disclosure --------------------------------------------------------

def test_picks_page_through_the_whole_filtered_set(tmp_path):
    db, schedule = _seeded_db(tmp_path)

    first = hub_service.compute_track_record_picks(db, schedule, limit=2, offset=0)
    second = hub_service.compute_track_record_picks(db, schedule, limit=2, offset=2)

    # g1's counted pick, its rerun, g2's rebuilt pick, and g1's h2h row.
    assert first.total == 4
    assert len(first.picks) == 2 and len(second.picks) == 2
    seen = [(p.market, p.game_id, p.created_at) for p in first.picks + second.picks]
    assert len(set(seen)) == 4, "pages must not overlap"
    stamps = [p.created_at for p in first.picks + second.picks]
    assert stamps == sorted(stamps, reverse=True), "newest stamp first across pages"


def test_picks_filter_by_market_and_game_ids(tmp_path):
    db, schedule = _seeded_db(tmp_path)

    by_market = hub_service.compute_track_record_picks(db, schedule, market="game_outcome")
    assert by_market.total == 3
    assert all(p.market == "game_outcome" for p in by_market.picks)

    by_game = hub_service.compute_track_record_picks(db, schedule, game_ids=["g1"])
    assert by_game.total == 3  # outcome counted + outcome rerun + h2h counted
    assert all(p.game_id == "g1" for p in by_game.picks)
    counted = [p for p in by_game.picks if p.counted]
    assert len(counted) == 2, "the disclosure keeps the counted flag per row"
    assert any(p.counted is False for p in by_game.picks), "and the rerun stays labelled"


# --- the routes ------------------------------------------------------------------

def _client(tmp_path, monkeypatch, schedule):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor import config

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sched_dir = tmp_path / "cache" / "schedule"
    sched_dir.mkdir(parents=True)
    (sched_dir / "games.json").write_text(json.dumps(schedule))
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    return TestClient(app), db_path


def test_routes_serve_the_split(tmp_path, monkeypatch):
    schedule = [
        _completed("g1", "BOS", "MIA", 110, 100),
        _completed("g2", "LAL", "GSW", 95, 105),
    ]
    client, db_path = _client(tmp_path, monkeypatch, schedule)
    _outcome(db_path, "g1", 0.7, "2026-03-01T09:00:00+00:00")
    _outcome(db_path, "g2", 0.4, "2026-09-20T08:00:00+00:00", version="v2")

    body = client.get("/hub/track-record").json()
    games = next(r for r in body if r["market"] == "game_outcome")
    assert games["per_pick"] == []
    assert [p["game_id"] for p in games["recent_picks"]] == ["g1"]
    assert games["total_predictions"] == 2

    page = client.get("/hub/track-record/picks", params={"limit": 1, "offset": 0}).json()
    assert page["total"] == 2 and len(page["picks"]) == 1
    assert page["picks"][0]["game_id"] == "g2", "newest stamp first"

    by_game = client.get(
        "/hub/track-record/picks", params=[("game_id", "g1"), ("game_id", "g2")]
    ).json()
    assert by_game["total"] == 2
