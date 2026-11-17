"""A long track-record / prop-grading read must not delay /games.

`get_connection` holds the process-wide `_DB_LOCK` for its whole body, and the
aggregates read hundreds of thousands of rows inside it, so every other read
(the /games prediction lookup) queued behind them with no timeout: live /games
sat on "Loading games..." while /hub/track-record computed.
"""
import threading
import time

from nba_predictor.tracking import player_props, store


def test_games_read_not_blocked_by_slow_aggregate_read(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store.init_db(db)
    store.insert_player_outcome(db, game_id="g1", player_id="p1", stat="pts", actual_value=10.0, recorded_at="2026-11-01T00:00:00+00:00")

    inside = threading.Event()
    real = player_props._earliest_outcome

    def slow(rows):  # runs inside the aggregate's DB read
        inside.set()
        time.sleep(1.5)
        return real(rows)

    monkeypatch.setattr(player_props, "_earliest_outcome", slow)
    t = threading.Thread(target=player_props.resolved_player_props, args=(db, [{"game_id": "g1"}]))
    t.start()
    assert inside.wait(5)

    start = time.perf_counter()
    store.get_predictions_for_game(db, "g1")
    elapsed = time.perf_counter() - start
    t.join()
    assert elapsed < 0.5, f"/games read waited {elapsed:.2f}s behind an aggregate read"
