"""Tonight's report as the availability input: Out counts, Day-To-Day falls back, unlisted plays; off by default."""
import pandas as pd

from nba_predictor import config
from nba_predictor.features.availability import compute_availability_features
from nba_predictor.pipeline import ingest

GAMES = pd.DataFrame([dict(game_id="g1", game_date="2026-11-01", home_team="BOS", away_team="NYK"),
                      dict(game_id="g2", game_date="2026-11-03", home_team="BOS", away_team="NYK")])
# g1: BOS players a,b,c play, d sits (0 min). g2 is tonight.
PLAYERS = pd.DataFrame([dict(game_id="g1", game_date="2026-11-01", team="BOS", player_id=p, minutes=m,
                             points=10, rebounds=0, assists=0) for p, m in [("a", 30), ("b", 30), ("c", 30), ("d", 0)]])


def _home_share(report_by_game=None):
    return compute_availability_features(GAMES, PLAYERS, report_by_game).set_index("game_id").loc["g2", "home_absent_min_share"]


def test_report_replaces_previous_game_only_where_given():
    assert _home_share() == 0.0                                   # d sat but has no cumulative minutes: not in the rotation
    assert _home_share({"g2": {"a": "Out"}}) == 1 / 3              # a played last game, Out tonight -> missing
    assert _home_share({"g2": {"a": "Day-To-Day"}}) == 0.0         # DTD falls back to "played last game"
    assert _home_share({"g1": {"a": "Out"}}) == 0.0                # a report for another game does not leak into g2


def test_tonights_report_flag(monkeypatch):
    games = [{"game_id": "g1", "completed": True}, {"game_id": "g2", "completed": False}]
    feed = lambda: [{"player_id": "7", "status": "Out"}, {"player_id": "", "status": "Out"}]  # noqa: E731
    monkeypatch.setattr(config, "INJURY_REPORT_AVAILABILITY", False)
    assert ingest._tonights_report(games, feed) is None
    monkeypatch.setattr(config, "INJURY_REPORT_AVAILABILITY", True)
    assert ingest._tonights_report(games, feed) == {"g2": {"7": "Out"}}

    def down():
        raise RuntimeError("feed down")
    assert ingest._tonights_report(games, down) is None
