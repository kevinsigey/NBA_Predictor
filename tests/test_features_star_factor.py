"""Vacated production (features/star_factor.py): the 0-minutes + 1-game absence proxy, its spell
logic, its causality, and its wiring into the prop model frame.

The world: team X with a star S, seven rotation players R1..R7 and a ninth man B who is never in
the top-8 rotation. Stats are constant per player (except S's points, 20 then 30) so every rolling
baseline is an exact number the assertions can name.
"""
import numpy as np
import pandas as pd

from nba_predictor.features.prop_model_frame import PROP_FEATURE_COLUMNS, build_prop_model_frame
from nba_predictor.features.star_factor import STAR_FACTOR_COLUMNS, compute_star_factor_features

# pid -> (minutes, points, rebounds, assists, fg3m); S's points vary by game and are set per row.
_X = {
    "S": (30.0, None, 6.0, 7.0, 2.0),
    "R1": (28.0, 16.0, 5.0, 4.0, 1.0),
    "R2": (26.0, 12.0, 4.0, 3.0, 1.0),
    "R3": (24.0, 10.0, 7.0, 2.0, 0.0),
    "R4": (22.0, 9.0, 3.0, 2.0, 1.0),
    "R5": (20.0, 8.0, 3.0, 1.0, 1.0),
    "R6": (18.0, 7.0, 2.0, 1.0, 0.0),
    "R7": (16.0, 6.0, 2.0, 1.0, 0.0),
    "B": (6.0, 2.0, 1.0, 0.0, 0.0),
}
_S_POINTS = {"g1": 20.0, "g2": 30.0, "g3": 26.0, "g4": 26.0, "g5": 26.0, "g6": 26.0, "g7": 26.0}  # g3: only X's S sits; Y's namesake plays
_SITS = {"g3": {"S"}, "g6": {"R2", "B"}}  # who records no minutes in that game


def _line(pid, gid, sitting):
    """One box-score line: the player's constant line, S's per-game points, or zeroes if he sits."""
    minutes, pts, reb, ast, fg3m = _X[pid]
    if sitting:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return minutes, (_S_POINTS[gid] if pid == "S" else pts), reb, ast, fg3m


def _world(sit_row=False):
    """Seven X games. S sits g3 (back for g4); R2 and B sit g6. `sit_row`: the sitter keeps a
    0-minute box-score row instead of vanishing from the frame -- both shapes occur in real data."""
    rows = []
    for i in range(1, 8):
        gid = f"g{i}"
        for pid in _X:
            sitting = pid in _SITS.get(gid, set())
            if sitting and not sit_row:
                continue
            minutes, pts, reb, ast, fg3m = _line(pid, gid, sitting)
            rows.append(dict(
                player_id=pid, player_name=pid, team="X", game_id=gid, game_date=f"2026-03-{i:02d}",
                minutes=minutes, points=pts, rebounds=reb, assists=ast, fg3m=fg3m,
            ))
    return pd.DataFrame(rows)


def _row(frame, gid, pid):
    return frame[(frame.game_id == gid) & (frame.player_id == pid)].iloc[0]


def _features(df):
    out = compute_star_factor_features(df)
    return pd.concat([df[["game_id", "player_id"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def test_no_history_is_the_documented_zero_default():
    f = _features(_world())
    for col in STAR_FACTOR_COLUMNS:
        assert (_row(f, "g1", "R1")[col] == 0.0), col          # no previous completed game exists yet
        assert (_row(f, "g2", "R1")[col] == 0.0), col          # everyone played g1: nobody is absent
        assert (_row(f, "g3", "R1")[col] == 0.0), col          # S sits g3, but that is only knowable at g4


def test_the_spell_the_game_after_the_star_sits():
    f = _features(_world())
    r1 = _row(f, "g4", "R1")                                    # S recorded no minutes in g3
    assert r1["vac_n"] == 1.0
    assert r1["vac_pts"] == 25.0                                # S's rolling baseline: mean(20, 30)
    assert r1["vac_reb"] == 6.0 and r1["vac_ast"] == 7.0 and r1["vac_fg3m"] == 2.0
    assert r1["vac_min"] == 30.0
    assert r1["star_out"] == 1.0                                # S is the #1 usage player and he is absent


def test_a_returning_players_own_row_excludes_himself():
    f = _features(_world())
    s = _row(f, "g4", "S")                                      # S plays g4; his proxy-absence is not his own opportunity
    assert s["vac_n"] == 0.0 and s["vac_pts"] == 0.0 and s["star_out"] == 0.0


def test_the_spell_ends_when_minutes_are_recorded_again():
    f = _features(_world())
    assert _row(f, "g5", "R1")["vac_n"] == 0.0                  # S played g4, so g5 knows nobody missing
    assert _row(f, "g5", "R1")["star_out"] == 0.0


def test_only_the_rotation_counts_and_the_star_is_not_the_only_absence_that_registers():
    f = _features(_world())
    r1 = _row(f, "g7", "R1")                                    # R2 and B both sat g6; B is 9th in minutes, outside the top 8
    assert r1["vac_n"] == 1.0                                   # R2 only
    assert r1["vac_pts"] == 12.0                                # R2's baseline
    assert r1["star_out"] == 0.0                                # S played g6


def test_a_zero_minute_row_is_the_same_absence_as_no_row():
    no_row = _features(_world(sit_row=False))
    zero_row = _features(_world(sit_row=True))
    merged = no_row.merge(zero_row, on=["game_id", "player_id"], suffixes=("_a", "_b"))
    assert len(merged) == len(no_row)                          # every shared row is compared
    for col in STAR_FACTOR_COLUMNS:
        pd.testing.assert_series_equal(merged[f"{col}_a"], merged[f"{col}_b"], check_names=False)


def test_features_for_a_game_do_not_depend_on_that_game_or_any_later_one():
    base = _world()
    f_full = _features(base)
    f_trunc = _features(base[base.game_id.isin(["g1", "g2", "g3", "g4"])])
    for pid in ("R1", "S", "R2"):
        for col in STAR_FACTOR_COLUMNS:
            assert _row(f_full, "g4", pid)[col] == _row(f_trunc, "g4", pid)[col], (pid, col)

    leaked = base.copy()
    leaked.loc[leaked.game_id == "g4", ["points", "rebounds", "assists"]] = 99.0   # g4's own box score moves
    f_leaked = _features(leaked)
    for pid in ("R1", "R2"):
        for col in STAR_FACTOR_COLUMNS:
            assert _row(f_leaked, "g4", pid)[col] == _row(f_full, "g4", pid)[col], (pid, col)


def test_an_empty_frame_is_an_empty_answer_with_the_columns():
    empty = _world().iloc[0:0]
    out = compute_star_factor_features(empty)
    assert list(out.columns) == STAR_FACTOR_COLUMNS and len(out) == 0


def _prop_world():
    rows = []
    positions = {"S": "G", "R1": "G", "R2": "G", "R3": "F", "R4": "F", "R5": "F", "R6": "C", "R7": "C", "B": "C"}
    for i in range(1, 8):
        gid = f"g{i}"
        for team, opp in (("X", "Y"), ("Y", "X")):
            for pid in _X:
                if team == "X" and pid in _SITS.get(gid, set()):
                    continue
                minutes, pts, reb, ast, fg3m = _line(pid, gid, sitting=False)
                rows.append(dict(
                    player_id=f"{pid}{team}", player_name=pid, team=team, opponent=opp, game_id=gid,
                    game_date=f"2026-03-{i:02d}", minutes=minutes,
                    points=pts, rebounds=reb, assists=ast, fg3m=fg3m,
                    position=positions[pid], fga=10.0, fgm=5.0,
                    home_team="X", away_team="Y",
                ))
    rows.append(dict(player_id="R1X", player_name="R1", team="X", opponent="Y", game_id="next",
                     game_date="2026-03-30", minutes=None, points=None, rebounds=None, assists=None,
                     fg3m=None, position="G", fga=None, fgm=None, home_team="X", away_team="Y"))
    return pd.DataFrame(rows)


def test_the_prop_frame_carries_the_columns_and_the_upcoming_row_sees_the_absence():
    frame, cols = build_prop_model_frame(_prop_world())
    assert cols == PROP_FEATURE_COLUMNS
    assert set(STAR_FACTOR_COLUMNS) <= set(cols)
    assert frame[STAR_FACTOR_COLUMNS].notna().all().all()      # a stated default, never an invented NaN
    up = frame[(frame.player_id == "R1X") & frame.points.isna()].iloc[0]
    # The last completed game before "next" is g7, in which everyone played -- so this row is the
    # no-absence case, and g4's spell must show on the g4 rows themselves.
    assert up["vac_n"] == 0.0
    g4 = frame[(frame.player_id == "R1X") & (frame.game_id == "g4")].iloc[0]
    assert g4["vac_n"] == 1.0 and g4["vac_pts"] == 25.0 and g4["star_out"] == 1.0
