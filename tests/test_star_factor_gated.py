"""The star_out gate (trial 2, preregistered in docs/star-factor-gated-preregistration-2026-11.md):
the vacated-production magnitudes register only when the absent set includes the team's #1 usage
player. `vac_n` and `star_out` pass through ungated; compute_star_factor_features itself keeps
trial 1's ungated semantics (its own tests pin those).

The world mirrors tests/test_features_star_factor.py: team X with star S (sits g3, so the spell
shows at g4) and rotation man R2 (sits g6, so the spell shows at g7), constant lines so every
rolling baseline is an exact number.
"""
import pandas as pd

from nba_predictor.features.prop_model_frame import build_prop_model_frame
from nba_predictor.features.star_factor import (
    GATED_MAGNITUDE_COLUMNS,
    STAR_FACTOR_COLUMNS,
    apply_star_out_gate,
    compute_star_factor_features,
)

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
_S_POINTS = {"g1": 20.0, "g2": 30.0, "g3": 26.0, "g4": 26.0, "g5": 26.0, "g6": 26.0, "g7": 26.0}
_SITS = {"g3": {"S"}, "g6": {"R2", "B"}}
_POSITIONS = {"S": "G", "R1": "G", "R2": "G", "R3": "F", "R4": "F", "R5": "F", "R6": "C", "R7": "C", "B": "C"}


def _prop_world():
    rows = []
    for i in range(1, 8):
        gid = f"g{i}"
        for team, opp in (("X", "Y"), ("Y", "X")):
            for pid, (minutes, pts, reb, ast, fg3m) in _X.items():
                if team == "X" and pid in _SITS.get(gid, set()):
                    continue
                rows.append(dict(
                    player_id=f"{pid}{team}", player_name=pid, team=team, opponent=opp, game_id=gid,
                    game_date=f"2026-03-{i:02d}", minutes=minutes,
                    points=(_S_POINTS[gid] if pid == "S" else pts), rebounds=reb, assists=ast, fg3m=fg3m,
                    position=_POSITIONS[pid], fga=10.0, fgm=5.0,
                    home_team="X", away_team="Y",
                ))
    return pd.DataFrame(rows)


def _row(frame, gid, pid):
    return frame[(frame.game_id == gid) & (frame.player_id == pid)].iloc[0]


def test_gate_zeroes_magnitudes_only_where_the_star_is_not_out():
    feats = pd.DataFrame({
        "vac_pts": [25.0, 12.0, 0.0], "vac_reb": [6.0, 4.0, 0.0], "vac_ast": [7.0, 3.0, 0.0],
        "vac_fg3m": [2.0, 1.0, 0.0], "vac_min": [30.0, 26.0, 0.0],
        "vac_n": [1.0, 1.0, 0.0], "star_out": [1.0, 0.0, 0.0],
    })
    gated = apply_star_out_gate(feats)
    for col in GATED_MAGNITUDE_COLUMNS:
        assert gated.loc[0, col] == feats.loc[0, col]      # star out: untouched
        assert gated.loc[1, col] == 0.0                    # rotation absence only: zeroed
    assert gated["vac_n"].tolist() == [1.0, 1.0, 0.0]      # counts pass through ungated
    assert gated["star_out"].tolist() == [1.0, 0.0, 0.0]
    assert feats.loc[1, "vac_pts"] == 12.0                 # the input is not mutated


def test_the_prop_frame_gates_a_non_star_absence_and_keeps_a_star_absence():
    world = _prop_world()
    raw = pd.concat(
        [world[["game_id", "player_id"]].reset_index(drop=True),
         compute_star_factor_features(world).reset_index(drop=True)], axis=1)
    assert _row(raw, "g7", "R1X")["vac_pts"] == 12.0       # R2's baseline, before the gate

    frame, _ = build_prop_model_frame(world)
    g7 = _row(frame, "g7", "R1X")                          # R2 sat g6; S played
    assert g7["vac_n"] == 1.0 and g7["star_out"] == 0.0
    for col in GATED_MAGNITUDE_COLUMNS:
        assert g7[col] == 0.0, col                         # the gate zeroed a real 12.0 / 26.0
    g4 = _row(frame, "g4", "R1X")                          # S sat g3: the star is out
    assert g4["vac_n"] == 1.0 and g4["star_out"] == 1.0
    assert g4["vac_pts"] == 25.0 and g4["vac_min"] == 30.0  # magnitudes survive intact
    assert frame[STAR_FACTOR_COLUMNS].notna().all().all()  # still never NaN
