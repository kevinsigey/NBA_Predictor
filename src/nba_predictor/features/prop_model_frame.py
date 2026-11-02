"""The prop models' feature frame.

His own rolling means (features/player_stats.py) plus what chronological-holdout evidence says is worth having
(nba-eval/props_race2.py, 2021-2025, games played, paired bootstrap vs "his rolling mean"): minutes (last game, a
5-game EWMA, the trend), the matchup block already in features/prop_matchup.py (opponent defence against his position,
rest, usage trend) and home/away. With all of them the model beats the rolling mean in points (4.737 -> 4.661, interval
[-0.082,-0.068]), rebounds (-0.026) and assists (-0.013), and ties on threes; the five rolling means alone only tie it.

Plus the vacated-production block (features/star_factor.py): per-stat sums of what the game's absent rotation
teammates were producing, under the 0-minutes + 1-game absence proxy. Those columns are never NaN -- a team-game
with no previous completed game gets the documented 0.0 ("nobody known to be missing"), like the availability
features, because an invented absence would be worse than a stated default.

Everything is shift(1)-based, so an upcoming row (no stats yet) is built from history only. XGBoost reads NaN as
"missing", so a first game or an unmapped position is left NaN rather than invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_predictor.features.player_stats import PLAYER_FEATURE_COLUMNS, build_player_feature_frame
from nba_predictor.features.prop_matchup import NEW_FEATURE_COLUMNS, build_prop_features
from nba_predictor.features.star_factor import STAR_FACTOR_COLUMNS, compute_star_factor_features

EXTRA_FEATURE_COLUMNS = NEW_FEATURE_COLUMNS + ["is_home", "min_last", "min_ewm5"]
PROP_FEATURE_COLUMNS = PLAYER_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS + STAR_FACTOR_COLUMNS

#: ESPN's roster endpoint sends PG/SG/SF/PF/C; its box scores send G/F/C. One position group per player either way.
_POSITION_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def _group(position):
    return _POSITION_GROUP.get(position.strip().upper(), position) if isinstance(position, str) else position


def build_prop_model_frame(player_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """(frame, feature columns): the rolling-mean frame with the extra columns, rows with no rolling history dropped."""
    games = player_games.copy()
    # An upcoming row's stats are None, which makes the whole column object-typed after a concat; XGBoost refuses object.
    for col in ("points", "rebounds", "assists", "fg3m", "minutes", "fga", "fgm"):
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")
    for col in ("team", "opponent", "position"):          # absent = unknown: the matchup features come out NaN, never invented
        if col not in games.columns:
            games[col] = pd.Series([None] * len(games), index=games.index, dtype=object)
    games["position"] = games["position"].map(_group).astype(object)
    # A game with no known opponent (only ever a degenerate fixture; real rows always have one) pools under one label,
    # which makes opp_def_vs_pos the league rate for his position rather than a crash on an all-null merge key.
    games["opponent"] = games["opponent"].astype(object).where(games["opponent"].notna(), "UNKNOWN")
    if games["position"].isin(["G", "F", "C"]).any():
        m = build_prop_features(games)
    else:
        # No row has a known position (only ever a degenerate fixture): there is no defence-vs-position to compute.
        m = games.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        for col in NEW_FEATURE_COLUMNS:
            m[col] = np.nan
    home = m["home_team"] if "home_team" in m.columns else pd.Series(np.nan, index=m.index)
    m["is_home"] = np.where(home.notna(), (m["team"] == home).astype(float), np.nan)
    minutes = m.sort_values(["player_id", "game_date"]).groupby("player_id")["minutes"]
    m["min_last"] = minutes.shift(1).reindex(m.index)
    m["min_ewm5"] = minutes.transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean()).reindex(m.index)
    # Vacated production (features/star_factor.py): what this game's absent rotation teammates were producing.
    # Same function on the training frame and the scoring frame, so the columns mean the same thing in both.
    m[STAR_FACTOR_COLUMNS] = compute_star_factor_features(m)
    frame, _ = build_player_feature_frame(m)
    frame[PROP_FEATURE_COLUMNS] = frame[PROP_FEATURE_COLUMNS].astype(float)
    return frame, PROP_FEATURE_COLUMNS
