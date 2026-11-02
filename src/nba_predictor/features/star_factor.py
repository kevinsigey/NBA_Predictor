"""star_factor.py -- vacated production: what a team's absent rotation leaves behind.

The prop models project a player from his own rolling means and the matchup.
Nothing in that frame knows that the team's star sat last night, so on the
nights a rotation player is out, every teammate's projection is biased low by
roughly the share of the absent man's production the survivors absorb.
Measured on 2018-19..2024-25 (causal study, teammates vs their own shift(1)
rolling baselines, temporary absences): points +2.17 [+2.05, +2.29], rebounds
+0.61, assists +0.55, threes +0.18 -- and the dose variable is not the absent
player's usage share but his PER-STAT production (his rebounds predict the
rebound bump, his assists the assist bump). So the features are per-stat sums
of what the absent rotation players were producing, not one "star is out"
dummy -- though that flag is here too (`star_out`), because the first game of
an absence runs hotter than the sum alone explains.

**Absence is the 0-minutes + 1-game proxy (Kevin's call, 2026-11-01).** A
rotation player who recorded 0 minutes in his team's previous completed game
is absent for the current game, and stays absent until he records minutes
again. Box scores only, no injury feed: the feed is real (data/espn.py) but
it is not archived, so a feature the feed defined could not be rebuilt for
training rows -- train and serve would be measuring different things. The
proxy is reproducible for every historical row and for tonight's slate alike.

Population and mechanics:

* The rotation is a team's top 8 by cumulative minutes BEFORE the game --
  the same population as features/availability.py, ranked the same way.
* A player's contribution to the vacated sums is his own shift(1) last-10
  rolling baseline per stat (the same window player_stats.py rolls), so the
  sums are in the units the bump was measured in.
* `star_out`: the absent set includes the team's #1 usage player -- highest
  season-to-date (points + assists) per game among players averaging at
  least 25 minutes, all computed before the game.
* The sums on a player's row exclude the player himself: vacated production
  is what his absent TEAMMATES leave for him. A returning player's own
  proxy-absence is not his own opportunity, and he is not his own star-out.
* Each season is scored over the window starting the previous season's
  Oct 1, exactly like compute_availability_by_season, so a seven-season
  training frame and the production window give the columns the same
  meaning (a plain cumulative pass over years would leave departed players
  in the top 8 forever -- the failure that function exists to prevent).

Causality: a game's features are recorded BEFORE its own box score is folded
into state, so changing a game's result cannot move that game's features.
Upcoming games (no stats yet) are emitted from state as it stands and never
folded in.

Default (documented): a team-game with no previous completed game in the
window gets 0.0 on every column -- "nobody known to be missing" -- the same
no-history default as the availability features. The columns are never NaN.

Input contract: one row per (player, game), as both production frames
guarantee (the training frame by construction, the scoring frame by an
explicit dedupe). Columns read: player_id, team, game_id, game_date,
minutes, points, rebounds, assists, fg3m.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

STAR_FACTOR_COLUMNS = [
    "vac_pts", "vac_reb", "vac_ast", "vac_fg3m", "vac_min", "vac_n", "star_out",
]
ROTATION_SIZE = 8
ROLLING_WINDOW = 10
#: The #1-usage title only counts players who actually carry a starter's load.
STAR_MIN_MINUTES_PER_GAME = 25.0
NO_HISTORY_DEFAULT = 0.0

_STATS = ("points", "rebounds", "assists", "fg3m", "minutes")
#: feature column -> the stat whose vacated baseline feeds it
_VAC_OF_STAT = (
    ("vac_pts", "points"), ("vac_reb", "rebounds"), ("vac_ast", "assists"),
    ("vac_fg3m", "fg3m"), ("vac_min", "minutes"),
)
#: The magnitude columns: how much production the absent rotation teammates
#: were producing. The gate below zeroes exactly these.
GATED_MAGNITUDE_COLUMNS = [col for col, _ in _VAC_OF_STAT]


def apply_star_out_gate(features: pd.DataFrame) -> pd.DataFrame:
    """The star_out gate (trial 2, preregistered in
    docs/star-factor-gated-preregistration-2026-11.md): the magnitude columns
    keep their value only on rows where the absent set includes the team's
    #1 usage player (`star_out == 1`); everywhere else they are 0.0. `vac_n`
    and `star_out` pass through ungated.

    Trial 1's ungated columns moved the absence-slice bias the right way but
    converted to no MAE, on a slice (vac_n > 0) covering 95.8% of rows; the
    causal study's effect lives in the ~12% of team-games where a star sits.
    The gate concentrates the magnitudes on that population. Applied by
    build_prop_model_frame, the one path training and serving share, so the
    gated columns mean the same thing in both. The input is not mutated.
    """
    gated = features.copy()
    gated.loc[gated["star_out"] == 0, GATED_MAGNITUDE_COLUMNS] = 0.0
    return gated


def _season_of(game_date: str) -> int:
    """NBA season (start year) of an ISO date: October on is the new season."""
    year, month = int(game_date[:4]), int(game_date[5:7])
    return year if month >= 10 else year - 1


def compute_star_factor_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """STAR_FACTOR_COLUMNS for every input row, aligned to the input's index.

    Rows are emitted by the window of their own game's season (see the module
    docstring); every row gets exactly one value per column, default 0.0.
    Values accumulate into positional numpy arrays, not per-row .loc writes --
    on a seven-season frame the .loc version costs minutes for arithmetic the
    pass itself does in seconds.
    """
    values = {col: np.full(len(player_games), NO_HISTORY_DEFAULT) for col in STAR_FACTOR_COLUMNS}
    if not len(player_games):
        return pd.DataFrame(values, index=player_games.index, columns=STAR_FACTOR_COLUMNS)

    g = player_games[["player_id", "team", "game_id", "game_date", *_STATS]].copy()
    for col in _STATS:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g["pos"] = range(len(g))
    seasons = g["game_date"].map(_season_of)

    for season in sorted(seasons.unique()):
        lo, hi = f"{season - 1}-10-01", f"{season + 1}-09-30"
        in_window = (g["game_date"] >= lo) & (g["game_date"] <= hi)
        keep = set(g.loc[seasons == season, "pos"])
        _run_window(g[in_window], keep, values)
    return pd.DataFrame(values, index=player_games.index, columns=STAR_FACTOR_COLUMNS)


def _run_window(window: pd.DataFrame, keep: set[int], values: dict[str, np.ndarray]) -> None:
    """One chronological pass over a window's team-games, writing kept rows into `values`."""
    # Team-games in global date order. A traded player's rolling baseline is
    # his own last 10 wherever he played them (player_stats.py rolls across
    # teams the same way), so the history deques are per player, not per team;
    # rotation / absence / star state is per team.
    team_games: dict[tuple, list] = {}
    for row in window.itertuples():
        team_games.setdefault((row.game_date, row.game_id, row.team), []).append(row)
    ordered = sorted(team_games.items(), key=lambda kv: kv[0])

    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    cum_min: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cum_pts_ast: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    games_played: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prev_played: dict[str, set | None] = defaultdict(lambda: None)

    for (_, _, team), rows in ordered:
        t_min, t_pa, t_gp = cum_min[team], cum_pts_ast[team], games_played[team]
        prev = prev_played[team]

        if prev is not None and t_min:
            rotation = sorted(t_min, key=lambda p: (-t_min[p], str(p)))[:ROTATION_SIZE]
            absent = [p for p in rotation if p not in prev]
            baselines = {
                p: _baseline(history[p]) for p in absent
            }
            totals = {
                col: sum(b[stat] for b in baselines.values()) for col, stat in _VAC_OF_STAT
            }
            star = _star(t_min, t_pa, t_gp)
            for row in rows:
                if row.pos not in keep:
                    continue
                own = baselines.get(row.player_id)
                for col, stat in _VAC_OF_STAT:
                    values[col][row.pos] = totals[col] - (own[stat] if own is not None else 0.0)
                values["vac_n"][row.pos] = float(len(absent) - (1 if own is not None else 0))
                values["star_out"][row.pos] = float(
                    star is not None and star in baselines and star != row.player_id
                )

        # Fold this game in only if it has been played (an upcoming row's stats
        # are all NaN); its features above were recorded before this point.
        played = [r for r in rows if pd.notna(r.minutes)]
        if not played:
            continue
        prev_played[team] = {r.player_id for r in played if r.minutes > 0}
        for r in played:
            if r.minutes <= 0:
                continue  # dressed but never entered: not a game he played
            history[r.player_id].append(tuple(
                0.0 if pd.isna(getattr(r, stat)) else float(getattr(r, stat)) for stat in _STATS
            ))
            t_min[r.player_id] += float(r.minutes)
            t_pa[r.player_id] += (0.0 if pd.isna(r.points) else float(r.points)) + (
                0.0 if pd.isna(r.assists) else float(r.assists)
            )
            t_gp[r.player_id] += 1


def _baseline(hist: deque) -> dict[str, float]:
    """Mean of a player's last-10 played games per stat (his shift(1) rolling baseline)."""
    n = len(hist)
    return {stat: sum(game[i] for game in hist) / n for i, stat in enumerate(_STATS)}


def _star(cum_min: dict, cum_pts_ast: dict, games_played: dict) -> str | None:
    """The team's #1 usage player so far, or None if nobody carries 25+ minutes a game."""
    eligible = [
        p for p, gp in games_played.items()
        if gp > 0 and cum_min[p] / gp >= STAR_MIN_MINUTES_PER_GAME
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda p: (-(cum_pts_ast[p] / games_played[p]), str(p)))
