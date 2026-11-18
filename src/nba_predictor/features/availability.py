"""availability.py -- lagged rotation-availability proxy (single chronological pass).

For game G and team T: of T's top-8 rotation players (ranked by cumulative minutes in
T's games BEFORE G), the share of that rotation's minutes / production (pts + 1.2 reb +
1.5 ast) held by players who did NOT play in T's PREVIOUS completed game. A player who
missed the last game is more likely to be out again; this is known before tip-off and
uses box scores only (no injury feed).

Causality: the value for G is recorded BEFORE G's own box score is folded into state, so
changing G's box score cannot move G's features.

Unplayed / upcoming games, and completed games with no player box score, are emitted from
state as it stands (T's last game that HAS a box score and the cumulative minutes so far)
and are never folded in. They are never NaN.

Default (documented): a team with no player-box history yet (its first game, or no player
boxes at all) gets 0.0 absent share on both columns, i.e. "nobody known to be missing".

Evidence: NBA#74 follow-up, held-out 2022-23..2024-25 confirmation (docs/nba-availability-evaluation).
Reference implementation: predictor-hub docs/research/2026-10-13-nba-model-gaps-scripts/feats.py.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

AVAILABILITY_COLUMNS = [
    "home_absent_min_share", "away_absent_min_share",
    "home_absent_prod_share", "away_absent_prod_share",
]
ROTATION_SIZE = 8
NO_HISTORY_DEFAULT = 0.0
_PLAYER_COLS = ["game_id", "team", "player_id", "minutes", "points", "rebounds", "assists"]


def _absent(top, prev_played, report):
    """Who in `top` counts as missing. No report: whoever sat the previous game (the trained definition).
    With tonight's report ({player_id: status}, ESPN's words): Out is missing, Day-To-Day falls back to the previous
    game, and a player the report does not list is available -- a rest day last game is not an absence tonight."""
    if report is None:
        return [x for x in top if x not in prev_played]
    status = {str(k): str(v).strip().lower() for k, v in report.items()}
    return [x for x in top if status.get(str(x)) == "out" or (status.get(str(x)) == "day-to-day" and x not in prev_played)]


def compute_availability_features(games: pd.DataFrame, players: pd.DataFrame | None,
                                  report_by_game: dict | None = None) -> pd.DataFrame:
    """One row per game_id with the four AVAILABILITY_COLUMNS.

    `report_by_game`: {game_id: {player_id: status}} -- the injury report seen before that game's tip. A game with an
    entry is scored from the report (see `_absent`); every other game keeps the previous-game definition. Serving only:
    reports are stored from 2026-10-20 on, too few seasons to train on.

    `games`: game_id, game_date, home_team, away_team (played or not).
    `players`: one row per (player, completed game) as `ingest.to_player_training_frame`
    builds it; None or empty means no history (every value is NO_HISTORY_DEFAULT).
    """
    order = defaultdict(list)
    for r in games.sort_values(["game_date", "game_id"], kind="stable").itertuples():
        order[r.home_team].append(r.game_id)
        order[r.away_team].append(r.game_id)

    by_team_game = {}
    if players is not None and len(players):
        p = players.sort_values(["game_date", "game_id"], kind="stable")[_PLAYER_COLS]
        for (gid, team), g in p.groupby(["game_id", "team"], sort=False):
            by_team_game[(gid, team)] = list(zip(g.player_id, g.minutes, g.points, g.rebounds, g.assists))

    feat = {}
    for team, gids in order.items():
        cum_min, cum_prod = defaultdict(float), defaultdict(float)
        prev_played = None
        for gid in gids:
            rows = by_team_game.get((gid, team))
            m_share = p_share = NO_HISTORY_DEFAULT
            if prev_played is not None and cum_min:
                top = sorted(cum_min, key=cum_min.get, reverse=True)[:ROTATION_SIZE]
                absent = _absent(top, prev_played, (report_by_game or {}).get(gid))
                tot = sum(cum_min[x] for x in top)
                if tot:
                    m_share = sum(cum_min[x] for x in absent) / tot
                totp = sum(cum_prod[x] for x in top)
                if totp:
                    p_share = sum(cum_prod[x] for x in absent) / totp
            feat[(gid, team)] = (m_share, p_share)  # recorded BEFORE this game is folded in
            if rows is not None:
                prev_played = {pid for pid, mn, *_ in rows if mn > 0}
                for pid, mn, pts, reb, ast in rows:
                    cum_min[pid] += mn
                    cum_prod[pid] += pts + 1.2 * reb + 1.5 * ast

    d = (NO_HISTORY_DEFAULT, NO_HISTORY_DEFAULT)
    out = games[["game_id"]].copy()
    for side in ("home", "away"):
        teams = games[f"{side}_team"]
        out[f"{side}_absent_min_share"] = [feat.get((g, t), d)[0] for g, t in zip(games.game_id, teams)]
        out[f"{side}_absent_prod_share"] = [feat.get((g, t), d)[1] for g, t in zip(games.game_id, teams)]
    return out[["game_id", *AVAILABILITY_COLUMNS]]


def _season_of(game_date: str) -> int:
    """NBA season (start year) of an ISO date: October on is the new season."""
    year, month = int(game_date[:4]), int(game_date[5:7])
    return year if month >= 10 else year - 1


def compute_availability_by_season(games: pd.DataFrame, players: pd.DataFrame | None,
                                   report_by_game: dict | None = None) -> pd.DataFrame:
    """`compute_availability_features`, with each season ranked over its own production-sized window.

    The plain function ranks the rotation by cumulative minutes over WHATEVER frame it is given. Production only ever
    feeds it about a season (the ingest window starts at the previous season's Oct 1), where that is fine. Over a longer
    frame it silently degrades: measured over 2018-2025 the mean absent-minutes share is 0.49, against 0.34 over the
    production window, because players who left years ago stay in the top 8 and can never have "played last game".
    So a longer training frame must not change what the feature means: each season's games are scored from the previous
    season's Oct 1 on, exactly what production sees. On a frame that already starts there this is identical to the plain
    function (pinned by a test), so nothing changes today.
    """
    seasons = games["game_date"].map(_season_of)
    parts = []
    for season in sorted(seasons.unique()):
        lo, hi = f"{season - 1}-10-01", f"{season + 1}-09-30"
        in_window = (games["game_date"] >= lo) & (games["game_date"] <= hi)
        p = None if players is None or not len(players) else players[(players["game_date"] >= lo) & (players["game_date"] <= hi)]
        feat = compute_availability_features(games[in_window], p, report_by_game)
        parts.append(feat[feat["game_id"].isin(games.loc[seasons == season, "game_id"])])
    out = pd.concat(parts, ignore_index=True)
    return out.set_index("game_id").loc[games["game_id"]].reset_index()[["game_id", *AVAILABILITY_COLUMNS]]

