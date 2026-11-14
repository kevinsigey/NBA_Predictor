"""Keep upcoming player projections on the CURRENT roster.

Bug (live, 2026-10-10): the fixture detail showed one team's predictions and a
trickle of the other's. `/games/401909837` (CHI v TOR) had 28 predicted players,
of whom 2 played for CHI; BOS and WAS were the same. The CHI/BOS rows were
exactly the roster of the team's last pre-preseason game, and none of the
players from the preseason game after it had a projection.

Root cause: an upcoming game's player projections are a roster ESTIMATE, taken
from the team's latest completed game when `score_upcoming_player_props` last
ran -- and that only runs inside `run_ingest`, which in production is the manual
`POST /admin/refresh-full` (the daily CI uses `--skip-predictions`). Nothing
re-scored when a team played another game, so the estimate stayed frozen at the
roster of whatever game preceded the last manual refresh.

This re-scores the upcoming games inside the lead window whose estimate predates a
completed game of either team (or is `REFRESH_HOURS` old), then goes quiet.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from nba_predictor.tracking import store
from nba_predictor.tracking.timing import pick_cutoff

logger = logging.getLogger(__name__)

#: A game gets a projection only once tip-off is this close. The track record counts the EARLIEST row per
#: (game, player, stat), so a row written days ahead is the counted pick, frozen on a roster estimate and a rolling
#: window that miss every game played since. Replaying the live path on 2026-10-28/31 and 11-02 slates: points MAE
#: 4.9-5.8 for rows written 9-14 days ahead against 4.5-5.1 for rows written the day before (live served 5.55 against
#: 4.67 for the same model re-scored on settled data). So nothing is recorded early.
LEAD_HOURS = 30
#: Inside the lead window the projection is re-scored when its newest row is this old, so the shown pick (the newest
#: pre-tip row) tracks injuries and lineup news up to tip-off. Reruns are history; they never replace the counted pick.
REFRESH_HOURS = 6
#: Box-score history behind the rolling features. The full-season ingest uses a
#: whole season; this reaches back over the summer to the last regular-season
#: games so a player's rolling average is not three preseason minutes.
HISTORY_DAYS = 240
#: A game is "complete" this long after tip-off (stored `tip_off` is the start).
GAME_LENGTH = timedelta(hours=3)
TICK_SECONDS = 1800


def _completed_at(game: dict) -> pd.Timestamp:
    tip = game.get("tip_off")
    start = pd.Timestamp(tip) if tip else pd.Timestamp(f"{game['game_date']}T23:59:59Z")
    return start + GAME_LENGTH


def stale_upcoming_games(schedule: list[dict], db_path: Path, now: pd.Timestamp) -> list[dict]:
    """Upcoming games whose newest projection predates a completed game of either team."""
    team_last: dict[str, pd.Timestamp] = {}
    for g in schedule:
        if g.get("completed"):
            done = _completed_at(g)
            for side in ("home_team", "away_team"):
                if g.get(side) and done > team_last.get(g[side], done - timedelta(days=1)):
                    team_last[g[side]] = done
    with store.get_connection(db_path) as conn:
        newest = {
            r[0]: pd.Timestamp(r[1])
            for r in conn.execute(
                "SELECT game_id, MAX(created_at) FROM player_prediction_snapshots GROUP BY game_id"
            )
        }
    out = []
    for g in schedule:
        if g.get("completed"):
            continue
        tip = pd.Timestamp(pick_cutoff(g))
        if not now < tip <= now + timedelta(hours=LEAD_HOURS):
            continue
        last = newest.get(g["game_id"])
        latest_game = max((team_last[t] for t in (g["home_team"], g["away_team"]) if t in team_last), default=None)
        if (
            last is None
            or now - last >= timedelta(hours=REFRESH_HOURS)
            or (latest_game is not None and latest_game > last)
        ):
            out.append(g)
    return out


def refresh_stale_player_props(
    schedule: list[dict], db_path: Path, models_dir: Path, *, fetch=None, score=None, now=None, rosters=None, history=None
) -> int:
    """Re-score the stale upcoming games. Returns rows stored (0 when nothing was stale)."""
    from nba_predictor.data.rosters import get_rosters
    from nba_predictor.pipeline.history import load_history
    from nba_predictor.pipeline.ingest import fetch_player_boxscores, record_names_from_boxscores, score_upcoming_player_props

    fetch = fetch or fetch_player_boxscores
    score = score or score_upcoming_player_props
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    stale = stale_upcoming_games(schedule, db_path, now)
    if not stale:
        return 0
    since = (now - timedelta(days=HISTORY_DAYS)).date().isoformat()
    completed = [g for g in schedule if g.get("completed") and g["game_date"] >= since]
    boxes = fetch(completed)
    record_names_from_boxscores(boxes, db_path)
    # Same inputs as the full ingest: the frozen earlier seasons behind the rolling window, and the current rosters.
    history = history if history is not None else load_history("2100-01-01")[1]
    rosters = rosters if rosters is not None else get_rosters()
    return score(completed + stale, boxes, models_dir, db_path, model_version="roster-refresh",
                 rosters=rosters, extra_completed=history)


def rebuild_player_names(schedule: list[dict], db_path: Path, *, fetch=None, now=None) -> int:
    """Re-learn names from the recent box scores. Run on every boot: the
    container disk (ESPN cache, and the DB if not on a volume) may be fresh."""
    from nba_predictor.pipeline.ingest import fetch_player_boxscores

    from nba_predictor.pipeline.ingest import record_names_from_boxscores

    fetch = fetch or fetch_player_boxscores
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    since = (now - timedelta(days=HISTORY_DAYS)).date().isoformat()
    completed = [g for g in schedule if g.get("completed") and g["game_date"] >= since]
    return record_names_from_boxscores(fetch(completed), db_path)


def start_player_props_refresher(db_path: Path, schedule_path: Path, models_dir: Path, *, tick=TICK_SECONDS) -> threading.Thread:
    from nba_predictor.services.schedule_repository import load_schedule

    def _run() -> None:
        try:
            rebuild_player_names(load_schedule(schedule_path), db_path)
        except Exception:  # noqa: BLE001 - names fall back to the id; never stop the refresher
            logger.exception("player-name rebuild failed")
        while True:
            try:
                n = refresh_stale_player_props(load_schedule(schedule_path), db_path, models_dir)
                if n:
                    logger.info("re-scored %d player projections on the current roster", n)
            except Exception:  # noqa: BLE001 - a failed pass keeps the previous projections
                logger.exception("player-props roster refresh failed")
            time.sleep(tick)

    thread = threading.Thread(target=_run, name="player-props-refresher", daemon=True)
    thread.start()
    return thread
