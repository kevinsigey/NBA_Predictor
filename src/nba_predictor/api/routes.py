import json
import logging
import os
import sqlite3
import threading
from contextvars import ContextVar
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from nba_predictor.api.availability import for_teams, resolve_doubtful_players, resolve_out_players
from nba_predictor.models.probability import cover_prob, over_prob
from nba_predictor.models.manifest import build_manifest
from nba_predictor.api.deps import (
    get_db_path,
    get_injury_report,
    get_roster_cache,
    get_roster_index,
    get_injury_report_best_effort,
    get_models_dir,
    get_schedule,
    get_schedule_path,
    get_today,
    get_training_games_path,
    require_admin,
)
from nba_predictor.api import facts as facts_mod
from nba_predictor.services.schedule_repository import load_schedule
from nba_predictor.api.schemas import (
    DoubtfulPlayerOut,
    GameDetailOut,
    GameOut,
    KeyOutOut,
    HeadToHeadMeetingOut,
    MarketPredictionOut,
    OutPlayerOut,
    PlayerPropOut,
    PredictionOut,
    SeasonBoundsOut,
    TeamOut,
    TrackRecordOut,
    TrackRecordPicksOut,
    VsMarketOut,
)
from nba_predictor.models.player_props import in_sample_mae_by_stat
from nba_predictor.data.team_reference import TEAMS, get_team
from nba_predictor.odds.value_bets import std_from_mae
from nba_predictor.pipeline.ingest import run_ingest
from nba_predictor.odds import edge_gate
from nba_predictor.odds.edge_gate import _is_fresh
from nba_predictor.api.key_out import key_out_for_game
from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.services.calibration_service import compute_model_calibration
from nba_predictor.services.hub_service import (
    compute_track_record,
    compute_track_record_picks,
    compute_vs_market,
    load_hub_cache,
    player_name_map,
    warm_player_props_cache,
)
from nba_predictor.services.schedule_repository import (
    default_week_start,
    get_game,
    get_games_for_date,
    get_games_for_week,
    get_head_to_head,
    get_recent_form,
    load_schedule,
)
from nba_predictor.tracking import store
from nba_predictor import config
from nba_predictor.tracking.store import get_recent_market_predictions
from nba_predictor.tracking.player_props import (
    actuals_by_player_stat,
    picks_by_player_stat,
    resolved_player_props,
)
from nba_predictor.tracking.timing import latest_by_instant, latest_pre_tip, made_before_tip
from nba_predictor import config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/teams", response_model=list[TeamOut])
def list_teams() -> list[TeamOut]:
    return [
        TeamOut(abbreviation=t.abbreviation, name=t.name, conference=t.conference, division=t.division)
        for t in TEAMS
    ]


@router.get("/teams/{abbreviation}", response_model=TeamOut)
def get_team_detail(abbreviation: str) -> TeamOut:
    try:
        team = get_team(abbreviation)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbreviation}")
    return TeamOut(abbreviation=team.abbreviation, name=team.name, conference=team.conference, division=team.division)


def _cover_and_sigma_from_manifest(manifest: dict | None) -> tuple[float | None, float | None, float | None]:
    """(margin_sigma, total_sigma, win_auc) from the manifest, or None if not present."""
    if not manifest:
        return None, None, None
    margin = manifest.get("metrics", {}).get("margin", {})
    total = manifest.get("metrics", {}).get("total", {})
    win = manifest.get("metrics", {}).get("win_probability", {})
    return margin.get("residual_sigma"), total.get("residual_sigma"), win.get("auc")


def _market_lines(db_path: Path, g: dict) -> tuple[float | None, float | None]:
    """(home spread point, total line) the books are quoting for this game, or None for each with no quote.

    The predictions table carries no lines (they live in game_market_predictions, written by the odds pass), so the API
    never had a line to price a cover or over chance against and showed none. Taken from the latest odds batch: the
    line most books agree on (the mode), the home point in the books' own sign (home -3.5 = home favoured by 3.5).
    """
    from collections import Counter

    rows = [r for r in _read_or([], store.get_market_predictions_for_game, db_path, g["game_id"]) if r["market"] in ("spread", "total") and r["point"] is not None]
    if not rows:
        return None, None
    latest = max(r["created_at"] for r in rows)
    rows = [r for r in rows if r["created_at"] == latest]
    totals = [float(r["point"]) for r in rows if r["market"] == "total" and r["selection"] == "over"]
    spreads = [float(r["point"]) for r in rows if r["market"] == "spread" and r["selection"] == g["home_team"]]
    mode = lambda xs: Counter(xs).most_common(1)[0][0] if xs else None
    return mode(spreads), mode(totals)


_READ_FAILED: ContextVar[bool] = ContextVar("tracking_read_failed", default=False)


def _read_or(default, fn, *args, **kwargs):
    """Run a read against the tracking DB, degrading instead of 500ing the page.

    The tracking DB is not the schedule: a prediction or a market line is an
    OPTIONAL field (`prediction: Prediction | null`, `markets: []`), and the page
    already renders without one. When the file is unreadable -- corrupted, locked
    by a writer, missing -- the truthful answer is the game with no pick, not a
    500 that reads to the visitor as "the whole site is down".

    On 2026-10-26 a mid-commit I/O error left the DB malformed and every one of
    these reads raised `database disk image is malformed`, so `/games/week`
    failed for every week and the page showed "We couldn't load this week's
    games. Check your connection and try again." A data problem was rendered as
    a network problem, and no games were shown at all.

    Writers do NOT go through this: a failed write must stay a failure, because
    a silent one is a lost prediction rather than a missing one.
    """
    if _READ_FAILED.get():
        return default
    try:
        with store.bounded_reads():
            return fn(*args, **kwargs)
    except sqlite3.DatabaseError as exc:
        logger.warning("tracking DB read failed (%s): %s", getattr(fn, "__name__", fn), exc)
        # One failed read means the rest of this request would fail the same way, each waiting out the
        # lock timeout again (`/games/week` reads once per game): answer the rest with the default.
        _READ_FAILED.set(True)
        return default


def _prediction_out(db_path: Path, g: dict, manifest: dict | None = None) -> tuple[PredictionOut | None, bool]:
    """The pick to show and whether it was rebuilt after tip-off. The latest
    pick made before tip-off wins; a later backtest row only shows (labelled
    rebuilt) when no pre-tip pick exists."""
    rows = _read_or([], store.get_predictions_for_game, db_path, g["game_id"])
    if not rows:
        return None, False
    row = latest_pre_tip(rows, g)
    rebuilt = row is None
    row = row if row is not None else latest_by_instant(rows)
    margin_sigma, total_sigma, _ = _cover_and_sigma_from_manifest(manifest)
    home_point, total_point = _market_lines(db_path, g)
    return (
        PredictionOut(
            home_win_probability=row["home_win_prob"],
            predicted_margin=row["predicted_margin"],
            predicted_total=row["predicted_total"],
            market_point=home_point,
            market_total_point=total_point,
            cover_prob_spread=cover_prob(row["predicted_margin"], -home_point, margin_sigma) if home_point is not None and margin_sigma is not None else None,
            cover_prob_total=over_prob(row["predicted_total"], total_point, total_sigma) if total_point is not None and total_sigma is not None else None,
            margin_sigma=margin_sigma,
            total_sigma=total_sigma,
        ),
        rebuilt,
    )


def load_latest_manifest() -> dict | None:
    path = config.PROJECT_ROOT / "models" / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _prop_bands(models_dir: Path) -> dict[str, float] | None:
    """The props model's own holdout MAE per stat, or None when there is none.

    The band is what a projection has to miss by before it counts as wrong, so
    it is the model's measured error and it moves with each retrain. A
    made-up default would be a number nobody measured, and the post-game
    review would judge players against it -- so a missing manifest means no
    band, never a default.
    """
    path = Path(models_dir) / "player_props_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    bands: dict[str, float] = {}
    for stat, figures in (manifest.get("metrics") or {}).items():
        mae = (figures or {}).get("mae")
        if isinstance(mae, (int, float)) and not isinstance(mae, bool):
            bands[str(stat)] = float(mae)
    return bands or None


def _game_out(g: dict, db_path: Path) -> GameOut:
    manifest = load_latest_manifest()
    prediction, rebuilt = _prediction_out(db_path, g, manifest)
    return GameOut(
        game_id=g["game_id"], game_date=g["game_date"], tip_off=g.get("tip_off"),
        home_team=g["home_team"], away_team=g["away_team"],
        prediction=prediction, rebuilt=rebuilt,
        completed=g.get("completed", False), home_pts=g.get("home_pts"), away_pts=g.get("away_pts"),
    )


@router.get("/games", response_model=list[GameOut])
def list_games(date: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)) -> list[GameOut]:
    games = get_games_for_date(schedule, date)
    return [_game_out(g, db_path) for g in games]


@router.get("/games/week", response_model=list[GameOut])
def list_games_for_week(
    start: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path),
    rosters: dict[str, list[dict]] | None = Depends(get_roster_cache),
) -> list[GameOut]:
    games = get_games_for_week(schedule, start)
    minutes = {p["player_id"]: p.get("minutes_per_game", 0.0)
               for p in load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")} if rosters else {}
    out = []
    for g in games:
        item = _game_out(g, db_path)
        item.key_out = [KeyOutOut(**k) for k in key_out_for_game(g, rosters, minutes)]
        out.append(item)
    return out


@router.get("/games/{game_id}", response_model=GameDetailOut)
def get_game_detail(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report_best_effort),
    models_dir: Path = Depends(get_models_dir),
) -> GameDetailOut:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    markets = [
        MarketPredictionOut(
            market=row["market"], selection=row["selection"], model_probability=row["model_probability"],
            market_probability=row["market_probability"], edge=row["edge"], bookmaker=row["bookmaker"],
            american_odds=row["american_odds"], point=row["point"],
            rebuilt=not made_before_tip(row["created_at"], game),
        )
        for row in _read_or([], store.get_market_predictions_for_game, db_path, game_id)
    ]

    head_to_head = [
        HeadToHeadMeetingOut(
            game_id=g["game_id"], game_date=g["game_date"], home_team=g["home_team"], away_team=g["away_team"],
            home_pts=g.get("home_pts"), away_pts=g.get("away_pts"),
        )
        for g in get_head_to_head(schedule, game["home_team"], game["away_team"], before_date=game["game_date"])
    ]

    base = _game_out(game, db_path)

    # Injury summary: the availability gate's own sentence
    picks = _read_or({}, picks_by_player_stat, db_path, game_id, game)
    picks_unreadable = _READ_FAILED.get()
    out_entries = resolve_out_players(injuries, {key[0] for key in picks})
    doubtful_entries = resolve_doubtful_players(injuries, {key[0] for key in picks})
    # This game's two teams only: the report is league-wide, and a player who
    # is playing for neither of them is not this game's news.
    out_entries = for_teams(out_entries, {game["home_team"], game["away_team"]})
    doubtful_entries = for_teams(doubtful_entries, {game["home_team"], game["away_team"]})
    # Without the picks there is nothing to check availability against: say nothing rather than "0 Out".
    injury_summary = None if picks_unreadable else f"Checked against ESPN availability — {len(out_entries)} Out, {len(doubtful_entries)} Day-to-Day"

    return GameDetailOut(
        **base.model_dump(),
        markets=markets,
        head_to_head=head_to_head,
        home_recent_form=get_recent_form(schedule, game["home_team"], before_date=game["game_date"]),
        away_recent_form=get_recent_form(schedule, game["away_team"], before_date=game["game_date"]),
        injury_summary=injury_summary,
        prop_bands=_prop_bands(models_dir),
        ratings=_game_ratings(game),
    )


def _game_ratings(game: dict) -> dict | None:
    """These two teams' ratings and league ranks, or None when there are none.

    Read from the same warmed box history the duels read, as of the game's own
    date: a rating is the form going INTO the game, so a game that has been
    played is not allowed to borrow the form that came after it.

    When the window was not warmed (a finished game older than the warmer's few
    kept dates), the ratings are computed on demand from box scores in the cache
    volume and cached on disk keyed by the game. A cold game
    shows no table rather than blocking the request, and a game with no usable
    history shows no table, never a made-up one.
    """
    try:
        if game.get("completed"):
            return facts_mod.ratings_for_game(game)
        return facts_mod.ratings_payload(str(game["game_date"]), [game["home_team"], game["away_team"]])
    except Exception:  # noqa: BLE001 - a rating is an enhancement, never the page
        logger.exception("ratings unavailable for %s", game.get("game_id"))
        return None


# The MAE is a property of the tracking database and the schedule, not of the request:
# computing it ran two queries per scheduled game (about 3,500 for a full season) on
# EVERY /games/{id}/players call. It changes only when a game resolves, i.e. when the
# database file changes, so it is computed once per (file state, schedule size).
#
# Computed, it is not free: about a second of bulk reads over the resolved rows, so
# charging it to whichever visitor arrives first after the database changed is the
# whole of the cost this cache leaves behind. Two things remove that charge from the
# request path, and both have to hold:
#
#   * the entry is WARMED at startup and re-warmed on a database change
#     (start_mae_warmer / note_database_changed below), so the first request after
#     either already finds the value;
#   * the entry has an explicit age bound, because the file-state key is a
#     best-effort signal and a cache that can never expire is how the availability
#     gate ended up reading one frozen injury report on every request (NBA#18). Past
#     the bound the value is recomputed, and if the recompute FAILS the request gets
#     the failure -- an expired number is never served quietly.
_MAE_CACHE: dict[tuple, tuple[float, dict[str, float | None]]] = {}

# Fifteen minutes, the same bound as the injury report. The database state key
# invalidates this cache far more often than that in practice (every ingest, every
# refresh-odds); the TTL is the backstop for a change the key somehow misses, so it
# is short and never unbounded. Env-overridable like INJURY_CACHE_TTL_SECONDS.
MAE_CACHE_TTL_SECONDS = int(os.getenv("MAE_CACHE_TTL_SECONDS", "900"))

# How often the background warmer checks whether the database moved. Cheap: one
# stat() per tick, and a bulk read only when the state differs from the last warm.
# This is the backstop for a write made outside this process (a mounted volume
# updated elsewhere, a restore); the write paths this process owns re-warm
# themselves via note_database_changed, so the window is only ever this long.
MAE_WARM_POLL_SECONDS = float(os.getenv("MAE_WARM_POLL_SECONDS", "10"))


def _db_state(db_path: Path) -> tuple[int, int]:
    try:
        st = Path(db_path).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (-1, -1)


# One bulk read at a time. A visitor arriving while the startup warm is still
# running used to start a SECOND identical read, and the two then serialised on
# the tracking store's global lock -- measurably worse than either alone (2.96 s
# for the first local request against 1.09 s cold and 1.0 s warm, tests/
# test_api_mae_warm.py). Taking this lock and re-checking the cache under it means
# the late request waits for the warm it was about to duplicate and reuses it.
# The wait is bounded by MAE_COMPUTE_WAIT_SECONDS, after which the request
# computes for itself rather than waiting on a hung warm: serving must never
# block indefinitely on the cache.
_MAE_COMPUTE_LOCK = threading.Lock()
MAE_COMPUTE_WAIT_SECONDS = float(os.getenv("MAE_COMPUTE_WAIT_SECONDS", "30"))


def _drop_mae_entries(db_path: Path) -> None:
    """Every entry for one database. One entry per database is kept: an older
    state of the same file can never be asked for again, so holding it would only
    grow the dict -- and holding it past a failed refresh is worse than growing."""
    prefix = str(db_path)
    for stale in [k for k in _MAE_CACHE if k[0] == prefix]:
        del _MAE_CACHE[stale]


def _mae_record(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> dict[str, dict]:
    """The props error estimate per stat, as BOTH figures, in one pass.

    ``{"all": {stat: mae}, "pre_tip": {stat: mae}, "n_all": {...}, "n_pre_tip": {...}}``

    `all` is the headline: every COUNTED pick, one per (game, player, stat) --
    the earliest recorded, whenever it was made (predictor-hub #66,
    2026-10-01). `pre_tip` is the same summariser over the subset whose own
    timestamps prove they were made before tip-off, which is the honest read of
    what the model would have said on the night.

    Both come from ONE `resolved_player_props` call, not two: they are two
    frames of the same rows, filtered by the `made_before_tip` each row
    already carries, so they cannot disagree about which rows exist and the
    per-database-state cache stays worth having.

    ``now`` is a clock, injected so the age bound is testable without sleeping.
    A recompute that fails propagates, and drops the entry on the way out: a
    request must never be handed a value it already knows is past its bound.
    """
    key = (str(db_path), *_db_state(db_path), len(schedule))
    cached = _MAE_CACHE.get(key)
    if cached is not None and now() - cached[0] < MAE_CACHE_TTL_SECONDS:
        return cached[1]

    # Wait for an in-flight compute (the startup warm, most likely) rather than
    # duplicating it, then re-check: the entry that compute produced is the one we
    # wanted. Bounded, so a warm that hangs costs this request its own compute
    # instead of its availability.
    waited = _MAE_COMPUTE_LOCK.acquire(timeout=MAE_COMPUTE_WAIT_SECONDS)
    try:
        key = (str(db_path), *_db_state(db_path), len(schedule))
        cached = _MAE_CACHE.get(key)
        if cached is not None and now() - cached[0] < MAE_CACHE_TTL_SECONDS:
            return cached[1]
        try:
            rows = resolved_player_props(db_path, schedule)
            pre_tip_rows = [r for r in rows if r["made_before_tip"]]
            result = {
                "all": in_sample_mae_by_stat(rows),
                "pre_tip": in_sample_mae_by_stat(pre_tip_rows),
                "n_all": _rows_per_stat(rows),
                "n_pre_tip": _rows_per_stat(pre_tip_rows),
            }
        except Exception:
            _drop_mae_entries(db_path)
            raise
        _drop_mae_entries(db_path)
        _MAE_CACHE[key] = (now(), result)
        return result
    finally:
        if waited:
            _MAE_COMPUTE_LOCK.release()


def _rows_per_stat(rows: list[dict]) -> dict[str, int]:
    """How many resolved rows each stat's MAE was computed from.

    An error estimate with no n beside it is a number nobody can weigh, and the
    pre-tip figure is the one most likely to be thin -- so the n travels with
    both, on the wire and in the provenance sentence the site prints.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stat"]] = counts.get(row["stat"], 0) + 1
    return counts


def _mae_by_stat(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> dict[str, float | None]:
    """The HEADLINE in-sample MAE per stat: every counted pick, whenever made.

    Kept under this name and this shape because it is what `PlayerPropOut.mae`
    carries and what every existing caller reads. What changed is the
    population behind it -- see `_mae_record`, which is where both figures and
    their n are built.
    """
    return _mae_record(db_path, schedule, now=now)["all"]


def warm_mae_cache(db_path: Path, schedule: list[dict], *, now=time.monotonic) -> bool:
    """Fill the MAE cache for the current database state, off the request path.

    Returns whether it succeeded and never raises: a warm that fails must not take
    the API down, and must not leave the previous database's number behind looking
    current. It logs the failure instead of swallowing it, because a warm that
    silently never succeeds is a cold path nobody is told about.
    """
    try:
        _mae_by_stat(db_path, schedule, now=now)
    except Exception as exc:  # noqa: BLE001 - reported, not raised: see docstring
        logger.error("MAE cache warm failed for %s: %s: %s", db_path, type(exc).__name__, exc)
        return False
    return True


def note_database_changed(db_path: Path, schedule: list[dict]) -> bool:
    """The write path's own re-warm. run_ingest is what writes predictions and
    outcomes, so it knows the MAE has changed; it says so here instead of waiting
    for the poller to notice. Reuses the existing (mtime, size) invalidation --
    no second notion of "the database changed" is introduced.

    The track-record player-props aggregate is re-warmed here too. Its own bound
    is a TTL rather than a database-state key, so this is a courtesy, not the
    invalidation mechanism: a write that lands between two TTL expiries would
    otherwise leave a five-minute-old aggregate on the page for the rest of its
    window. A warm that fails is reported and ignored here -- the TTL still
    expires and the next request recomputes."""
    warm_player_props_cache(db_path)
    return warm_mae_cache(db_path, schedule)


def start_mae_warmer(
    db_path: Path,
    schedule_path: Path,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    should_stop=None,
) -> threading.Thread:
    """Warm the MAE cache now, then re-warm it whenever the database changes.

    A daemon thread, started from the app's lifespan: serving begins whether or not
    the warm finishes, and a warm that hangs cannot hold shutdown either. ``clock``,
    ``sleep`` and ``should_stop`` are injected so the loop is testable without
    waiting on a real 30-second tick.

    The invalidation signal is _db_state -- (mtime_ns, size) of the tracking
    database, the key this cache already keys on. An unchanged database costs one
    stat() per tick and nothing else.
    """
    db_path, schedule_path = Path(db_path), Path(schedule_path)

    def _run() -> None:
        last_state: tuple[int, int] | None = None
        while should_stop is None or not should_stop():
            state = _db_state(db_path)
            if state != last_state:
                schedule = load_schedule(schedule_path)
                warm_mae_cache(db_path, schedule, now=clock)
                last_state = state
            sleep(MAE_WARM_POLL_SECONDS)

    thread = threading.Thread(target=_run, name="mae-warmer", daemon=True)
    thread.start()
    return thread


@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
    roster_index: dict[str, set[str]] | None = Depends(get_roster_index),
) -> list[PlayerPropOut]:
    """One projection per player and stat, with an out player left out.

    The out player's rows are ABSENT from this ranking rather than flagged in
    place; they are served once by /games/{game_id}/players/out, attributed
    and dated.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    name_by_id = player_name_map(db_path, config.DATA_DIR / "cache" / "hub" / "players.json")
    picks = picks_by_player_stat(db_path, game_id, game)
    actual_by_key = actuals_by_player_stat(db_path, game_id)

    # An UPCOMING game shows only players on the two teams' current rosters: older scoring runs assigned players from
    # box scores, so a traded player's stale rows (Jaylen Brown on BOS, opening week 2026) can still sit in the store.
    # A finished game keeps whoever played in it. Unknown rosters filter nothing.
    team_by_player: dict[str, str] = {}
    if roster_index is not None and not game.get("completed"):
        for team in (game.get("home_team"), game.get("away_team")):
            team_by_player.update({player_id: team for player_id in roster_index.get(team, set())})
        if team_by_player:
            picks = {key: value for key, value in picks.items() if key[0] in team_by_player}

    # A FINISHED game with recorded results shows the players who played: a projection for someone who never
    # appeared (scored from an old roster guess, or a rested star) has nothing to be judged against.
    if game.get("completed") and actual_by_key:
        played = {player_id for player_id, _stat in actual_by_key}
        picks = {key: value for key, value in picks.items() if key[0] in played}

    # Resolve availability before ranking anything. Anything unresolvable here
    # removes nobody -- see api/availability.py for why that asymmetry.
    out_ids = {entry["player_id"] for entry in resolve_out_players(injuries, {key[0] for key in picks})}

    mae_record = _mae_record(db_path, schedule)
    mae_by_stat = mae_record["all"]
    mae_pre_tip_by_stat = mae_record["pre_tip"]
    n_by_stat = mae_record["n_all"]
    n_pre_tip_by_stat = mae_record["n_pre_tip"]

    props = []
    for (player_id, stat), (pick, rebuilt) in picks.items():
        if player_id in out_ids:
            continue
        props.append(
            PlayerPropOut(
                player_id=player_id,
                player_name=name_by_id.get(player_id, player_id),
                team=team_by_player.get(player_id),
                stat=stat,
                predicted_value=pick["predicted_value"],
                actual_value=actual_by_key.get((player_id, stat)),
                rebuilt=rebuilt,
                mae=mae_by_stat.get(stat),
                mae_pre_tip=mae_pre_tip_by_stat.get(stat),
                mae_n=n_by_stat.get(stat, 0),
                mae_n_pre_tip=n_pre_tip_by_stat.get(stat, 0),
            )
        )
    return props


@router.get("/games/{game_id}/players/out", response_model=list[OutPlayerOut])
def get_game_out_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
) -> list[OutPlayerOut]:
    """Players removed from this game's ranking by the availability gate.

    One entry per player, whatever their number of stat rows, each carrying
    the feed it came from and the date on that feed.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    picks = picks_by_player_stat(db_path, game_id, game)
    entries = resolve_out_players(injuries, {key[0] for key in picks})
    # This game's two teams only: the report is league-wide, and a player who
    # is playing for neither of them is not this game's news.
    entries = for_teams(entries, {game["home_team"], game["away_team"]})

    # Prefer the roster's own name for the id, the same source the ranking
    # uses, so the out line and the ranking never disagree on who this is.
    name_by_id = player_name_map(db_path, config.DATA_DIR / "cache" / "hub" / "players.json")
    return [
        OutPlayerOut(**{**entry, "player_name": name_by_id.get(entry["player_id"], entry["player_name"])})
        for entry in entries
    ]


@router.get("/games/{game_id}/players/doubtful", response_model=list[DoubtfulPlayerOut])
def get_game_doubtful_players(
    game_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report),
) -> list[DoubtfulPlayerOut]:
    """Players the gate flagged as doubtful, each STILL RANKED.

    A separate feed from /players/out on purpose, not duplication. /out is a
    removal instruction: TopCalls.tsx builds its out-id set from every row it is
    handed and filters the ranking by it. Putting doubtful players there would
    make the shipped frontend delete all 52 of them, which is the defect this
    feed exists to close rather than a rendering of it. The frontend joins this
    feed to the ranking on player_id -- an exact key, for the same reason the
    gate is -- and renders a subordinate note beside the pick. Nobody is removed.

    Like the out feed it depends on the real availability dependency, so a feed
    we cannot read refuses with 503 instead of serving an empty list. An empty
    list here means "the report was read and names nobody doubtful", which is a
    different fact from "nobody checked" and must never look alike.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    picks = picks_by_player_stat(db_path, game_id, game)
    entries = resolve_doubtful_players(injuries, {key[0] for key in picks})
    # This game's two teams only, for the reason the out feed does it.
    entries = for_teams(entries, {game["home_team"], game["away_team"]})

    # The roster's own name for the id, the same source the ranking uses, so
    # the note and the row it attaches to never disagree on who this is.
    name_by_id = player_name_map(db_path, config.DATA_DIR / "cache" / "hub" / "players.json")
    return [
        DoubtfulPlayerOut(**{**entry, "player_name": name_by_id.get(entry["player_id"], entry["player_name"])})
        for entry in entries
    ]


@router.get("/hub/teams")
def hub_teams() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "teams.json")


@router.get("/hub/players")
def hub_players() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")


@router.get("/hub/players/next-games")
def hub_players_next_games(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    today: str = Depends(get_today),
) -> dict:
    """Every player's next game, with the model's projection for it.

    The table's last column. A projection that is not there is null, never
    zero: a player with no stored projection has no projection.
    """
    from ..api.player_details import next_game_projections

    players = load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")
    team_of = {str(p.get("player_id")): str(p.get("team") or "") for p in players if p.get("player_id")}
    return next_game_projections(schedule, db_path, team_of, today)


@router.get("/hub/players/{player_id}")
def hub_player_detail(
    player_id: str,
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    injuries: list[dict] = Depends(get_injury_report_best_effort),
    today: str = Depends(get_today),
) -> dict:
    """One player: their last ten games, their status, their next game."""
    from ..api.player_details import next_game_projections, player_detail

    players = load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")
    row = next((p for p in players if str(p.get("player_id")) == str(player_id)), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown player: {player_id}")
    detail = player_detail(
        schedule, db_path, str(player_id), str(row.get("team") or ""), injuries, today
    )
    detail["next_game"] = next_game_projections(
        schedule, db_path, {str(player_id): str(row.get("team") or "")}, today
    ).get(str(player_id))
    detail["averages"] = {
        "points_per_game": row.get("points_per_game"),
        "rebounds_per_game": row.get("rebounds_per_game"),
        "assists_per_game": row.get("assists_per_game"),
        "fg_pct": row.get("fg_pct"),
        "three_pt_pct": row.get("three_pt_pct"),
        "ft_pct": row.get("ft_pct"),
        "minutes_per_game": row.get("minutes_per_game"),
        "rating": row.get("rating"),
    }
    return detail


@router.get("/hub/rankings")
def hub_rankings() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "rankings.json")


@router.get("/hub/standings")
def hub_standings(
    season: str | None = None,
    schedule: list[dict] = Depends(get_schedule),
) -> list[dict]:
    if season == "previous":
        return previous_season_standings(schedule)
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "standings.json")


def previous_season_standings(schedule: list[dict]) -> list[dict]:
    """Last season's standings, computed from the schedule cache.

    Reuses `compute_standings` (no second standings function). The schedule
    cache carries completed flags and scores but not box scores, and
    `compute_standings` now counts a game when it is completed and scored --
    standings need only W/L, box scores are a ratings/form requirement.
    Preseason is excluded via `is_preseason`.
    """
    from nba_predictor.pipeline.ingest import (
        compute_standings,
        is_preseason,
        _current_season_year,
        _game_date,
        _season_start_year,
    )

    current = _current_season_year(schedule)
    if current is None:
        return []
    previous = current - 1

    previous_games = [
        g
        for g in schedule
        if _game_date(g) is not None
        and _season_start_year(_game_date(g)) == previous
        and not is_preseason(g)
        and g.get("home_pts") is not None
        and g.get("away_pts") is not None
    ]
    return compute_standings(previous_games)


@router.get("/hub/team-overview")
def hub_team_overview(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    today: str = Depends(get_today),
) -> dict:
    """Everything the Teams page needs, in one payload.

    The league's ratings (the same window the game detail reads), every team's
    recent form, its last five games and its next game with the model's pick.
    One endpoint so the page does not join four feeds and hope they agree.
    """
    from ..api import facts as facts_mod
    from ..api.team_overview import overview
    from ..data.team_reference import TEAMS

    teams = [t.abbreviation for t in TEAMS]
    try:
        ratings = facts_mod.ratings_payload(today, teams)
    except Exception:  # noqa: BLE001 - the ratings are an enhancement, not the page
        logger.exception("ratings unavailable for the teams page")
        ratings = None
    return overview(schedule, db_path, ratings, teams, today)


@router.get("/hub/track-record", response_model=list[TrackRecordOut])
def hub_track_record(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> list[TrackRecordOut]:
    """The record's aggregates, plus each market's latest counted pre-tip
    picks (`recent_picks`). The per-pick disclosure is NOT inlined: it is
    tens of thousands of rows (~14 MB) and is paged from
    /hub/track-record/picks when a reader opens the table."""
    return compute_track_record(db_path, schedule, include_picks=False)


@router.get("/hub/track-record/picks", response_model=TrackRecordPicksOut)
def hub_track_record_picks(
    market: str | None = None,
    game_id: list[str] = Query(default=[]),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule),
) -> TrackRecordPicksOut:
    """One page of the every-recorded-pick disclosure, newest stamp first.

    `market` narrows to one market; repeat `game_id` to narrow to specific
    games (the odds-comparison block asks for exactly the games it lists).
    `total` counts the whole filtered set, so a paging reader knows when
    they have seen all of it."""
    return compute_track_record_picks(
        db_path, schedule, market=market, game_ids=game_id or None,
        limit=limit, offset=offset,
    )


@router.get("/hub/vs-market", response_model=VsMarketOut)
def hub_vs_market(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> VsMarketOut:
    """Model against the price on the moneyline: headline, weekly rows over
    the same window as /hub/track-record, the scope reconciliation, and the
    method sentences the page prints verbatim."""
    return compute_vs_market(db_path, schedule)


@router.get("/value-picks")
def value_picks(
    db_path: Path = Depends(get_db_path),
    max_age_minutes: int = 60,
    threshold: float = edge_gate.EDGE_THRESHOLD,
    ceiling: float = edge_gate.EDGE_CEILING,
) -> dict:
    """The gated picks: at most one single per game, 5% edge, fresh odds only.

    A measurement surface, not a recommendation. `picks` is frequently empty and
    that is a real answer -- a stale slate or a day with no edge both produce
    zero, and the thresholds are returned alongside so a reader can see what was
    applied instead of taking it on trust.

    The disclaimer is PL's framing, deliberately: a positive yield on a graded
    ledger is not evidence that the gate found an edge. Nothing here places a
    bet.
    """
    # `.isoformat()`, not a datetime object. `created_at` is TEXT holding an
    # ISO-8601 string, and SQLite compares it as text -- but sqlite3 renders a
    # datetime argument with a SPACE separator ("2026-10-06 13:16:09+00:00")
    # while these rows use "T". " " sorts before "T", so every row then looked
    # newer than `since` and the freshness filter was silently a no-op: a
    # two-hour-old line would sail through as fresh.
    since = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    rows = [dict(r) for r in get_recent_market_predictions(db_path, since=since)]

    # Count suspect rows: those that are otherwise eligible (fresh, have line, not parlay)
    # but have edge > ceiling
    suspect_rows = []
    for row in rows:
        if row.get("market") == "parlay":
            continue
        point = row.get("point")
        if row.get("market") in ("spread", "total") and point is None:
            continue
        edge = row.get("edge")
        if edge is None:
            continue
        if not _is_fresh(row, datetime.now(timezone.utc)):
            continue
        if not edge_gate._has_counterpart(row, rows):
            continue
        if edge > ceiling:
            suspect_rows.append(row)

    picks = edge_gate.gated_picks(rows, threshold=threshold, ceiling=ceiling)
    return {
        "picks": picks,
        "edge_threshold": threshold,
        "edge_ceiling": ceiling,
        "n_suspect": len(suspect_rows),
        "max_odds_age_minutes": max_age_minutes,
        "n_odds_rows_considered": len(rows),
        "disclaimer": (
            "A selection threshold, not advice. Historical hit rate and yield on "
            "this ledger are measurements of past games, not evidence of a "
            "profitable strategy, and nothing here places a bet."
        ),
    }


@router.get("/picks/history")
def picks_history(since: str = "", db_path: Path = Depends(get_db_path)) -> dict:
    """Return pick history rows keyed by (game_id, created_at).

    The `since` parameter filters to rows stamped at or after the given ISO-8601
    timestamp. If omitted, all rows are returned. Rows carry game_id, created_at,
    model_version, home_win_prob, predicted_margin, predicted_total, game_date,
    home_team, away_team, and total_sigma from the manifest.
    """
    import json

    from nba_predictor import config

    # The manifest's residual sigmas, for the total and spread markets. These
    # are read here rather than at module scope because the manifest is a FILE
    # that is rewritten by every retrain, and the route has to see the current
    # one rather than the one that was there when the process started.
    from nba_predictor.public_snapshot import (
        _manifest_margin_sigma,
        _manifest_total_sigma,
    )
    from nba_predictor.services.schedule_repository import load_schedule

    with store.get_connection(db_path) as conn:
        # The parameter list is built from the filter, so a query with NO
        # filter has NO parameters. Passing an empty tuple as the second
        # argument to execute() is a ProgrammingError, not a no-op.
        params: tuple = (since,) if since else ()
        cur = conn.execute(
            """
            SELECT p.game_id, p.created_at, p.model_version, p.home_win_prob,
                   p.predicted_margin, p.predicted_total
            FROM predictions p
            """
            + ("WHERE p.created_at >= ? " if since else "")
            + "ORDER BY p.created_at DESC, p.id DESC",
            params,
        )
        all_rows = cur.fetchall()

    if not all_rows:
        return {"rows": [], "note": "no pick rows in tracking DB"}

    schedule = load_schedule(get_schedule_path())
    manifest_path = config.PROJECT_ROOT / "models" / "manifest.json"
    total_sigma: float | None = None
    margin_sigma: float | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            total_sigma = _manifest_total_sigma(manifest)  # type: ignore[arg-type]
            margin_sigma = _manifest_margin_sigma(manifest)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            total_sigma = None
            margin_sigma = None

    by_game = {g["game_id"]: g for g in schedule}

    rows_out: list[dict] = []
    for row in all_rows:
        game_id = str(row[0])
        game = by_game.get(game_id)
        # A pick stamped after its game's own date is a rebuilt backtest of a game already played (the daily scoring
        # run re-scores every game in the window): it cannot be a pre-game call and would be ~90% of the feed.
        if game is not None and str(game.get("game_date") or "9999") < str(row[1])[:10]:
            continue
        rows_out.append(
            {
                "game_id": game_id,
                "created_at": str(row[1]),
                "model_version": str(row[2]) if row[2] is not None else "",
                "home_win_prob": float(row[3]) if row[3] is not None else None,
                "predicted_margin": float(row[4]) if row[4] is not None else None,
                "predicted_total": float(row[5]) if row[5] is not None else None,
                "game_date": game.get("game_date") if game else None,
                "home_team": game.get("home_team") if game else None,
                "away_team": game.get("away_team") if game else None,
                "total_sigma": total_sigma,
                # The margin's own sigma, so a spread rung can be priced at
                # its own line. Absent rather than zero when the manifest
                # publishes none.
                **({"margin_sigma": margin_sigma} if margin_sigma is not None else {}),
            }
        )

    return {"rows": rows_out}


@router.get("/snapshot-meta")
def get_snapshot_meta() -> dict:
    """When the data this site serves was last written, as an ISO-8601 UTC string.

    NBA is the odd one out: the routes above read the tracking DB, the schedule
    cache, and the hub caches, never public_snapshot.json, so there is no
    generated_at to report verbatim. The honest equivalent is the newest mtime
    across exactly the files those routes actually read -- the tracking DB
    (predictions, market rows, outcomes), the schedule cache (games.json, read
    by nearly every route including /games/week, which is what the hub's NBA
    teaser calls), and data/cache/hub/*.json (teams/players/rankings/
    standings, plus the player-name map) -- which is when the numbers the
    visitor is looking at were produced. Reporting the snapshot file's
    timestamp instead would describe numbers nobody is served, so that was
    ruled out deliberately; do not "fix" this to read the snapshot.

    "source" keeps the vocabulary shared with the other four sport APIs
    ("public_snapshot" when files exist, "live" otherwise) because the hub
    switches on it -- even though here the value is a filesystem mtime, not a
    snapshot stamp. Every path missing means "nothing written yet", reported
    as source "live" with a null timestamp rather than raising: a public
    deploy before its first refresh is a real state, not a failure. The
    exists() guard also covers a TRACKING_DB_PATH (env-overridable) pointing
    at a path with no local file.
    """
    # Built from config.DATA_DIR, not config.CACHE_DIR, deliberately: every
    # hub route above spells the path as config.DATA_DIR / "cache" / "hub" /
    # <name>.json, and CACHE_DIR is frozen at import from the original
    # DATA_DIR, so only the DATA_DIR form keeps this set exactly the files
    # those routes read when DATA_DIR is redirected (tests, PROJECT_ROOT env).
    # Identical to CACHE_DIR / "hub" in any normal deployment. The schedule
    # cache comes from the get_schedule_path() helper itself -- the same
    # function get_schedule() depends on -- rather than a re-spelled literal,
    # so the two cannot drift. data/cache/training/games.json is deliberately
    # excluded: only the admin retrain POST reads it (guarded on exists()),
    # no public GET serves it, so it is not part of "these numbers".
    hub_dir = config.DATA_DIR / "cache" / "hub"
    paths = [
        config.TRACKING_DB_PATH,
        get_schedule_path(),
        *sorted(hub_dir.glob("*.json")),
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if not mtimes:
        return {"generated_at": None, "source": "live"}
    newest = max(mtimes)
    return {
        "generated_at": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat(),
        "source": "public_snapshot",
    }


@router.get("/calibration")
def calibration(
    db_path: Path = Depends(get_db_path), schedule: list[dict] = Depends(get_schedule)
) -> list[dict]:
    return compute_model_calibration(db_path, schedule)


@router.get("/manifest")
def get_manifest(models_dir: Path = Depends(get_models_dir)) -> dict:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest found — run /retrain first")
    return json.loads(manifest_path.read_text())


@router.get("/player-props-manifest")
def get_player_props_manifest(models_dir: Path = Depends(get_models_dir)) -> dict:
    manifest_path = models_dir / "player_props_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No player props manifest found — run /retrain first")
    return json.loads(manifest_path.read_text())


@router.post("/retrain", dependencies=[Depends(require_admin)])
def retrain(
    models_dir: Path = Depends(get_models_dir),
    training_games_path: Path = Depends(get_training_games_path),
) -> dict:
    if not training_games_path.exists():
        raise HTTPException(status_code=400, detail="No training games cache found")

    games = pd.DataFrame(json.loads(training_games_path.read_text()))
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()

    # The availability features need player box scores; ingest writes them next to games.json.
    players_path = training_games_path.with_name("players.json")
    if not players_path.exists():
        # Without it the four availability columns would be trained as constant zeros and then served with real values:
        # the fitted-on-a-constant defect this repo has already paid for once (power_rating / missing_value).
        raise HTTPException(
            status_code=409,
            detail="No player box-score cache (players.json) next to the training games; run the ingest first. "
                   "Refusing to train the availability features as zeros.",
        )
    players = pd.DataFrame(json.loads(players_path.read_text()))
    return run_retrain_pipeline(games, models_dir, model_version=model_version, trained_at=trained_at, players=players)


def _market_stds_from_manifest(models_dir: Path) -> tuple[float, float]:
    margin_std, total_std = 12.0, 15.0
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        metrics = json.loads(manifest_path.read_text()).get("metrics", {})
        margin_mae = metrics.get("margin", {}).get("mae")
        total_mae = metrics.get("total", {}).get("mae")
        if margin_mae is not None:
            margin_std = std_from_mae(margin_mae)
        if total_mae is not None:
            total_std = std_from_mae(total_mae)
    return margin_std, total_std


@router.post("/refresh-odds", dependencies=[Depends(require_admin)], status_code=202)
def refresh_odds(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
    margin_std, total_std = _market_stds_from_manifest(models_dir)
    stored = refresh_market_predictions(schedule, db_path, margin_std=margin_std, total_std=total_std)
    # This wrote to the tracking database, which is what invalidates the cached
    # MAE -- even though these market rows are not what the MAE is computed from.
    # Re-warm here so the next visitor is not charged for a change we made.
    note_database_changed(db_path, schedule)
    return {"status": "ok", "market_predictions_stored": stored}


@router.get("/season/first-week", response_model=SeasonBoundsOut)
def season_first_week(schedule: list[dict] = Depends(get_schedule), today: str = Depends(get_today)) -> SeasonBoundsOut:
    return SeasonBoundsOut(first_week_start=default_week_start(schedule, today))


_ingest_status: dict = {"running": False, "last_result": None, "last_error": None}


def _run_ingest_background(db_path: Path, schedule_path: Path, models_dir: Path) -> None:
    _ingest_status["running"] = True
    _ingest_status["last_error"] = None
    try:
        summary = run_ingest(
            "2025-10-01", "2026-11-30", player_hub_days=9999, db_path=db_path, log=lambda _msg: None
        )
        # Team win/margin/total predictions are scored by run_ingest above;
        # the odds-derived markets (spread/total lines, bookmaker, edge)
        # need a separate pass against the sportsbook API, keyed off the
        # schedule and model MAE that ingest just refreshed.
        schedule = load_schedule(schedule_path)
        margin_std, total_std = _market_stds_from_manifest(models_dir)
        summary["market_predictions_stored"] = refresh_market_predictions(
            schedule, db_path, margin_std=margin_std, total_std=total_std
        )
        # Both passes above wrote to the tracking database, which is what
        # invalidates the cached MAE, so this is the moment to replace it rather
        # than leave the next visitor to pay for it. warm_mae_cache never raises,
        # so this cannot turn a successful ingest into a failed one.
        note_database_changed(db_path, schedule)
        _ingest_status["last_result"] = summary
    except Exception as exc:  # noqa: BLE001 - reported via status endpoint, not re-raised (background task)
        _ingest_status["last_error"] = str(exc)
    finally:
        _ingest_status["running"] = False


@router.post("/admin/refresh-full", dependencies=[Depends(require_admin)], status_code=202)
def refresh_full(
    background_tasks: BackgroundTasks,
    db_path: Path = Depends(get_db_path),
    schedule_path: Path = Depends(get_schedule_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
    """Runs the full real-data pipeline (schedule/box-score fetch, model
    retraining, and scoring — including upcoming games and player props)
    against this server's own live tracking DB. For a deployment with no
    baked-in predictions and no interactive shell access — the same thing
    `python -m nba_predictor.pipeline.ingest` does locally, triggered over
    HTTP instead. Runs in the background; poll GET /admin/refresh-full/status
    for progress, since a full run can take a long time."""
    if _ingest_status["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_ingest_background, db_path, schedule_path, models_dir)
    return {"status": "started"}


@router.get("/admin/refresh-full/status", dependencies=[Depends(require_admin)])
def refresh_full_status() -> dict:
    return _ingest_status
