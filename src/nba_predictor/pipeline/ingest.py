import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

from nba_predictor import config
from nba_predictor.data import espn
from nba_predictor.data.team_reference import get_team
from nba_predictor.features.build import build_feature_frame, build_training_frame
from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.features.context import current_streak
from nba_predictor.features.ratings import compute_possessions
from nba_predictor.models.game_outcome import predict_win_probability
from nba_predictor.models.manifest import build_manifest, write_manifest
from nba_predictor.models.probability import win_prob
from nba_predictor.models.player_props import predict_player_stat, train_player_stat_model
from nba_predictor.data.rosters import get_rosters
from nba_predictor.features.early_season import shrunk_projections
from nba_predictor.features.prop_model_frame import build_prop_model_frame
from nba_predictor.models.prop_holdout import MARKETS, prop_holdout_metrics
from nba_predictor.pipeline.history import merge_history
from nba_predictor.pipeline.retrain import manifest_total_drift, run_retrain_pipeline
from nba_predictor.tracking import store

BOX_FIELDS = ["fgm", "fga", "fg3m", "tov", "oreb", "dreb", "fta"]
# Carried when ESPN's team box has it (3P% needs attempts); never required, so a game
# without it still trains and the 3P% matchup is simply skipped.
OPTIONAL_BOX_FIELDS = ["fg3a"]


def default_window_start(today: date | None = None) -> str:
    """Oct 1 of the season **before the one in progress**.

    The window used to be `date -d '60 days ago'`, which cannot cover a season
    and left the offseason run training on one game before crashing in
    chronological_split -- see docs/nba-retrain-diagnosis-2026-10.md. The rolling
    features in features/build.py need a season of history behind them, so the
    floor is one full season back whatever day it is.

    Anchored to the season, not to the calendar year. Oct-Dec belongs to the
    season starting that year; Jan-Sep to the season that started the year
    before -- the same rule `pipeline/retrain.py::_nba_season_start_year` uses.
    So the *previous* season's Oct 1 is:

        2026-10-05 -> 2025-10-01   (offseason: the season just finished)
        2027-02-03 -> 2025-10-01   (mid-season: last season plus this one so far)
        2027-10-20 -> 2026-10-01   (early season)

    An earlier version returned `today.year - 1` unconditionally, which for the
    nine months from January to September was the *current* season's start --
    four months of history, not the full season the docstring promised.
    """
    today = today or date.today()
    season_year = today.year if today.month >= 10 else today.year - 1
    return f"{season_year - 1}-10-01"


def default_window_end(today: date | None = None) -> str:
    """Fourteen days out: enough of the schedule to score every upcoming game
    the same horizon the workflow always wanted."""
    return ((today or date.today()) + timedelta(days=14)).isoformat()


def _daterange(start_date: str, end_date: str) -> list[str]:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    days = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(days + 1)]


def _parse_made_attempted(value: str) -> tuple[float, float]:
    try:
        made, attempted = value.split("-")
        return float(made), float(attempted)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def store_player_outcomes(training_df: pd.DataFrame, db_path: Path, log=print) -> int:
    """Stores the real actual stat value for every completed player-game
    row, for each of the four tracked stat targets — settlement data.

    Returns the number of facts that were NEW. Re-presenting a fact already on
    the record is not an error and not a row either: the scheduled refresh
    walks a rolling window of finished games on every run, so this re-presents
    most of what is already stored and the count is "what today's run added",
    which is the honest reading of the summary key
    `player_outcomes_stored`.

    A fact that DISAGREES with what is already recorded -- a source that revised
    a box score, say -- is counted and logged by key rather than written.
    Recorded outcomes are immutable, and `store.ConflictingOutcome` is raised
    instead of silently overwriting or silently ignoring it. It does not abort
    the ingest: one revised box score should not stop the other three stats and
    every other player in the window from being recorded.
    """
    recorded_at = datetime.now(timezone.utc).isoformat()
    before = _outcome_row_count(db_path)
    conflicts: list[str] = []
    for _, row in training_df.iterrows():
        for stat, column in PLAYER_STAT_TARGET_COLUMNS.items():
            try:
                store.insert_player_outcome(
                    db_path,
                    game_id=row["game_id"],
                    player_id=row["player_id"],
                    stat=stat,
                    actual_value=float(row[column]),
                    recorded_at=recorded_at,
                )
            except store.ConflictingOutcome as exc:
                conflicts.append(str(exc))
    if conflicts:
        log(
            f"  WARNING: {len(conflicts)} player outcome(s) disagreed with a value "
            f"already recorded and were NOT written (recorded stays recorded). "
            f"First: {conflicts[0]}"
        )
    return _outcome_row_count(db_path) - before


def _outcome_row_count(db_path: Path) -> int:
    """How many outcome rows exist, so the summary can report facts ADDED.

    Counted rather than incremented per call, because re-presenting a fact
    already on the record returns successfully and writes nothing: incrementing
    would report a day's refresh as thousands of new rows when it added none.
    """
    with store.get_connection(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM game_player_outcomes").fetchone()[0]


PLAYER_STAT_TARGET_COLUMNS = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "fg3m"}

#: ESPN's season type for the preseason. One constant, two shapes: the frame
#: rule (`regular_and_post_only`) and the game-dict rule (`is_preseason`).
PRESEASON_SEASON_TYPE = 1


def is_preseason(game: dict) -> bool:
    """True for an ESPN preseason game (season type 1).

    The same rule `regular_and_post_only` applies to a frame, for the dict
    shape the schedule cache and the hub caches carry. A game with no season
    type at all -- a schedule cache written before the field existed -- is
    "unknown", not "preseason": refusing to read it would empty every window
    until the next ingest.
    """
    return game.get("season_type") == PRESEASON_SEASON_TYPE


def regular_and_post_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drops preseason rows (ESPN season type 1). A star's preseason minutes are a handful and say little about
    his regular-season output: with them in the last-10 window, opening-week projections were ~half the truth."""
    if "season_type" not in df.columns:
        return df
    return df[df["season_type"] != PRESEASON_SEASON_TYPE].reset_index(drop=True)


def to_player_scoring_frame(
    games: list[dict], player_boxscores: dict[str, list[dict]], roster_window: int = 5,
    rosters: dict[str, list[dict]] | None = None, extra_completed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate upcoming roster from recent completed games; rows for
    upcoming games have None stats (causal — never touches a future result).

    A team's estimated roster is the players who appeared for it in any of
    its last `roster_window` completed games (by date, not list order) AND
    whose latest appearance anywhere is still for that team. The previous
    version took only the team's single last box score, in list order:

    * a star who sat that one game (rest, a minor knock) vanished from the
      roster entirely;
    * a traded player stayed on his OLD team's roster -- its last box score
      in the window predated the trade -- while also joining his new team's,
      so he was projected in two games on the same day, one of them a game
      his team was not playing in (live 2026-10-20: Jaylen Brown in both
      DET-BOS and NYK-PHI after BOS->PHI).

    Keying membership to the player's LATEST team makes a wrong-team row
    impossible by construction, and the (player, date) dedupe below makes
    two projections for one player on one day impossible even if the
    schedule itself lists his team twice that day.
    """
    completed_rows = regular_and_post_only(to_player_training_frame(games, player_boxscores))
    completed_rows = completed_rows[completed_rows["minutes"] > 0]       # rolling means over games he played, as in training
    if extra_completed is not None and len(extra_completed):
        # Earlier seasons (the frozen history): a player's rolling average must not start over each October.
        completed_rows = pd.concat([extra_completed, completed_rows], ignore_index=True).drop_duplicates(["player_id", "game_id"], keep="last")

    rows = []
    if len(completed_rows):
        completed_rows = completed_rows.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
        upcoming = sorted(
            (g for g in games if not g.get("completed")),
            key=lambda g: (g["game_date"], g["game_id"]),
        )
        if rosters is not None:
            # The CURRENT roster (ESPN) is the authority on who plays for whom: box scores only know where a player
            # last appeared. A reported-Out player gets no projection; everyone else on the roster does (history
            # follows the player_id across teams in build_player_feature_frame).
            def roster_for(team):
                return [(p["player_id"], p["player_name"], p.get("position") or None) for p in rosters.get(team, []) if p.get("injury_status") != "Out"]
        else:
            # No roster feed: the latest-team rule over each team's last `roster_window` games (by date).
            latest_team = completed_rows.groupby("player_id")["team"].last()
            team_game_ids: dict[str, list[str]] = {}
            for team, team_rows in completed_rows.groupby("team"):
                dated = team_rows.drop_duplicates(subset=["game_id"]).sort_values(["game_date", "game_id"], kind="mergesort")
                team_game_ids[team] = list(dated["game_id"])[-roster_window:]
            box_roster: dict[str, dict[str, pd.Series]] = {}
            for _, row in completed_rows.iterrows():
                team = row["team"]
                if row["game_id"] in team_game_ids.get(team, []) and latest_team.get(row["player_id"]) == team:
                    box_roster.setdefault(team, {})[row["player_id"]] = row

            def roster_for(team):
                return [(pid, r["player_name"], r.get("position")) for pid, r in box_roster.get(team, {}).items()]

        for g in upcoming:
            for side in ["home_team", "away_team"]:
                team = g.get(side)
                for player_id, player_name, position in roster_for(team):
                    rows.append({
                        "player_id": player_id,
                        "player_name": player_name,
                        "team": team,
                        "game_id": g["game_id"],
                        "game_date": g["game_date"],
                        "points": None,
                        "rebounds": None,
                        "assists": None,
                        "fg3m": None,
                        "minutes": None,
                        "position": position,
                        "fga": None,
                        "home_team": g.get("home_team"),
                        "away_team": g.get("away_team"),
                        "opponent": g.get("away_team") if side == "home_team" else g.get("home_team"),
                    })
    upcoming_rows = pd.DataFrame(rows)
    if len(upcoming_rows):
        # At most one projection per player per day, by construction.
        upcoming_rows = upcoming_rows.drop_duplicates(subset=["player_id", "game_date"], keep="first")
    # Combine, drop duplicates for upcoming (keep first per player/game)
    combined = pd.concat([completed_rows, upcoming_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id", "game_id"], keep="first").reset_index(drop=True)
    return combined


#: A served projection must stay within this band around the player's own
#: rolling average for that stat. The band is wide -- a real projection can
#: legitimately sit well off the average on matchup or minutes grounds --
#: and exists to catch a model whose output has collapsed (or exploded)
#: relative to the very features it was fed, which is what the Sep-18
#: vintage prop models do: fed a 25.0 points_roll they answer ~5.
PROJECTION_GUARD_LOW = 0.4
PROJECTION_GUARD_HIGH = 2.5
#: Below this rolling average a ratio band is noise (a non-shooter's threes), so the guard has nothing to say.
GUARD_MIN_AVERAGE = 0.25
#: A clamp is a quality signal, not a fix: when more than this share of a run's projections needed one, the model is
#: wrong and publishing a pile of "0.4 x average" as if it were modelled would be worse than publishing nothing.
MAX_CLAMP_SHARE = 0.10
# The trip-wire judges rotation players (rolling minutes at or above this): a broken model misses them by a lot, while
# a deep-bench player with two prior games has a rolling average of noise and trips the band on any model.
ROTATION_MINUTES = 15.0


def guard_player_prediction(stat: str, predicted: float, rolling_average: float | None) -> float:
    """Clamp one projection into the band around the player's own rolling
    average for that stat, logging whenever the clamp fires.

    Without a rolling average there is nothing to judge the number against,
    so it passes through; `build_player_feature_frame` drops such rows
    before scoring anyway. A negative prediction is impossible for a
    counting stat and is clamped by the same lower bound.
    """
    if rolling_average is None or pd.isna(rolling_average) or float(rolling_average) < GUARD_MIN_AVERAGE:
        return predicted
    low = PROJECTION_GUARD_LOW * float(rolling_average)
    high = PROJECTION_GUARD_HIGH * float(rolling_average)
    guarded = min(max(float(predicted), low), high)
    if guarded != float(predicted):
        logger.warning(
            "player projection guard: %s prediction %.2f outside [%.2f, %.2f] "
            "(rolling average %.2f); serving %.2f",
            stat, predicted, low, high, rolling_average, guarded,
        )
    return guarded


def score_upcoming_player_props(
    games: list[dict], player_boxscores: dict[str, list[dict]], models_dir: Path, db_path: Path, model_version: str,
    rosters: dict[str, list[dict]] | None = None, extra_completed: pd.DataFrame | None = None,
) -> int:
    """Score upcoming player props: current roster (ESPN) x causal rolling features.

    Fails closed: if more than MAX_CLAMP_SHARE of the projections fall outside the sanity band around the player's own
    rolling average, nothing is stored and the reason is logged. Returns the number of projections stored.
    """
    scoring_df = to_player_scoring_frame(games, player_boxscores, rosters=rosters, extra_completed=extra_completed)
    frame, feature_cols = build_prop_model_frame(scoring_df)
    upcoming_frame = frame[frame["points"].isna()].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    models: dict[str, object] = {}
    for stat in PLAYER_STAT_TARGET_COLUMNS:
        path = models_dir / f"player_{stat}_model.pkl"
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            logger.warning("player model missing, skipping market: %s", path)
            continue
        n_fitted = getattr(model, "n_features_in_", len(feature_cols))
        if n_fitted != len(feature_cols):
            # A model fitted before the feature set changed: skip it until the next retrain replaces it.
            logger.warning("player model %s was fitted on %d features, the frame has %d: skipping %s until the next retrain", path.name, n_fitted, len(feature_cols), stat)
            continue
        models[stat] = model
    predictions = {stat: predict_player_stat(model, upcoming_frame[feature_cols]) for stat, model in models.items()}
    # Early in a player's season the model's rolling window is still last season; lean on last season's average instead.
    early = shrunk_projections(
        scoring_df[scoring_df["points"].notna()], upcoming_frame[["game_id", "player_id", "game_date"]], {s: PLAYER_STAT_TARGET_COLUMNS[s] for s in models},
    )

    guarded: list[tuple] = []
    clamped = rot_clamped = rot_total = 0
    for i, row in upcoming_frame.iterrows():
        rotation = float(row.get("minutes_trend", 0) or 0) >= ROTATION_MINUTES
        for stat in models:
            raw = early.get((row["game_id"], row["player_id"], stat), float(predictions[stat][i]))
            value = guard_player_prediction(stat, raw, row.get(f"{PLAYER_STAT_TARGET_COLUMNS[stat]}_roll"))
            clamped += value != raw
            rot_total += rotation
            rot_clamped += rotation and value != raw
            guarded.append((row, stat, value))
    if guarded and rot_total and rot_clamped / rot_total > MAX_CLAMP_SHARE:
        logger.error(
            "player props NOT published: %d of %d rotation-player projections (%.0f%%; %d of %d overall) fell outside the sanity "
            "band around the player's own rolling average; the player models are not trustworthy, retrain them",
            rot_clamped, rot_total, 100 * rot_clamped / rot_total, clamped, len(guarded),
        )
        return 0
    logger.info("player prop guard: %d of %d rotation projections and %d of %d overall clamped", rot_clamped, rot_total, clamped, len(guarded))

    created_at = datetime.now(timezone.utc).isoformat()
    store.insert_player_predictions(db_path, [
        {"game_id": row["game_id"], "player_id": row["player_id"], "stat": stat, "predicted_value": value,
         "created_at": created_at, "position": row.get("position")}
        for row, stat, value in guarded
    ])
    return len(guarded)


def train_player_prop_models(training_df: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    """Trains one XGBRegressor per stat on real per-player rolling features, over games the player PLAYED.

    Reports a chronological HOLDOUT MAE per stat next to the player's own rolling-mean baseline (and the in-sample
    figure for the gap), then refits on every row for serving. The earlier version trained on a ~3-week slice and
    reported only in-sample numbers: fed a 25-point scorer it answered ~5 (mean prediction 5.3 vs actual 10.7 over
    2021-2025 slices of that size, nba-eval/props_race.py). Trained on full history the model matches the baseline
    (points 4.70 vs 4.69, rebounds 1.95 vs 1.95); the metrics now say so instead of flattering it.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    played = regular_and_post_only(training_df[training_df["minutes"] > 0])
    frame, feature_cols = build_prop_model_frame(played)

    holdout = prop_holdout_metrics(played, {m: train_player_stat_model for m in MARKETS}, frame_builder=build_prop_model_frame) if len(frame) >= 50 else {}
    metrics: dict[str, dict] = {}
    for stat, target_col in PLAYER_STAT_TARGET_COLUMNS.items():
        if len(frame) == 0:
            metrics[stat] = {"mae": None}
            continue
        model = train_player_stat_model(frame[feature_cols], frame[target_col])
        joblib.dump(model, models_dir / f"player_{stat}_model.pkl")
        in_sample = float(np.mean(np.abs(predict_player_stat(model, frame[feature_cols]) - frame[target_col])))
        h = holdout.get(stat, {})
        metrics[stat] = {
            "mae": h.get("holdout_mae", in_sample),          # the headline is the holdout figure when there is one
            "naive_mae": h.get("naive_mae"),
            "in_sample_mae": in_sample,
            # The conformal cold start, from the same holdout: published so
            # the served 80% intervals have a measured calibration before
            # the live counted record builds up (None until a holdout
            # exists -- no parameters are invented for a model that has
            # never been held out).
            "conformal": h.get("conformal"),
        }

    manifest = build_manifest(
        model_names=list(PLAYER_STAT_TARGET_COLUMNS.keys()),
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
        training={"n_train_player_games": int(len(frame)), "in_sample_metrics": not holdout},
    )
    write_manifest(manifest, models_dir / "player_props_manifest.json")
    return manifest


def score_and_store_player_predictions(training_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str) -> int:
    """Backtest: scores every completed player-game row that survived
    feature assembly with the trained per-stat models, stores each as a
    tracked player prediction."""
    frame, feature_cols = build_prop_model_frame(training_df)
    if len(frame) == 0:
        return 0

    models: dict[str, object] = {}
    for stat in PLAYER_STAT_TARGET_COLUMNS:
        path = models_dir / f"player_{stat}_model.pkl"
        try:
            models[stat] = joblib.load(path)
        except FileNotFoundError:
            logger.warning("player model missing, skipping market: %s", path)
    predictions = {stat: predict_player_stat(model, frame[feature_cols]) for stat, model in models.items()}

    created_at = datetime.now(timezone.utc).isoformat()
    rows = [
        dict(game_id=row["game_id"], player_id=row["player_id"], stat=stat, predicted_value=float(predictions[stat][i]),
             created_at=created_at, position=row.get("position"))
        for i, row in frame.iterrows()
        for stat in models
    ]
    return store.insert_player_predictions(db_path, rows)


def fetch_schedule_range(start_date: str, end_date: str) -> list[dict]:
    """All games (completed and upcoming) for each date in the range, via ESPN."""
    games = []
    for day in _daterange(start_date, end_date):
        games.extend(espn.get_scoreboard(day))
    return games


def _is_missing_event(err: requests.HTTPError) -> bool:
    return err.response is not None and err.response.status_code == 404


def enrich_with_boxscores(games: list[dict]) -> list[dict]:
    """Adds flat home_*/away_* box-score fields to completed games with a full box score."""
    enriched = []
    for game in games:
        game = dict(game)
        if game["completed"]:
            try:
                box = espn.get_boxscore(game["game_id"])
            except requests.HTTPError as err:
                if not _is_missing_event(err):
                    raise
                # ESPN has no summary for this event (postponed/voided): leave it un-enriched. to_training_frame
                # already skips completed games without box fields, so one dead event must not stop the refresh.
                logging.getLogger(__name__).warning("no box score for event %s (404); skipping", game["game_id"])
                enriched.append(game)
                continue
            home_box, away_box = box.get(game["home_team"]), box.get(game["away_team"])
            if home_box and away_box and all(k in home_box for k in BOX_FIELDS) and all(k in away_box for k in BOX_FIELDS):
                for field in BOX_FIELDS:
                    game[f"home_{field}"] = home_box[field]
                    game[f"away_{field}"] = away_box[field]
                for field in OPTIONAL_BOX_FIELDS:
                    if field in home_box and field in away_box:
                        game[f"home_{field}"] = home_box[field]
                        game[f"away_{field}"] = away_box[field]
        enriched.append(game)
    return enriched


def to_schedule_cache(games: list[dict]) -> list[dict]:
    """Schedule repository shape, including final scores/completion status
    so settlement (track record, calibration) can determine actual outcomes
    without a separate DB table."""
    return [
        {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "completed": g.get("completed", False),
            "home_pts": g.get("home_pts"),
            "away_pts": g.get("away_pts"),
            # ESPN's season type: 1 is the preseason, which is not a game in the
            # race and not a team's form. The window that ranks teams needs to
            # know, and the schedule cache is where it reads from.
            "season_type": g.get("season_type"),
            # The pre-tip cutoff (tracking/timing.py) reads this.
            "tip_off": g.get("tip_off"),
        }
        for g in games
    ]


def to_training_frame(games: list[dict]) -> pd.DataFrame:
    """Completed games with full box scores, shaped for features.build.build_training_frame."""
    rows = []
    for g in games:
        if not g.get("completed") or f"home_{BOX_FIELDS[0]}" not in g:
            continue
        row = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_pts": g["home_pts"],
            "away_pts": g["away_pts"],
            "home_win": int(g["home_pts"] > g["away_pts"]),
        }
        for field in BOX_FIELDS:
            row[f"home_{field}"] = g[f"home_{field}"]
            row[f"away_{field}"] = g[f"away_{field}"]
        for field in OPTIONAL_BOX_FIELDS:
            if f"home_{field}" in g:
                row[f"home_{field}"] = g[f"home_{field}"]
                row[f"away_{field}"] = g[f"away_{field}"]
        rows.append(row)
    return pd.DataFrame(rows)


def to_scoring_frame(games: list[dict]) -> pd.DataFrame:
    """All games (completed + upcoming), shaped for
    features.build.build_feature_frame. Upcoming games (or completed games
    missing a full box score) get None for home_win and every box-score
    field — build_feature_frame computes their rolling features from real
    prior games only, never touching a nonexistent result."""
    rows = []
    for g in games:
        row = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
        }
        if g.get("completed") and f"home_{BOX_FIELDS[0]}" in g:
            row["home_win"] = int(g["home_pts"] > g["away_pts"])
            row["home_pts"] = g["home_pts"]
            row["away_pts"] = g["away_pts"]
            for field in BOX_FIELDS:
                row[f"home_{field}"] = g[f"home_{field}"]
                row[f"away_{field}"] = g[f"away_{field}"]
        else:
            row["home_win"] = None
            row["home_pts"] = None
            row["away_pts"] = None
            for field in BOX_FIELDS:
                row[f"home_{field}"] = None
                row[f"away_{field}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def score_upcoming_games(
    games: list[dict], models_dir: Path, db_path: Path, model_version: str, players: pd.DataFrame | None = None,
) -> int:
    """Scores every not-yet-completed game using real prior-game rolling
    features (via to_scoring_frame + build_feature_frame) and stores the
    result in the same predictions table score_and_store_predictions
    writes to. Only rows for games that are not completed are scored —
    a completed game already gets its backtest prediction from
    score_and_store_predictions.

    Win probability is now derived from margin/sigma (Phi(margin/sigma)) for
    coherence with spread/total cover probabilities. The old classifier is
    not used; if the manifest lacks margin_sigma, we fail loudly rather than
    silently falling back."""
    scoring_df = to_scoring_frame(games)
    frame, feature_cols = build_feature_frame(scoring_df, players=players)
    upcoming_frame = frame[frame["home_win"].isna()].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    # Trusted artifacts: these .pkl files are written by run_retrain_pipeline
    # (via joblib.dump) in this same pipeline run — not from an external or
    # user-uploaded source (same trust boundary as score_and_store_predictions).
    try:
        win_model = joblib.load(models_dir / "win_probability_model.pkl")
        margin_model = joblib.load(models_dir / "margin_model.pkl")
        total_model = joblib.load(models_dir / "total_model.pkl")
    except FileNotFoundError as exc:
        logger.warning("game model missing, skipping game scoring: %s", exc.filename)
        return 0

# Load margin_sigma from manifest (populated by retrain from walk-forward residuals)
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}; run /retrain first")
    manifest = json.loads(manifest_path.read_text())
    margin_sigma = manifest.get("metrics", {}).get("margin", {}).get("residual_sigma")
    if margin_sigma is None or not np.isfinite(margin_sigma) or margin_sigma <= 0:
        raise ValueError(f"manifest missing positive finite margin_sigma; cannot serve win-from-margin. Got: {margin_sigma!r}")

    margins = margin_model.predict(upcoming_frame[feature_cols])
    totals = total_model.predict(upcoming_frame[feature_cols]) + manifest_total_drift(manifest)

    created_at = datetime.now(timezone.utc).isoformat()
    store.insert_predictions(db_path, [
        dict(game_id=row["game_id"], created_at=created_at, model_version=model_version,
             home_win_prob=win_prob(float(margins[i]), margin_sigma),
             predicted_margin=float(margins[i]), predicted_total=float(totals[i]))
        for i, row in upcoming_frame.iterrows()
    ])
    return len(upcoming_frame)


def _completed_with_box(games: list[dict]) -> list[dict]:
    return [g for g in games if g.get("completed") and f"home_{BOX_FIELDS[0]}" in g]


def compute_team_hub(games: list[dict]) -> list[dict]:
    completed = sorted(_completed_with_box(games), key=lambda g: g["game_date"])
    stats: dict[str, dict] = {}

    for g in completed:
        for side, opp_side in [("home", "away"), ("away", "home")]:
            abbr = g[f"{side}_team"]
            pts, opp_pts = g[f"{side}_pts"], g[f"{opp_side}_pts"]
            s = stats.setdefault(abbr, {"pts": 0, "opp_pts": 0, "poss": 0.0, "games": 0, "results": []})
            s["games"] += 1
            s["pts"] += pts
            s["opp_pts"] += opp_pts
            s["poss"] += compute_possessions(g[f"{side}_fga"], g[f"{side}_fta"], g[f"{side}_oreb"], g[f"{side}_tov"])
            s["results"].append("W" if pts > opp_pts else "L")

    rows = []
    for abbr, s in stats.items():
        team = get_team(abbr)
        n = s["games"]
        rows.append(
            {
                "abbreviation": abbr,
                "conference": team.conference,
                "division": team.division,
                "wins": s["results"].count("W"),
                "losses": s["results"].count("L"),
                "points_per_game": round(s["pts"] / n, 1),
                "opp_points_per_game": round(s["opp_pts"] / n, 1),
                "net_rating": round((s["pts"] - s["opp_pts"]) / n, 1),
                "pace": round(s["poss"] / n, 1),
                "streak": current_streak(s["results"]),
            }
        )
    return rows


def compute_power_rankings(games: list[dict]) -> list[dict]:
    from nba_predictor.features.ratings import elo_expected, elo_update, init_elo_ratings

    completed = sorted(_completed_with_box(games), key=lambda g: g["game_date"])
    teams = sorted({g["home_team"] for g in completed} | {g["away_team"] for g in completed})
    ratings = init_elo_ratings(teams)
    history: dict[str, list[float]] = {t: [ratings[t]] for t in teams}

    for g in completed:
        home, away = g["home_team"], g["away_team"]
        expected_home = elo_expected(ratings[home], ratings[away], home_adjustment=50)
        actual_home = 1.0 if g["home_pts"] > g["away_pts"] else 0.0
        ratings[home] = elo_update(ratings[home], expected_home, actual_home)
        ratings[away] = elo_update(ratings[away], 1 - expected_home, 1 - actual_home)
        history[home].append(ratings[home])
        history[away].append(ratings[away])

    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    rows = []
    for rank, (team, rating) in enumerate(ranked, start=1):
        h = history[team]
        trend = "steady"
        if len(h) >= 2:
            trend = "up" if h[-1] > h[-2] else "down" if h[-1] < h[-2] else "steady"
        rows.append({"rank": rank, "abbreviation": team, "power_rating": round(rating, 1), "trend": trend})
    return rows


#: The phases a schedule can be in, in the order they occur. Off-season first so
#: a lookup by index reads as the calendar.
SEASON_STATES = ("offseason", "preseason", "regular", "postseason")

#: How stale the most recent completed game may be and still read as a season
#: that just ended. Sixty days is roughly the gap between the Finals and opening
#: night; past it, the next season simply has not loaded and calling it
#: "postseason" would be a claim about a race that is months away.
OFFSEASON_GAP_DAYS = 60


def _game_date(game: dict) -> date | None:
    """`game_date` as a `date`, or None when it is missing or unreadable.

    Unreadable dates are skipped rather than fatal: one malformed row in a
    schedule cache should cost that row, not the whole classification.
    """
    raw = game.get("game_date")
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None


def _season_start_year(game_date: date) -> int:
    """NBA season starting year: Oct-Dec belongs to the season starting that
    calendar year, Jan-Sep to the season that started the prior year."""
    return game_date.year if game_date.month >= 10 else game_date.year - 1


def _current_season_year(games: list[dict]) -> int | None:
    """The latest season start year present in the schedule."""
    years = []
    for g in games:
        d = _game_date(g)
        if d is not None:
            years.append(_season_start_year(d))
    return max(years) if years else None


def season_state(games: list[dict], today: date | None = None) -> str:
    """Which phase of the season this schedule is in, derived from the schedule.

    The schedule is the only thing here that knows whether there is a season to
    be in -- everything else in this repo can look like a live league while the
    league is shut, which is exactly how "clinched" ended up describing a table
    nobody is playing in.

    Four cases, from the two questions that matter: are there games scheduled on
    or after today, and are there any results at all?

    * no games scheduled on or after today, and no recent result -> `offseason`
    * scheduled ahead, nothing recent played -> `preseason` (the league is
      loaded, nobody has played)

    * a recent result and games ahead -> `regular`
    * a recent result, nothing ahead -> `postseason`

    "Recent" is the load-bearing word. The rolling ingest window holds last
    season's finished games all summer *alongside* the new season's upcoming
    schedule, so "there are results" and "there are games ahead" are both true
    in October. Reading that as a season in progress is exactly how six teams
    got labelled "clinched" about a league nobody had played in, so a result
    only counts toward the current season if it is within `OFFSEASON_GAP_DAYS`.

    That gap is the one thing the schedule cannot settle by shape alone -- a
    season that ended last month and one that ended in April are both "finished,
    nothing ahead". Rather than guess from a month number, it asks how long ago,
    which is a fact in the data.

    A game dated *today* with no score counts as scheduled. "Today" is ambiguous
    at the edges, and reading it the other way labels the entire league
    `preseason` on opening day.
    """
    today = today or date.today()
    upcoming = False
    newest_finished: date | None = None
    current_season_year = _current_season_year(games)
    for g in games:
        game_date = _game_date(g)
        if game_date is None:
            continue
        played = g.get("home_pts") is not None and g.get("away_pts") is not None
        if played:
            # Only count games from the current season toward "in season" status.
            # This prevents a completed preseason game (or prior-season game)
            # from activating the regular season.
            if current_season_year is not None and _season_start_year(game_date) == current_season_year:
                newest_finished = max(filter(None, (newest_finished, game_date)))
        if game_date >= today:
            upcoming = True

    # A result only counts as belonging to THIS season if it is recent enough.
    # The rolling ingest window holds last season's finished games all summer,
    # alongside the new season's upcoming schedule -- so "there are results" and
    # "there are games ahead" are both true in October, and treating that as a
    # season in progress is precisely how six teams got labelled "clinched"
    # about a league nobody had played in.
    in_season = (
        newest_finished is not None
        and (today - newest_finished).days <= OFFSEASON_GAP_DAYS
    )
    if not upcoming and not in_season:
        return "offseason"
    if in_season:
        return "regular" if upcoming else "postseason"
    return "preseason"


def compute_standings(games: list[dict], today: date | None = None) -> list[dict]:
    """Conference standings by win percentage.

    `playoff_status` is `None` whenever the schedule says there is no season in
    progress. During the off-season the completed-games window holds last
    season's results, and seeding on those produced confident labels about a
    season nobody is playing: six teams "clinched" because they led an empty
    table, the leader "eliminated" because it was the only team with a record.
    The seed ordering and the win percentages stay -- they are real arithmetic
    on real games -- but the label is dropped rather than softened, because the
    honest statement is that the race does not exist yet.

    When the season is active (regular/postseason), standings are computed from
    current-season games only. Prior-season results are excluded from the
    racing calculation but remain visible in the win/loss columns during
    offseason/preseason when the page explicitly labels them as prior-season.
    """
    completed = [
        g
        for g in games
        if g.get("completed") and g.get("home_pts") is not None and g.get("away_pts") is not None
    ]
    # Standings need a score, not box scores: the schedule cache carries
    # completed flags and points but not box fields, so a game is counted when
    # it is completed and scored. Box scores are required by the ratings and
    # form feeds (compute_team_hub), not by standings, which only reads
    # home_team / away_team / home_pts / away_pts.
    completed = [g for g in completed if not is_preseason(g)]
    state = season_state(games, today=today)
    racing = state in ("regular", "postseason")

    # Filter to current-season games only when the season is active.
    current_season_year = _current_season_year(games)
    if racing and current_season_year is not None:
        completed = [
            g for g in completed
            if _game_date(g) is not None and _season_start_year(_game_date(g)) == current_season_year
        ]

    records: dict[str, list[int]] = {}

    for g in completed:
        winner = g["home_team"] if g["home_pts"] > g["away_pts"] else g["away_team"]
        loser = g["away_team"] if winner == g["home_team"] else g["home_team"]
        records.setdefault(winner, [0, 0])[0] += 1
        records.setdefault(loser, [0, 0])[1] += 1

    standings = []
    for conference in ["East", "West"]:
        conf_teams = [t for t in records if get_team(t).conference == conference]
        conf_teams.sort(key=lambda t: -(records[t][0] / (records[t][0] + records[t][1])))
        if not conf_teams:
            continue
        leader_wins, leader_losses = records[conf_teams[0]]
        for seed, team in enumerate(conf_teams, start=1):
            wins, losses = records[team]
            win_pct = wins / (wins + losses) if (wins + losses) else 0.0
            games_back = ((leader_wins - wins) + (losses - leader_losses)) / 2
            status = (
                ("clinched" if seed <= 6 else "play-in" if seed <= 10 else "eliminated")
                if racing
                else None
            )
            standings.append(
                {
                    "conference": conference,
                    "seed": seed,
                    "abbreviation": team,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": round(win_pct, 3),
                    "games_back": round(games_back, 1),
                    "playoff_status": status,
                    "season_state": state,
                }
            )
    return standings


def fetch_player_boxscores(games: list[dict]) -> dict[str, list[dict]]:
    """game_id -> per-player box score rows, for completed games only.

    A second ESPN request per game (get_player_boxscore is cached
    separately from get_boxscore) — callers should bound `games` to a
    recent window rather than a full season to avoid doubling the whole
    backfill's request count.
    """
    result = {}
    for game in games:
        if game.get("completed"):
            try:
                result[game["game_id"]] = espn.get_player_boxscore(game["game_id"])
            except requests.HTTPError as err:
                if not _is_missing_event(err):
                    raise
                logging.getLogger(__name__).warning("no player box score for event %s (404); skipping", game["game_id"])
    return result


def record_names_from_boxscores(player_boxscores: dict[str, list[dict]], db_path: Path) -> int:
    """Persist every (player_id, displayName) in the box scores, hub player or not."""
    return store.record_player_names(db_path, [r for rows in player_boxscores.values() for r in rows])


def backfill_names_from_espn_cache(db_path: Path, cache_dir: Path | None = None) -> int:
    """One-off backfill: names from every cached player_boxscore_*.json on disk."""
    cache_dir = cache_dir or espn.ESPN_CACHE_DIR
    store.init_db(db_path)
    rows = [r for f in cache_dir.glob("player_boxscore_*.json") for r in json.loads(f.read_text()).get("rows", [])]
    return store.record_player_names(db_path, rows)


def to_player_training_frame(games: list[dict], player_boxscores: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per (player, completed game) with real box-score stats,
    shaped for features.player_stats.build_player_feature_frame."""
    games_by_id = {g["game_id"]: g for g in games}
    rows = []
    for game_id, boxscore_rows in player_boxscores.items():
        game = games_by_id.get(game_id)
        if game is None:
            continue
        for row in boxscore_rows:
            fg3m, _ = _parse_made_attempted(row["three_made_attempted"])
            fgm, fga = _parse_made_attempted(row.get("fg_made_attempted"))
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "game_id": game_id,
                    "game_date": game["game_date"],
                    "points": row["points"],
                    "rebounds": row["rebounds"],
                    "assists": row["assists"],
                    "fg3m": fg3m,
                    "minutes": row["minutes"],
                    "position": row.get("position"),
                    # Shot volume, carried so features/prop_matchup.py has a real
                    # usage rate to work with. ESPN sends `fg_made_attempted`
                    # ("10-18"); the frame was dropping it, which left
                    # usage_trend uncomputable from real data rather than absent.
                    "fgm": fgm,
                    "fga": fga,
                    # Who the player was playing against, and where. The
                    # opponent is what makes a matchup feature a matchup
                    # feature; without it opp_def_vs_pos has nothing to key on.
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "opponent": (game["away_team"] if row["team"] == game.get("home_team")
                                 else game.get("home_team")),
                    # ESPN season type (1 preseason): rolling features skip preseason minutes.
                    "season_type": game.get("season_type"),
                }
            )
    # With no rows (an off-season 21-day window) a bare DataFrame has no columns and the next stage died with
    # KeyError: 'player_id'. Keep the schema so an empty window is an empty frame, not a crash.
    return pd.DataFrame(rows, columns=None if rows else PLAYER_TRAINING_COLUMNS)


PLAYER_TRAINING_COLUMNS = [
    "player_id", "player_name", "team", "game_id", "game_date", "points", "rebounds", "assists", "fg3m", "minutes",
    "position", "fgm", "fga", "home_team", "away_team", "opponent", "season_type",
]


def compute_player_hub(games: list[dict], player_boxscores: dict[str, list[dict]], team_by_player: dict[str, str] | None = None) -> list[dict]:
    """Real per-player aggregates from real box scores.

    `rating`/`live_form_rating` are a simple game-score-style composite
    (PTS + 0.4*REB + 0.7*AST) over the full window vs. the last 5 games —
    a simplified, clearly-labeled formula, not the league's own advanced
    metric. `usage_rate` is each game's (player FGA + 0.44*player FTA) /
    team FGA, averaged — a field-goal-attempt-share proxy, not the full
    NBA usage% formula (which also needs team possessions/minutes-on-court
    context this data doesn't carry per-player).

    `team_by_player` ({ESPN athlete id: team}) is the current roster: it overrides the team of the player's last box
    score, which is wrong after a trade until he has played for his new team.
    """
    team_by_player = team_by_player or {}
    # The preseason is not a player's form: a handful of preseason minutes said
    # little about his regular-season output and were in these averages.
    # Filtering the games drops their box scores too -- a box score whose game
    # is not in `games_by_id` is skipped below.
    games_by_id = {g["game_id"]: g for g in games if not is_preseason(g)}
    per_player_games: dict[str, list[dict]] = {}

    ordered_game_ids = sorted(player_boxscores, key=lambda gid: games_by_id.get(gid, {}).get("game_date", ""))
    for game_id in ordered_game_ids:
        game = games_by_id.get(game_id)
        if game is None:
            continue
        for row in player_boxscores[game_id]:
            team_fga = game["home_fga"] if row["team"] == game["home_team"] else game.get("away_fga")
            fgm, fga = _parse_made_attempted(row["fg_made_attempted"])
            fg3m, fg3a = _parse_made_attempted(row["three_made_attempted"])
            ftm, fta = _parse_made_attempted(row["ft_made_attempted"])
            per_player_games.setdefault(row["player_id"], []).append(
                {
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "position": row["position"],
                    "minutes": row["minutes"],
                    "points": row["points"],
                    "rebounds": row["rebounds"],
                    "assists": row["assists"],
                    "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
                    "usage_share": (fga + 0.44 * fta) / team_fga if team_fga else 0.0,
                }
            )

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _game_score(pts: float, reb: float, ast: float) -> float:
        return pts + 0.4 * reb + 0.7 * ast

    rows = []
    for player_id, entries in per_player_games.items():
        total_fgm = sum(e["fgm"] for e in entries)
        total_fga = sum(e["fga"] for e in entries)
        total_fg3m = sum(e["fg3m"] for e in entries)
        total_fg3a = sum(e["fg3a"] for e in entries)
        total_ftm = sum(e["ftm"] for e in entries)
        total_fta = sum(e["fta"] for e in entries)
        last5 = entries[-5:]

        rows.append(
            {
                "player_id": player_id,
                "player_name": entries[-1]["player_name"],
                "team": team_by_player.get(player_id, entries[-1]["team"]),
                "position": entries[-1]["position"],
                "rating": round(
                    _game_score(
                        _avg([e["points"] for e in entries]),
                        _avg([e["rebounds"] for e in entries]),
                        _avg([e["assists"] for e in entries]),
                    ),
                    1,
                ),
                "live_form_rating": round(
                    _game_score(
                        _avg([e["points"] for e in last5]),
                        _avg([e["rebounds"] for e in last5]),
                        _avg([e["assists"] for e in last5]),
                    ),
                    1,
                ),
                "points_per_game": round(_avg([e["points"] for e in entries]), 1),
                "rebounds_per_game": round(_avg([e["rebounds"] for e in entries]), 1),
                "assists_per_game": round(_avg([e["assists"] for e in entries]), 1),
                # Threes made a game: the Players page's fourth leader board.
                # A rate (three_pt_pct) is not a count, and the leader board wants
                # the count.
                "threes_per_game": round(_avg([e["fg3m"] for e in entries]), 1),
                "fg_pct": round(total_fgm / total_fga, 3) if total_fga else 0.0,
                "three_pt_pct": round(total_fg3m / total_fg3a, 3) if total_fg3a else 0.0,
                "ft_pct": round(total_ftm / total_fta, 3) if total_fta else 0.0,
                "usage_rate": round(_avg([e["usage_share"] for e in entries]), 3),
                "minutes_per_game": round(_avg([e["minutes"] for e in entries]), 1),
            }
        )
    return rows


def score_and_store_predictions(
    games_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str, players: pd.DataFrame | None = None,
    only_game_ids: set | None = None,
) -> int:
    """Scores every game that survives feature assembly with the trained model
    and stores the result as a tracked prediction. Returns the number stored.

    Games in `games_df` are expected to already be completed (their real
    pre-game rolling features are leak-free since build_training_frame only
    ever looks at *prior* games via shift(1)) - this is how a backtest lets
    us show real model output for real historical games in the UI.

    Win probability is derived from margin/sigma (Phi(margin/sigma)) for
    coherence. If the manifest lacks margin_sigma, we fail loudly.

    `only_game_ids` stores predictions for those games only: features are built over the whole `games_df` (so a
    multi-season frame keeps its mature rolling state) but earlier seasons are not re-stored in the tracking DB.
    """
    # Trusted artifacts: these .pkl files are written by run_retrain_pipeline
    # (via joblib.dump) in this same pipeline run — not from an external or
    # user-uploaded source.
    try:
        win_model = joblib.load(models_dir / "win_probability_model.pkl")
        margin_model = joblib.load(models_dir / "margin_model.pkl")
        total_model = joblib.load(models_dir / "total_model.pkl")
    except FileNotFoundError as exc:
        logger.warning("game model missing, skipping game scoring: %s", exc.filename)
        return 0

    # Load margin_sigma from manifest (populated by retrain from walk-forward residuals)
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}; run /retrain first")
    manifest = json.loads(manifest_path.read_text())
    margin_sigma = manifest.get("metrics", {}).get("margin", {}).get("residual_sigma")
    if margin_sigma is None or not np.isfinite(margin_sigma) or margin_sigma <= 0:
        raise ValueError(f"manifest missing positive finite margin_sigma; cannot serve win-from-margin. Got: {margin_sigma!r}")

    frame, feature_cols = build_training_frame(games_df, players=players)
    if only_game_ids is not None:
        frame = frame[frame["game_id"].isin(only_game_ids)].reset_index(drop=True)
    if len(frame) == 0:
        return 0

    margins = margin_model.predict(frame[feature_cols])
    totals = total_model.predict(frame[feature_cols]) + manifest_total_drift(manifest)

    created_at = datetime.now(timezone.utc).isoformat()
    store.insert_predictions(db_path, [
        dict(game_id=row["game_id"], created_at=created_at, model_version=model_version,
             home_win_prob=win_prob(float(margins[i]), margin_sigma),
             predicted_margin=float(margins[i]), predicted_total=float(totals[i]))
        for i, row in frame.iterrows()
    ])
    return len(frame)


def run_ingest(
    start: str,
    end: str,
    player_hub_days: int = 21,
    skip_predictions: bool = False,
    db_path: Path | None = None,
    log=print,
) -> dict:
    """The full real-data pipeline: fetch schedule/box scores, refresh
    schedule/hub caches, train the team-outcome and player-prop models,
    and (unless skip_predictions) score + store predictions for every
    completed and upcoming game. Returns a summary dict of what was done.

    Used by both the CLI (`main`, below) and the admin HTTP trigger
    (`api/routes.py::refresh_full`) so a live deployment with no shell
    access can populate its own tracking DB the same way a local run does.
    """
    db_path = db_path or config.TRACKING_DB_PATH
    summary: dict = {"start": start, "end": end, "player_hub_days": player_hub_days}

    log(f"Fetching schedule {start} to {end} from ESPN...")
    games = fetch_schedule_range(start, end)
    summary["games_found"] = len(games)
    log(f"  {len(games)} games found")

    log("Fetching box scores for completed games...")
    games = enrich_with_boxscores(games)

    schedule_path = config.DATA_DIR / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(to_schedule_cache(games)))
    log(f"  wrote {schedule_path}")

    hub_dir = config.DATA_DIR / "cache" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "teams.json").write_text(json.dumps(compute_team_hub(games)))
    (hub_dir / "rankings.json").write_text(json.dumps(compute_power_rankings(games)))
    (hub_dir / "standings.json").write_text(json.dumps(compute_standings(games)))

    # Player box scores for EVERY completed game in the window (not just the hub window): the availability
    # features fold cumulative minutes from the first game of the window, so a bounded recent fetch would
    # rebuild them from ~3 weeks of history. The ESPN cache is gitignored, so on a CI runner this is one extra
    # request per game (~0.2s throttle each) -- the same count as the team box scores fetched above.
    all_player_boxscores = fetch_player_boxscores(games)
    if player_hub_days > 0:
        cutoff = (date.fromisoformat(end) - timedelta(days=player_hub_days)).isoformat()
        recent_games = [g for g in games if g["game_date"] >= cutoff]
        recent_ids = {g["game_id"] for g in recent_games}
        player_boxscores = {gid: rows for gid, rows in all_player_boxscores.items() if gid in recent_ids}
        log(f"Player box scores: {len(all_player_boxscores)} games in window, {len(player_boxscores)} since {cutoff} for the hub/props...")
        store.init_db(db_path)
        record_names_from_boxscores(player_boxscores, db_path)
        hub_rosters = get_rosters()
        team_by_player = {p["player_id"]: team for team, ps in (hub_rosters or {}).items() for p in ps}
        (hub_dir / "players.json").write_text(json.dumps(compute_player_hub(recent_games, player_boxscores, team_by_player=team_by_player)))
    else:
        recent_games = []
        player_boxscores = {}
        (hub_dir / "players.json").write_text(json.dumps([]))
    availability_players = to_player_training_frame(games, all_player_boxscores)
    # Frozen earlier seasons in front of the window: the retrain and the scoring below both see them, so the model is
    # fitted on, and scores from, the same mature rolling state. Only the window's own games are stored as predictions.
    window_game_ids = {g["game_id"] for g in games}
    all_games, availability_players = merge_history(games, availability_players, start)
    summary["history_games"] = len(all_games) - len(games)
    log(f"  frozen history: {summary['history_games']} earlier games in front of the window")
    log(f"  wrote hub caches to {hub_dir}")

    training_df = to_training_frame(all_games)
    log(f"Training on {len(training_df)} completed games with full box scores...")
    # api/deps.py resolves this path and POST /retrain 400s without it, but
    # nothing in src/ ever wrote it -- so that endpoint was unrunnable in every
    # environment, including the VPS. It carries the box-score fields, which
    # to_schedule_cache deliberately drops, so this is its own file.
    training_path = config.DATA_DIR / "cache" / "training" / "games.json"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(training_df.to_dict(orient="records")))
    summary["training_games_written"] = len(training_df)
    log(f"  wrote {training_path}")
    # Slim player rows for the admin /retrain endpoint (same untracked cache dir as games.json).
    slim = ["game_id", "game_date", "team", "player_id", "minutes", "points", "rebounds", "assists"]
    (training_path.parent / "players.json").write_text(
        json.dumps(availability_players[slim].to_dict(orient="records") if len(availability_players) else [])
    )

    models_dir = config.PROJECT_ROOT / "models"
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()
    manifest = run_retrain_pipeline(
        training_df, models_dir, model_version=model_version, trained_at=trained_at, players=availability_players,
    )
    summary["team_model_metrics"] = manifest["metrics"]
    log(f"  trained {model_version}: {manifest['metrics']}")

    player_training_df = to_player_training_frame(recent_games, player_boxscores)
    log(f"Training player prop models on {len(availability_players)} real player-game rows (frozen history + window)...")
    player_manifest = train_player_prop_models(availability_players, models_dir, model_version=model_version, trained_at=trained_at)
    summary["player_model_metrics"] = player_manifest["metrics"]
    log(f"  trained player props: {player_manifest['metrics']}")

    if skip_predictions:
        log("  skip_predictions set: not scoring/storing predictions")
    else:
        store.init_db(db_path)

        stored = score_and_store_predictions(training_df, models_dir, db_path, model_version, players=availability_players, only_game_ids=window_game_ids)
        summary["completed_predictions_stored"] = stored
        log(f"  stored {stored} real predictions for browsing in the UI")

        stored_upcoming = score_upcoming_games(all_games, models_dir, db_path, model_version, players=availability_players)
        summary["upcoming_predictions_stored"] = stored_upcoming
        log(f"  stored {stored_upcoming} predictions for upcoming games")

        player_stored = score_and_store_player_predictions(player_training_df, models_dir, db_path, model_version)
        summary["completed_player_predictions_stored"] = player_stored
        log(f"  stored {player_stored} real player predictions")

        # Score against the FULL window's box scores, not the hub's recent
        # slice: a rolling average built from the last 21 days alone is a
        # few preseason minutes in October, which is the failure
        # player_props_refresher.HISTORY_DAYS exists to avoid on its path.
        # Training and scoring both use the window plus the frozen earlier seasons (a recent slice trained a model
        # that answered half the truth); the hub's recent slice only bounds what is shown and stored for browsing.
        # player_hub_days == 0 keeps its documented meaning -- no player
        # scoring at all -- so the widened call only replaces the recent
        # one when the hub window is on.
        if player_hub_days > 0:
            # Current rosters decide who is on which team (None = feed down: the box-score rule, with a warning).
            rosters = get_rosters()
            hist_players = availability_players[~availability_players["game_id"].isin(window_game_ids)]
            player_stored_upcoming = score_upcoming_player_props(
                games, all_player_boxscores, models_dir, db_path, model_version, rosters=rosters, extra_completed=hist_players,
            )
        else:
            player_stored_upcoming = score_upcoming_player_props(recent_games, player_boxscores, models_dir, db_path, model_version)
        summary["upcoming_player_predictions_stored"] = player_stored_upcoming
        log(f"  stored {player_stored_upcoming} player predictions for upcoming games")

        outcomes_stored = store_player_outcomes(player_training_df, db_path)
        summary["player_outcomes_stored"] = outcomes_stored
        log(f"  stored {outcomes_stored} real player outcomes for settlement")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real NBA schedule/box-score data and refresh caches.")
    parser.add_argument("--start", default=default_window_start(), help="Defaults to the start of the previous NBA season (Oct 1) -- see default_window_start.")
    parser.add_argument("--end", default=default_window_end(), help="Defaults to 14 days out.")
    parser.add_argument(
        "--player-hub-days", type=int, default=21,
        help="How many days (most recent, within --start/--end) to fetch per-player box scores for. "
        "0 skips Player Hub and player-prop training/scoring entirely. Also bounds the player-prop "
        "model's training window — pass a value covering the whole --start/--end range (e.g. 9999) "
        "for a real full backfill. Bounded by default since it's a second ESPN request per game.",
    )
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip scoring/storing predictions into the tracking DB. For CI contexts (e.g. a scheduled "
        "workflow with no persistent tracking.db) that only need fresh schedule/hub caches and a "
        "retrained model committed to git — predictions belong on the deployed server's live DB, "
        "not a stateless CI runner's throwaway one.",
    )
    args = parser.parse_args()
    run_ingest(args.start, args.end, player_hub_days=args.player_hub_days, skip_predictions=args.skip_predictions)


if __name__ == "__main__":
    main()
