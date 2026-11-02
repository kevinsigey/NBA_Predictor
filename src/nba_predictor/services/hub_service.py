import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from nba_predictor.api.schemas import (
    ConfidenceBucketOut,
    PropStatOut,
    TrackRecordOut,
    TrackRecordPickOut,
    TrackRecordPicksOut,
    TrackRecordTallyOut,
    TrackRecordWeekOut,
    VsMarketOut,
    VsMarketScopeOut,
    VsMarketWeekOut,
)
from nba_predictor.tracking import store
from nba_predictor.tracking.store import get_connection
from nba_predictor.tracking.timing import (
    earliest_recorded,
    earliest_recorded_outcome,
    earliest_run,
    latest_pre_tip,
    made_before_tip,
)


_CONFIDENCE_BUCKETS = [("50-60%", 0.50, 0.60), ("60-70%", 0.60, 0.70), ("70%+", 0.70, 1.01)]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# The player-props aggregate cache.
#
# WHY THIS EXISTS. `_summarize_player_props` aggregates 270,660 snapshots and
# 124,000 outcomes to produce eight rows. Measured on the live database, it is
# 1.465s, and `GET /hub/track-record` was 3.3s where every other hub route is
# 0.09-0.19s. Kevin accepted a few minutes of staleness on this route
# (NBA_Predictor#48) to get there.
#
# WHAT IS CACHED, AND WHAT IS NOT. Only this aggregate. The per-pick tables,
# the weekly breakdowns and the counted record are recomputed on every request,
# because those answer "what did we say, and when" -- a question about picks
# still being made -- and a five-minute-old answer to it is not a rounding
# difference. The aggregate answers "how wrong have we been across the season",
# which is a different kind of claim and is the expensive one.
#
# WHY THE TTL IS THE ONLY KEY. `api/routes.py::_MAE_CACHE` also keys on the
# tracking file's (mtime, size), so a write invalidates it immediately. That is
# strictly fresher, and it is deliberately NOT done here: the decision accepted
# staleness bounded by time, so this is bounded by time. One entry per database,
# which also means the dict cannot grow.
#
# THE FAILURE RULE, WHICH IS NOT THE MAE CACHE'S RULE. On a failed recompute
# past the TTL this serves the last good value and says so in
# `served_stale_seconds`, rather than dropping the entry and raising the way
# `_mae_record` does. That is the decision for this route: an error page is
# worse than a disclosed old number. It is still never silent -- the field
# carries the age in seconds, so the site can print it. With nothing cached yet
# (a cold failure) there is no last good value and the error propagates, because
# inventing a figure is not on the menu.
# --------------------------------------------------------------------------

PLAYER_PROPS_CACHE_TTL_SECONDS = float(os.getenv("PLAYER_PROPS_CACHE_TTL_SECONDS", "300"))

# db_path -> (stored_at, value, last_failed_at). `last_failed_at` is recorded so
# the age of a value that could not be refreshed is reportable rather than lost.
_PROPS_AGG_CACHE: dict[str, tuple[float, TrackRecordOut | None, float | None]] = {}

# One aggregate in flight at a time, per the MAE cache's reasoning: two
# concurrent cold requests would otherwise run the identical 1.5s aggregate and
# serialise on the tracking store's global lock. Bounded, so a hung warm costs a
# request its own compute rather than its availability.
_PROPS_AGG_LOCK = threading.Lock()
PROPS_AGG_COMPUTE_WAIT_SECONDS = float(os.getenv("PROPS_AGG_COMPUTE_WAIT_SECONDS", "30"))


def clear_player_props_cache() -> None:
    """Drop every cached aggregate. For tests, and for a write path that wants a
    guaranteed cold read rather than whatever the TTL happens to allow."""
    _PROPS_AGG_CACHE.clear()


def cached_player_props(
    db_path: Path, *, now=time.monotonic, compute=None
) -> TrackRecordOut | None:
    """The player-props aggregate, from cache inside the TTL and fresh past it.

    `compute` is the aggregation, injected so a test can count calls and fail on
    demand without touching 400k rows. `now` is a clock, injected so the bound is
    testable without sleeping.

    Returns None when there is nothing to report and nothing cached, which is the
    same answer the uncached function gives.
    """
    key = str(db_path)
    if compute is None:
        def compute():
            return _summarize_player_props(db_path)

    entry = _PROPS_AGG_CACHE.get(key)
    if entry is not None and now() - entry[0] < PLAYER_PROPS_CACHE_TTL_SECONDS:
        return _with_freshness(entry[1], None)

    waited = _PROPS_AGG_LOCK.acquire(timeout=PROPS_AGG_COMPUTE_WAIT_SECONDS)
    try:
        # Re-read under the lock: an in-flight compute may have just landed.
        entry = _PROPS_AGG_CACHE.get(key)
        if entry is not None and now() - entry[0] < PLAYER_PROPS_CACHE_TTL_SECONDS:
            return _with_freshness(entry[1], None)
        try:
            value = compute()
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed: see below
            if entry is None or entry[1] is None:
                # Cold failure: there is no last good value to serve. Propagate --
                # a request must not be handed a figure that was never computed.
                logger.error(
                    "player-props aggregate failed with nothing cached for %s: %s: %s",
                    db_path, type(exc).__name__, exc,
                )
                raise
            # Warm failure: serve the last good value, aged and disclosed.
            logger.error(
                "player-props aggregate refresh failed for %s, serving the last good "
                "value (%.0fs old): %s: %s",
                db_path, now() - entry[0], type(exc).__name__, exc,
            )
            _PROPS_AGG_CACHE[key] = (entry[0], entry[1], now())
            return _with_freshness(entry[1], now() - entry[0])
        _PROPS_AGG_CACHE[key] = (now(), value, None)
        return _with_freshness(value, None)
    finally:
        if waited:
            _PROPS_AGG_LOCK.release()


def warm_player_props_cache(db_path: Path, *, now=time.monotonic) -> bool:
    """Fill the cache off the request path. Never raises: a warm that fails must
    not take the API down, and must not leave a previous value looking current."""
    try:
        cached_player_props(db_path, now=now)
    except Exception as exc:  # noqa: BLE001 - reported, not raised: see docstring
        logger.error("player-props cache warm failed for %s: %s: %s", db_path, type(exc).__name__, exc)
        return False
    return True


def _with_freshness(value: TrackRecordOut | None, stale_seconds: float | None) -> TrackRecordOut | None:
    """The value, tagged with how stale it is. None means fresh.

    A copy, so the cached object is never mutated by a consumer's rendering --
    and so a `served_stale_seconds` on one response cannot leak into the next
    request served from the same cached value.
    """
    if value is None:
        return None
    return value.model_copy(update={"served_stale_seconds": stale_seconds})


def _bucket_for(prob: float) -> str | None:
    for label, lo, hi in _CONFIDENCE_BUCKETS:
        if lo <= prob < hi:
            return label
    return None


def _bucket_tally(bucket_stats: dict[str, list[int]], prob: float | None, hit: bool) -> None:
    """Tallies one counted graded pick into its confidence bucket.

    Buckets are computed over the COUNTED picks only (one per game+market),
    so the bucket rates reconcile with the headline rate.
    """
    if prob is None:
        return
    bucket = _bucket_for(prob)
    if bucket is not None:
        bucket_stats[bucket][0] += 1
        if hit:
            bucket_stats[bucket][1] += 1


def _confidence_buckets(bucket_stats: dict[str, list[int]]) -> list[ConfidenceBucketOut] | None:
    """Builds the bucket panel; None when nothing was graded (never measured,
    which is a different statement from 0%)."""
    if sum(n for n, _ in bucket_stats.values()) == 0:
        return None
    return [
        ConfidenceBucketOut(
            bucket=label,
            total_predictions=b_total,
            correct_predictions=b_correct,
            hit_rate=round(b_correct / b_total, 3) if b_total else None,
        )
        for label, (b_total, b_correct) in bucket_stats.items()
    ]


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_player_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["player_id"]: row["player_name"] for row in rows}


def player_name_map(db_path: Path, hub_path: Path) -> dict[str, str]:
    """The ONE id -> name resolver: hub players first, then every player the
    pipeline has seen in a box score. Callers fall back to the id themselves
    (last resort) when the id is in neither."""
    from ..tracking import store

    # A blank hub name must not shadow a name we do know: only non-blank hub names take precedence.
    hub = {k: v for k, v in load_player_name_map(hub_path).items() if isinstance(v, str) and v.strip()}
    return {**store.load_player_names(db_path), **hub}


def counted_picks(db_path: Path, schedule: list[dict]) -> list[tuple[dict, list]]:
    """(game, [COUNTED pick, *reruns]) for every completed game.

    The counted pick is the EARLIEST recorded row for that game; the reruns are
    the later rows it displaced. The counting key here is the game.
    `predictions` holds one probability per game -- the model's own winner call,
    which is the `game_outcome` market -- so there is nothing else to key on.
    The odds tables are keyed on (game, market) and player props on
    (game, player, stat); see `_settle_market_predictions` and
    `tracking/player_props.py`.

    Every completed game with a recorded pick is in this list, whenever the pick
    was made. Before 2026-10-01 (predictor-hub #66) this function returned only
    the LATEST row made before tip-off and counted a game whose every row came
    from the retrain backtest as `n_rebuilt`, outside every rate. That is the
    "with every model change it will stop tracking" failure: re-running the
    models made the record worse, not longer.

    The reruns are returned rather than dropped because rule 1 is that
    recorded stays recorded, and the per-pick list is where a reader sees the
    history. They are never graded.

    Returns the picks as stored; `made_before_tip` is derived per pick at read
    time, so nothing here decides or records when a pick was made.
    """
    schedule_by_id = {g["game_id"]: g for g in schedule}
    by_game: dict[str, list] = {}
    with get_connection(db_path) as conn:
        for row in conn.execute("SELECT * FROM predictions"):
            by_game.setdefault(row["game_id"], []).append(row)

    picks: list[tuple[dict, list]] = []
    for game_id, rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        ordered = earliest_recorded(rows)
        if ordered:
            counted_stamp = ordered[0]["created_at"]
            picks.append((game, [ordered[0]] + [r for r in rows if r["created_at"] != counted_stamp]))
    return picks


def pre_tip_picks(db_path: Path, schedule: list[dict]) -> list[tuple[dict, dict]]:
    """The counted picks whose own timestamps prove they were made before
    tip-off -- the honest read of what the model would have said on the night.

    This is the secondary figure beside the headline, and the population
    `compute_model_calibration` grades. It is NOT a filter on the record: the
    record counts every counted pick, and this is a subset of that same
    counted set, so the two cannot describe different games.

    One counted pick per game, then filtered -- not "the newest pre-tip row per
    game". Those differ when the model was rerun before tip-off, and letting
    each view pick its own row is how two figures on one page end up counting
    two different runs of the same game.
    """
    return [(game, rows[0]) for game, rows in counted_picks(db_path, schedule)
            if made_before_tip(rows[0]["created_at"], game)]


def _settle_game_outcome(
    db_path: Path, schedule: list[dict]
) -> tuple[TrackRecordOut | None, list[tuple[date, bool]], list[tuple[date, bool]]]:
    """Settles the model's own win/loss call (predictions.home_win_prob >= 0.5)
    against each game's actual result, for every COUNTED pick. Every result is
    a real completed game, joined via the schedule cache's home_pts/away_pts
    (populated by pipeline/ingest.py).

    Two figures come out of one pass over one list, computed by the same code
    over two frames: the headline over every counted pick, and `pre_tip` over
    the subset whose own stamp proves it was made before tip-off. They cannot
    then disagree about a game, a grade or a week.

    Returns the row, its graded picks as (game_date, correct) pairs -- which is
    what the weekly breakdown is built from, so a week can never be graded more
    loosely than the headline above it -- and the same pairs for the pre-tip
    subset, which is the second week table."""
    picks = counted_picks(db_path, schedule)
    if not picks:
        return None, [], []

    graded: list[tuple[date, bool]] = []
    pre_tip_graded: list[tuple[date, bool]] = []
    per_pick: list[TrackRecordPickOut] = []
    n_rebuilt = 0
    n_unplaced = 0
    unplaced_correct = 0
    for game, rows in picks:
        pick = rows[0]
        day = _game_date(game)
        before = made_before_tip(pick["created_at"], game)
        correct = (pick["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"])
        if day is None:
            # The schedule holds neither a game_date nor a tip_off, so there is
            # no week to file this pick under. It still COUNTS -- rule 1 has no
            # exception for a pick nobody dated -- and it is published per pick
            # with `gameday: null`. What it cannot do is appear in a week table,
            # so `n_unplaced` states the size of that gap and the week table's
            # identity is `total - n_unplaced`, not `total`.
            n_unplaced += 1
            unplaced_correct += 1 if correct else 0
            # `pick_cutoff` cannot read a game with no game_date and no
            # tip_off, so `made_before_tip` failed closed to False for it: it
            # is a not-pre-tip pick, and counting it here is what keeps
            # `total == pre_tip.total + n_rebuilt` true with unplaced picks in
            # the headline.
            n_rebuilt += 1
        else:
            graded.append((day, correct))
            if before:
                pre_tip_graded.append((day, correct))
            else:
                n_rebuilt += 1
        per_pick.append(_winner_pick_row(game, pick, correct, before, day))
        for rerun in rows[1:]:
            per_pick.append(_winner_pick_row(
                game, rerun,
                (rerun["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"]),
                made_before_tip(rerun["created_at"], game), day, counted=False,
            ))

    return _tally(
        market="game_outcome", graded=graded, pre_tip_graded=pre_tip_graded,
        n_rebuilt=n_rebuilt, n_unplaced=n_unplaced, unplaced_correct=unplaced_correct,
        per_pick=_sorted_picks(per_pick),
    ), graded, pre_tip_graded


def _sorted_picks(per_pick: list[TrackRecordPickOut]) -> list[TrackRecordPickOut]:
    """Published oldest-stamp-first, so the list reads as the record's history."""
    return sorted(per_pick, key=lambda p: (p.created_at, p.game_id))


def _winner_pick_row(game: dict, pick, correct: bool, before: bool, day: date | None,
                     counted: bool = True) -> TrackRecordPickOut:
    """One disclosed pick for `game_outcome`, in words a reader can check."""
    side = game["home_team"] if pick["home_win_prob"] >= 0.5 else game["away_team"]
    actual = game["home_team"] if game["home_pts"] > game["away_pts"] else game["away_team"]
    return TrackRecordPickOut(
        game_id=game["game_id"], market="game_outcome", pick=side, actual=actual,
        hit=correct, made_before_tip=before, created_at=pick["created_at"],
        counted=counted, gameday=day.isoformat() if day else None,
    )


def _tally(
    *, market: str, graded: list[tuple[date, bool]], pre_tip_graded: list[tuple[date, bool]],
    n_rebuilt: int, n_push: int = 0, n_unplaced: int = 0, unplaced_correct: int = 0,
    per_pick: list[TrackRecordPickOut] | None = None, window: list[date] | None = None,
    confidence_buckets: list[ConfidenceBucketOut] | None = None,
) -> TrackRecordOut:
    """One market's record: the headline, the pre-tip subset beside it, and the
    per-pick disclosure.

    `total_predictions`, `correct_predictions` and `hit_rate` keep their names
    and now mean the headline -- every counted pick, whenever it was made. What
    changed is which picks that is, so a site reading `hit_rate` reads the
    fuller record and nothing has to be renamed.

    `n_rebuilt` keeps its name and changes meaning: it is no longer a count of
    finals left out, it is the count of graded counted picks made at or after
    their own tip-off. That keeps it the reconciliation between the two
    figures -- `total_predictions == pre_tip.total_predictions + n_rebuilt` --
    which is the identity a reader needs to see how much of the headline is
    the rerun rather than the night. Pushes are in `n_push` for both figures
    and in neither rate, so the identity survives one.

    `n_unplaced` counts counted picks the schedule cannot date, so they appear
    in the headline and in `per_pick` but in no week. It is stated rather than
    absorbed: a week table that quietly sums to `total - n_unplaced` reads as a
    broken identity until the difference is named.

    `pre_tip` is a whole sub-record rather than three loose numbers, so the
    secondary figure carries its own n, its own rate and its own week table
    exactly as the headline does.
    """
    correct = sum(1 for _, ok in graded if ok) + unplaced_correct
    total = len(graded) + n_unplaced
    pre_correct = sum(1 for _, ok in pre_tip_graded if ok)
    pre_total = len(pre_tip_graded)
    return TrackRecordOut(
        market=market, total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else None,
        n_rebuilt=n_rebuilt, n_push=n_push, n_pre_tip=pre_total, n_unplaced=n_unplaced,
        settled=True,
        pre_tip=TrackRecordTallyOut(
            total_predictions=pre_total, correct_predictions=pre_correct,
            hit_rate=round(pre_correct / pre_total, 3) if pre_total else None,
            n_push=n_push, weekly=_weekly_rows(pre_tip_graded, window or []),
        ),
        per_pick=per_pick or [],
        confidence_buckets=confidence_buckets,
    )


def _grade_market_row(row, game: dict) -> bool | None:
    """True = the pick cleared its line, False = it did not, None = no grade.

    A grade is the pick against the line it was priced at, never against a
    line re-read from anywhere else:

    - **h2h**: `selection` is the side the model backed; it wins the game.
    - **spread**: `point` is the selection's own line in book convention
      (BOS -4.5 stores -4.5 for BOS), so the side covers when its margin
      beats `-point`. The same threshold `refresh_odds._model_probability`
      prices with (`line=-point`); `test_spread_grading_uses_the_same_
      threshold_as_the_odds_writer` pins the two together, because a grader
      reading the sign the other way produces plausible rates that are all
      mirrored.
    - **total**: `point` is the line; `selection` is "over" or "under".

    None is a push (the result landed exactly on the line) or a row with no
    line to judge: nobody won that one, so it is counted in `n_push` and
    stays out of the rate rather than becoming a miss.
    """
    market = row["market"]
    home_pts, away_pts = game["home_pts"], game["away_pts"]

    if market == "h2h":
        winner = game["home_team"] if home_pts > away_pts else game["away_team"]
        return row["selection"] == winner

    point = row["point"]
    if point is None:
        return None

    if market == "spread":
        if row["selection"] == game["home_team"]:
            side_margin = home_pts - away_pts
        elif row["selection"] == game["away_team"]:
            side_margin = away_pts - home_pts
        else:
            return None
        # Covers when side_margin > -point; == is the push.
        margin_vs_line = side_margin + point
        return None if margin_vs_line == 0 else margin_vs_line > 0

    if market == "total":
        margin_vs_line = (home_pts + away_pts) - point
        if margin_vs_line == 0:
            return None
        return (margin_vs_line > 0) == (str(row["selection"]).lower() == "over")

    return None


def _settle_market_predictions(
    db_path: Path, schedule: list[dict], market: str
) -> tuple[TrackRecordOut | None, list[tuple[date, bool]], list[tuple[date, bool]]]:
    """Settles one priced market against actual results, once per game.

    **Counting key: (game, market).** The query already narrows to one market,
    so the key reduces to the game, and that is the market's own unit -- one
    graded pick per game per market. It is NOT (game, selection) or (game,
    bookmaker): odds refresh writes both sides for every bookmaker on every
    run, so keying any finer would grade the same game four times over and pin
    the rate near 50%.

    **Which run:** the EARLIEST recorded one (`earliest_run`), and within it
    the selection the model priced highest -- that row is the model's call. The
    rule used to be "the latest run made before tip-off", which meant a market
    re-run after a game stopped counting at all, and a model changed twice
    before tip-off was graded on its second opinion. Reversing that is rule 2
    of the 2026-10-01 spec, and it is the same rule `earliest_recorded`
    applies to the other two tables, so one rerun can never be graded twice
    across the three markets.

    Returns (None, [], []) when the schedule holds no finished game for this
    market -- the caller then reports the stored rows as unsettled rather than
    as 0%.
    """
    schedule_by_id = {g["game_id"]: g for g in schedule}
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM game_market_predictions WHERE market = ?", (market,)
        ).fetchall()

    bucket_stats: dict[str, list[int]] = {label: [0, 0] for label, _, _ in _CONFIDENCE_BUCKETS}
    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)

    completed_games = 0
    graded: list[tuple[date, bool]] = []
    pre_tip_graded: list[tuple[date, bool]] = []
    per_pick: list[TrackRecordPickOut] = []
    n_rebuilt = 0
    n_push = 0
    for game_id, game_rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        completed_games += 1
        day = _game_date(game)
        if day is None:
            continue
        pick, history = _counted_market_pick(game, game_rows, market)
        if pick is None:
            continue
        before = made_before_tip(pick["created_at"], game)
        verdict = _grade_market_row(pick, game)
        if verdict is None:
            # A push, or a row with no line: nobody won it, so it is counted
            # in `n_push` and stays out of BOTH rates rather than becoming a
            # miss. It is still published per pick, because it is a pick.
            n_push += 1
        else:
            graded.append((day, verdict))
            _bucket_tally(bucket_stats, pick["model_probability"], verdict)
            if before:
                pre_tip_graded.append((day, verdict))
            else:
                n_rebuilt += 1
        per_pick.append(_market_pick_row(game, pick, market, verdict, before, day))
        # Every other recorded row for this game+market: a later rerun, kept as
        # history. It stays in the table, stays in this list, and is marked
        # `counted: false` so the rows a reader tallies are exactly the rows
        # that produced the headline.
        for row in history:
            per_pick.append(_market_pick_row(game, row, market, _grade_market_row(row, game),
                                             made_before_tip(row["created_at"], game), day,
                                             counted=False))

    if completed_games == 0:
        return None, [], []
    return _tally(
        market=market, graded=graded, pre_tip_graded=pre_tip_graded,
        n_rebuilt=n_rebuilt, n_push=n_push, per_pick=_sorted_picks(per_pick),
        confidence_buckets=_confidence_buckets(bucket_stats),
    ), graded, pre_tip_graded


def _counted_market_pick(game: dict, game_rows: list, market: str):
    """The counted pick for one (game, market), and the reruns it displaced.

    The counted pick is the model's own call inside the EARLIEST recorded run:
    the selection the model priced highest, which is the side it backed and the
    line it priced it at. Everything else recorded for this game and market is
    history -- returned, never deleted, never graded.
    """
    run = earliest_run(game_rows)
    if not run:
        return None, []
    counted_stamp = run[0]["created_at"]
    pick = max(run, key=lambda r: r["model_probability"])
    history = [r for r in game_rows if r["created_at"] != counted_stamp]
    return pick, history


def _market_pick_row(game: dict, row, market: str, verdict: bool | None, before: bool,
                     day: date | None, counted: bool = True) -> TrackRecordPickOut:
    """One disclosed priced pick, in words a reader can check against the score."""
    point = row["point"] if "point" in row.keys() else None
    return TrackRecordPickOut(
        game_id=game["game_id"], market=market, pick=_pick_words(row, market),
        actual=_actual_words(game, market, row), hit=verdict, made_before_tip=before,
        created_at=row["created_at"], counted=counted,
        gameday=day.isoformat() if day else None,
        point=point,
    )


def _pick_words(row, market: str) -> str:
    """What the model backed, with the line it was priced at."""
    point = row["point"]
    if market in ("spread", "total") and point is not None:
        return f"{row['selection']} {float(point):+.1f}"
    return str(row["selection"])


def _actual_words(game: dict, market: str, row) -> str:
    """What actually happened, in the same words as the pick."""
    home_pts, away_pts = game["home_pts"], game["away_pts"]
    if market == "h2h":
        return game["home_team"] if home_pts > away_pts else game["away_team"]
    if market == "spread":
        margin = home_pts - away_pts
        return f"{game['home_team']} by {abs(margin)}"
    return f"{home_pts + away_pts} points"


# The markets this repo has a grading rule for. Anything else in
# game_market_predictions is reported with its stored count and `settled=false`.
_SETTLED_MARKETS = ("h2h", "spread", "total")


def _game_date(game: dict | None) -> date | None:
    """The day a game was played, for week grouping.

    `game_date` first (the schedule's own key); a `tip_off` date only, for a
    row written before `game_date` existed. None when neither parses -- and
    such a game can never have been graded anyway, because `timing.pick_cutoff`
    would have raised before `made_before_tip` could return True.
    """
    if game is None:
        return None
    raw = game.get("game_date")
    if raw:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    tip = game.get("tip_off")
    if tip:
        try:
            return datetime.fromisoformat(str(tip).replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _summarize_player_props(db_path: Path) -> TrackRecordOut | None:
    """Joins prediction snapshots to recorded outcomes, counting the EARLIEST
    snapshot per (game_id, player_id, stat) so re-scores don't double-count.
    mean_signed_error is mean(predicted - actual): positive means systematic
    over-prediction. Snapshots with no recorded position group as "Unknown".

    **The dedupe and both aggregations run in SQL, and that is a measured
    change, not a stylistic one.** This function was 95% of a 5.1 s
    `/hub/track-record` (cProfile on the live container): 270,660 snapshot rows
    and 124,000 outcome rows were pulled into Python and sorted, which meant
    ~400k timestamp parses and a 400k-element sort to produce two small
    tables. It now returns ~#stats x #positions rows and the endpoint's share
    of the work is gone.

    **What that trades away, and why it is safe here.** `earliest_recorded`
    orders by parsed *instant*, so a key whose stamps are spelled differently
    is resolved by when it happened, not by how it was written. `ORDER BY
    created_at` orders by string. These agree only if one writer emits one
    format, so that is now checked rather than assumed: every writer here
    stamps with `datetime.now(timezone.utc).isoformat()`, and on the live data
    all 394,660 stamps are length 32 with a `+00:00` suffix, with zero keys
    carrying mixed spellings and zero string-vs-instant disagreements.
    `test_stamps_share_one_format_so_string_order_is_instant_order` pins it, so
    a writer that changes format fails a test instead of silently reordering
    the record.

    The dedupe itself is unchanged in meaning -- `ROW_NUMBER() ... ORDER BY
    created_at, id` takes the earliest row per key, with `id` breaking an exact
    tie deterministically, and `rn = 1` keeps it. Both tables are deduped in
    their own CTE, before the join, because a shared dedupe after joining would
    let one side's choice depend on join order.

    The outcome side is deduped even though `uq_game_player_outcomes_key` should
    make it a no-op: on the live file it is one (0 duplicate keys in 124,000
    rows), and `test_prop_mae_pairs_the_earliest_recorded_outcome` proves it
    must not be skipped -- a file predating the index, or any writer that
    escapes it, doubles every count.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            WITH counted AS (
                SELECT game_id, player_id, stat, predicted_value, position,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_id, player_id, stat
                           ORDER BY created_at, id
                       ) AS rn
                FROM player_prediction_snapshots
            ),
            resolved AS (
                SELECT game_id, player_id, stat, actual_value,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_id, player_id, stat
                           ORDER BY recorded_at, id
                       ) AS rn
                FROM game_player_outcomes
            )
            SELECT s.stat AS stat,
                   COALESCE(s.position, 'Unknown') AS position,
                   COUNT(*) AS n,
                   SUM(ABS(s.predicted_value - o.actual_value)) AS sum_abs_error,
                   SUM(s.predicted_value - o.actual_value) AS sum_signed_error
            FROM counted s
            JOIN resolved o
              ON o.game_id = s.game_id
             AND o.player_id = s.player_id
             AND o.stat = s.stat
            WHERE s.rn = 1 AND o.rn = 1
            GROUP BY s.stat, COALESCE(s.position, 'Unknown')
            """
        ).fetchall()
    if not rows:
        return None

    by_stat: dict[str, list[tuple[int, float, float]]] = {}
    by_pos: dict[str, list[tuple[int, float]]] = {}
    counted = 0
    for r in rows:
        counted += r["n"]
        by_stat.setdefault(r["stat"], []).append((r["n"], r["sum_abs_error"], r["sum_signed_error"]))
        by_pos.setdefault(r["position"], []).append((r["n"], r["sum_abs_error"]))
    per_stat = [
        PropStatOut(
            stat=stat,
            n=sum(n for n, _, _ in parts),
            mae=round(sum(sa for _, sa, _ in parts) / sum(n for n, _, _ in parts), 3),
            mean_signed_error=round(sum(ss for _, _, ss in parts) / sum(n for n, _, _ in parts), 3),
        )
        for stat, parts in sorted(by_stat.items())
    ]
    per_position_mae = {
        position: round(sum(sa for _, sa in parts) / sum(n for n, _ in parts), 3)
        for position, parts in sorted(by_pos.items())
    }
    return TrackRecordOut(
        market="player_props",
        total_predictions=counted,
        # Props are judged by how far off the projection was, not by a hit/miss rule, so there is NO hit rate. These
        # used to be correct=0 / hit_rate=0.0 / settled=True, which a reader sees as "0% right on 121,148 picks".
        correct_predictions=0,
        hit_rate=None,
        settled=False,
        per_stat=per_stat,
        per_position_mae=per_position_mae,
    )


def _week_start(day: date) -> date:
    """Monday on or before `day` -- the same Monday-based week the site's
    `frontend/src/lib/weeks.ts` groups its schedule by, so a week the visitor
    already knows by name is the week this table reports."""
    return day - timedelta(days=day.weekday())


def _tracking_window(db_path: Path, schedule: list[dict], today: date) -> list[date]:
    """Every week from the first game the tracker wrote a pick for, through
    the current week.

    Enumerating weeks from the data (rather than emitting one row per group)
    is the whole point: a week the tracker skipped must read as "not tracked",
    not vanish, because a reader cannot tell a gap from an absence. The start
    is games with ROWS rather than with grades, so a week whose picks were all
    rebuilt or all unsettled still belongs to the window -- it just appears
    untracked.

    `/hub/vs-market` derives its window from this same function, so the two
    week tables on the page line up row for row.

    Empty when the tracking DB has nothing the schedule can date.
    """
    schedule_by_id = {g["game_id"]: g for g in schedule}
    with get_connection(db_path) as conn:
        ids = {row[0] for row in conn.execute("SELECT DISTINCT game_id FROM predictions")}
        ids |= {row[0] for row in conn.execute("SELECT DISTINCT game_id FROM game_market_predictions")}

    weeks = [_week_start(day) for day in
             (d for d in (_game_date(schedule_by_id.get(game_id)) for game_id in ids) if d is not None)]
    if not weeks:
        return []
    first = min(weeks)
    # Never before tracking began; always up to this week, so an in-progress
    # week with nothing graded yet shows as a row instead of ending the table.
    last = max(_week_start(today), first)
    return [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]


def _weekly_rows(graded: list[tuple[date, bool]], window: list[date]) -> list[TrackRecordWeekOut]:
    """One row per week in `window` for one market.

    `hit_rate` is None (not 0.0) when a week graded nothing: 0% is a claim
    that every pick missed, which is not what "never measured" means. The
    headline's rule is reused verbatim -- this is the only other place the
    market's record is computed, deliberately.
    """
    by_week: dict[date, list[bool]] = defaultdict(list)
    for day, correct in graded:
        by_week[_week_start(day)].append(correct)

    rows = []
    for week in window:
        results = by_week.get(week, [])
        n = len(results)
        correct = sum(1 for ok in results if ok)
        rows.append(TrackRecordWeekOut(
            week_start=week.isoformat(), n=n, correct=correct,
            hit_rate=round(correct / n, 3) if n else None, tracked=n > 0,
        ))
    return rows


#: How many of a market's latest counted pre-tip picks the summary carries in
#: `recent_picks`. The page shows 12 across all markets and its adapter merges
#: the per-market lists before slicing, so each market contributing its newest
#: 24 makes the merged top 12 exact while the summary stays small.
RECENT_PICKS_PER_MARKET = 24


def _recent_picks(per_pick: list[TrackRecordPickOut]) -> list[TrackRecordPickOut]:
    """A market's latest counted pre-tip picks, newest first.

    The sort mirrors the frontend adapter's: the game day when the schedule
    dated the game, else the pick's own stamp. Same population the page's
    "Latest counted picks" block has always shown -- counted picks made
    before tip-off -- lifted out of the full disclosure so the summary route
    does not have to ship the disclosure to render a dozen rows.
    """
    counted = [p for p in per_pick if p.counted and p.made_before_tip]
    counted.sort(key=lambda p: (p.gameday or p.created_at, p.created_at), reverse=True)
    return counted[:RECENT_PICKS_PER_MARKET]


def compute_track_record(
    db_path: Path, schedule: list[dict] | None = None, today: date | None = None,
    *, include_picks: bool = True,
) -> list[TrackRecordOut]:
    schedule = schedule or []
    today = today or date.today()
    rows: list[TrackRecordOut] = []
    settled_rows: dict[str, TrackRecordOut] = {}
    graded_by_market: dict[str, list[tuple[date, bool]]] = {}
    pre_tip_by_market: dict[str, list[tuple[date, bool]]] = {}

    with get_connection(db_path) as conn:
        market_counts = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    for market in _SETTLED_MARKETS:
        row, graded, pre_tip_graded = _settle_market_predictions(db_path, schedule, market)
        if row is not None:
            settled_rows[market] = row
            graded_by_market[market] = graded
            pre_tip_by_market[market] = pre_tip_graded

    for row in market_counts:
        settled_row = settled_rows.get(row["market"])
        if settled_row is not None:
            rows.append(settled_row)
        else:
            # Stored rows we cannot judge against results: the count is real,
            # the rate is not measured. hit_rate stays None so no surface can
            # print a 0% for it.
            rows.append(TrackRecordOut(
                market=row["market"], total_predictions=row["total"],
                correct_predictions=0, hit_rate=None, settled=False,
            ))

    game_outcome, outcome_graded, outcome_pre_tip = _settle_game_outcome(db_path, schedule)
    if game_outcome is not None:
        settled_rows["game_outcome"] = game_outcome
        graded_by_market["game_outcome"] = outcome_graded
        pre_tip_by_market["game_outcome"] = outcome_pre_tip
        rows.append(game_outcome)

    player_props = cached_player_props(db_path)
    if player_props is not None:
        rows.append(player_props)
    # Every settled market gets the SAME window, so the weekly tables align
    # row for row on the page without the frontend inventing a join -- and the
    # pre-tip week table gets the same window, or the two figures would be laid
    # out over different spans and invite a comparison that is not one.
    window = _tracking_window(db_path, schedule, today)
    if window:
        for row in rows:
            if row.settled:
                row.weekly = _weekly_rows(graded_by_market.get(row.market, []), window)
                if row.pre_tip is not None:
                    row.pre_tip.weekly = _weekly_rows(pre_tip_by_market.get(row.market, []), window)

    if not include_picks:
        # The summary the page loads first: identical aggregates, no per-pick
        # rows. The bounded recent list is lifted out of the disclosure
        # before the disclosure is dropped, so "Latest counted picks" still
        # renders from this response; the full disclosure is paged from
        # compute_track_record_picks. Inlined, the rows were ~14 MB of a
        # ~14.4 MB response -- the page deferred rendering the table until
        # its fold opened, but the download happened up front.
        for row in rows:
            row.recent_picks = _recent_picks(row.per_pick)
            row.per_pick = []

    return rows


def compute_track_record_picks(
    db_path: Path, schedule: list[dict] | None = None, *,
    market: str | None = None, game_ids: list[str] | None = None,
    limit: int = 200, offset: int = 0,
) -> TrackRecordPicksOut:
    """One page of the every-recorded-pick disclosure, newest stamp first.

    The rows are the same `per_pick` rows compute_track_record builds -- one
    builder, so the table can never disagree with the aggregates above it --
    filtered and paged here rather than shipped whole. `market` narrows to
    one market's rows; `game_ids` narrows to specific games (the odds
    comparison block asks for exactly the games it lists). `total` counts
    the filtered set before paging.
    """
    rows = compute_track_record(db_path, schedule)
    picks = [p for row in rows for p in row.per_pick]
    if market is not None:
        picks = [p for p in picks if p.market == market]
    if game_ids:
        wanted = set(game_ids)
        picks = [p for p in picks if p.game_id in wanted]
    picks.sort(key=lambda p: (p.created_at, p.game_id), reverse=True)
    return TrackRecordPicksOut(
        total=len(picks), limit=limit, offset=offset,
        picks=picks[offset:offset + limit],
    )


# The sentences the page prints verbatim, once per key it sends. A number
# nobody can interpret is not a decision aid, and a disclaimer nobody reads is
# not one either -- so the backend owns the wording and the frontend does not
# get to paraphrase it away. No profit, ROI or "beats the bookies" claim lives
# in here; `not_a_profit_claim` exists to say that out loud in the payload.
_VS_MARKET_METHOD: dict[str, str] = {
    "market_probability": (
        "The market number is the book's own price with the overround removed "
        "(Shin's method), so the two sides of a moneyline add to 100%. It is "
        "read straight from the odds-refresh run that priced the pick, not "
        "recomputed here."
    ),
    "edge": (
        "Edge is the model's probability for its pick minus the probability "
        "the price carried for that same side, in percentage points, positive "
        "when the model likes a side more than the price does. It measures "
        "disagreement with a price, not superiority: a closing price is the "
        "market's best estimate, so a well-calibrated model's average edge is "
        "near zero by design. A large average edge means one of the two is "
        "miscalibrated, not that the model is right."
    ),
    "disagreement": (
        "The disagreement cohort is the games where the model backed the side "
        "the price did not favour -- a pick against the price. Its hit rate is "
        "how often that side won, over exactly those games. Games the price "
        "split 50/50 are compared but left out of the cohort: an even price "
        "favours nobody, so there is no side to disagree with. This is the "
        "number to read for a decision; the mean edge is a calibration check."
    ),
    "not_a_profit_claim": (
        "This is agreement with a price, not a profit claim. No figure here is "
        "a return, a yield, a stake or a cent. We do not publish profit or ROI "
        "figures, and this comparison does not become one by being labelled "
        "'edge'."
    ),
    "population": (
        "The figures above cover every finished game whose moneyline was "
        "priced before tip-off, in every season the tracker holds. The week "
        "table covers the same window as the rest of this page: from the first "
        "game the tracker wrote a pick for through this week. The scope line "
        "states how many compared games sit inside that window and how many "
        "fall outside it, and the two add up to the count above."
    ),
}


def compute_vs_market(
    db_path: Path, schedule: list[dict] | None = None, today: date | None = None
) -> VsMarketOut:
    """The model's moneyline pick beside the price it was measured against.

    **Not relaxed by the 2026-10-01 track-record reversal, deliberately, and
    for the same reason `compute_model_calibration` is not.** This block is a
    comparison with a PRICE, not a track record: the whole point of it is what
    the model said against what the book said at the same moment. A row written
    after tip-off carries a price that did not exist when the reader could have
    taken it, and averaging those in would measure the model against a
    backtest. So the run is still the latest one made before tip-off, and
    "one pick per game -- the same rule the h2h record grades with" now refers
    to one pick per game, not to which run that pick came from: the record
    counts the earliest run (rule 2) while this keeps the last pre-tip price,
    because a price comparison has to be made against a price that was live.

    Per finished game: that run, the row with the higher model probability, and
    the price carried by that row. Eligible means a result to judge it
    against, a market_probability on the pick, and the other side priced at the
    SAME bookmaker in the same run: a price with only one side stored cannot be
    said to have favoured either side, and the disagreement test below would be
    reading a sum-to-100% that was never summed.

    `edge` is model minus price in percentage points on the model's own side.
    The disagreement cohort is `price < 0.5` -- the price fancied the other
    side; exactly 0.5 favours nobody, so it stays out of the cohort but
    remains in `n`.
    """
    schedule = schedule or []
    today = today or date.today()
    schedule_by_id = {g["game_id"]: g for g in schedule}

    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM game_market_predictions WHERE market = 'h2h'"
        ).fetchall()

    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)

    compared: list[dict] = []
    for game_id, game_rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        latest = latest_pre_tip(game_rows, game)
        if latest is None:
            continue
        day = _game_date(game)
        if day is None:
            continue
        run = [r for r in game_rows if r["created_at"] == latest["created_at"]]
        pick = max(run, key=lambda r: r["model_probability"])
        price = pick["market_probability"]
        if price is None:
            continue
        other_priced = [
            r for r in run
            if r["bookmaker"] == pick["bookmaker"]
            and r["selection"] != pick["selection"]
            and r["market_probability"] is not None
        ]
        if not other_priced:
            continue
        winner = game["home_team"] if game["home_pts"] > game["away_pts"] else game["away_team"]
        compared.append({
            "day": day,
            "game_id": str(game_id),
            "model": float(pick["model_probability"]),
            "price": float(price),
            "edge_points": (float(pick["model_probability"]) - float(price)) * 100.0,
            "disagrees": float(price) < 0.5,
            "hit": pick["selection"] == winner,
        })

    n = len(compared)
    disagreements = [c for c in compared if c["disagrees"]]
    hits = sum(1 for c in disagreements if c["hit"])
    n_disagree = len(disagreements)

    window = _tracking_window(db_path, schedule, today)
    by_week: dict[date, list[dict]] = defaultdict(list)
    for c in compared:
        by_week[_week_start(c["day"])].append(c)

    weekly = []
    for week in window:
        group = by_week.get(week, [])
        week_disagreements = [c for c in group if c["disagrees"]]
        week_hits = sum(1 for c in week_disagreements if c["hit"])
        weekly.append(VsMarketWeekOut(
            week_start=week.isoformat(), tracked=bool(group), n=len(group),
            mean_edge_points=round(_mean(c["edge_points"] for c in group), 1) if group else None,
            disagreement_n=len(week_disagreements),
            disagreement_hit_rate=(
                round(week_hits / len(week_disagreements), 3) if week_disagreements else None
            ),
        ))

    in_weekly = sum(w.n for w in weekly)
    return VsMarketOut(
        market="h2h",
        n=n,
        mean_model_probability=round(_mean(c["model"] for c in compared), 4) if n else None,
        mean_market_probability=round(_mean(c["price"] for c in compared), 4) if n else None,
        mean_edge_points=round(_mean(c["edge_points"] for c in compared), 1) if n else None,
        disagreement_n=n_disagree,
        disagreement_hit_rate=round(hits / n_disagree, 3) if n_disagree else None,
        disagreement_game_ids=sorted(c["game_id"] for c in disagreements),
        weekly=weekly,
        scope=VsMarketScopeOut(
            population="finished games with a pre-tip moneyline price",
            weekly_from=window[0].isoformat() if window else None,
            weekly_through=window[-1].isoformat() if window else None,
            n_games_total=n,
            n_games_in_weekly=in_weekly,
            n_games_outside_weekly=n - in_weekly,
        ),
        method=dict(_VS_MARKET_METHOD),
    )


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
