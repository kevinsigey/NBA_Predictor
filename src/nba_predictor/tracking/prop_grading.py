"""Player-prop grading: how close each counted projection landed.

The track record already grades player props as one aggregate MAE per stat
(`services/hub_service.py::_summarize_player_props`). This is the per-pick
counterpart: every counted projection, laid against the player's actual
box score and graded in units of the model's own error -- the PL
predictor's fixture-grading shape, applied to players instead of matches.
One graded population, three reads of it: the most recent completed game
day (`build_prop_grading`), a single game's complete grading for its
post-game review (`build_game_prop_grading`), and the model's tracking
aggregates over the whole counted record (inside `build_prop_grading`'s
payload -- per stat, pre-tip rows only, because an aggregate is the
model's record and the record is pre-tip by definition).

**Which pick is graded.** The COUNTED pick, exactly the one the track
record counts: the earliest recorded projection per (game, player, stat),
whenever it was made (`tracking/player_props.py::resolved_player_props`,
predictor-hub #66, 2026-10-01). A row is LABELLED `made_before_tip` where
its own timestamp proves it and is graded either way -- the label is the
honesty mechanism, not an exclusion. Nothing here claims a pre-tip record:
the payload carries the day's pre-tip count beside its graded count so the
two can never be silently conflated.

**The grade.** A miss is measured against the props model's own holdout
error for that stat (the `mae` in `models/player_props_manifest.json` --
the headline figure `train_player_prop_models` publishes, which is the
chronological-holdout MAE whenever a holdout exists). Bands:

* ``On the number`` -- |actual - projected| <= 0.5 x MAE
* ``Close``         -- |actual - projected| <= 1.0 x MAE
* ``Off``           -- |actual - projected| <= 2.0 x MAE
* ``Way off``       -- beyond that

A stat with no published MAE has no grade: the row is served with its delta
and `grade=None`, never judged against an invented band.

**Most confident.** The section ranks the day's projections by how far the
model moved from the player's own rolling average -- |projected - baseline|,
the model's biggest departures from what the player usually does. The
baseline is the same rolling average the features use
(`features/player_stats.py`: the mean of his previous 10 games, shift(1) so
the graded game never enters its own baseline), reconstructed from the
RECORDED outcomes in the tracking store. A player with no recorded prior
game has no baseline in stored data; his rows are graded but cannot be
ranked here, and the section says how many it ranked rather than quietly
covering a subset.

**Delta direction.** ``delta = actual - projected``: positive is the player
beating the projection, negative is the player falling short. It says
nothing about whether the model was "right" to move -- that is what the
grade is for.

**The interval.** Every graded row also carries the model's 80% range for
that line (``interval_low`` / ``interval_high`` / ``inside_interval``):
scaled conformal intervals calibrated on the counted record itself
(`tracking/conformal_intervals.py` -- the method the preregistered
trial-2 race selected, docs/conformal-scaled-results-2026-11.md). A
game's ranges are computed from counted residuals of game days strictly
before its own day, so the range a reader sees after the game is the
range the calibration available before tip-off produces; the aggregates
count how often the actual landed inside that range over the same
pre-tip population the grades count, so the coverage figure and the
ranges it judges can never describe two different intervals.
"""

from pathlib import Path

from nba_predictor.tracking import store
from nba_predictor.tracking.conformal_intervals import (
    BASIS_HOLDOUT,
    BASIS_ROLLING,
    BASIS_SEASON,
    LEVEL as INTERVAL_LEVEL,
    METHOD as INTERVAL_METHOD,
    PropIntervalCalibrator,
)
from nba_predictor.tracking.player_props import resolved_player_props
from nba_predictor.tracking.timing import earliest_recorded_outcome

#: The four graded stats, in the order the box score prints them.
STATS = ("points", "rebounds", "assists", "threes")

GRADE_ON_THE_NUMBER = "On the number"
GRADE_CLOSE = "Close"
GRADE_OFF = "Off"
GRADE_WAY_OFF = "Way off"

#: Grade bands as multiples of the stat's holdout MAE, tightest first. The
#: comparison carries the same epsilon discipline as the frontend review
#: (a miss of exactly the band is inside the band).
_BANDS = ((0.5, GRADE_ON_THE_NUMBER), (1.0, GRADE_CLOSE), (2.0, GRADE_OFF))
_EPSILON = 1e-9

#: How many rows each ranked section shows.
SECTION_LIMIT = 8

#: The rolling-average window, matching `build_player_feature_frame`.
BASELINE_WINDOW = 10

COUNTING_NOTE = (
    "Graded from the recorded predictions in the tracking store: the earliest "
    "recorded projection per player and stat, the same pick the track record "
    "counts. A row is labelled made before tip-off only where its own "
    "timestamp proves it; rows that cannot prove it are graded and labelled, "
    "not dropped."
)
CONFIDENCE_NOTE = (
    "Most confident ranks the day's projections by how far the model moved "
    "from the player's own rolling average -- the mean of his previous 10 "
    "recorded games. It is the model's biggest departures from baseline, not "
    "a probability, and it is ranked only over rows whose baseline the stored "
    "record can prove."
)
GRADE_NOTE = (
    "Grades measure the miss against the props model's own holdout error for "
    "that stat (MAE): On the number is within half the error, Close within "
    "the error, Off within twice it, Way off beyond that. The delta is "
    "actual minus projected: positive means the player beat the projection."
)
AGGREGATES_NOTE = (
    "Prop-model tracking over the whole counted record: every counted "
    "projection made before tip-off (the earliest recorded projection per "
    "player and stat whose own timestamp proves it was made in time), graded "
    "against what the player did. Per stat: the average miss, the average "
    "signed miss, and how the grades fall across the bands, beside the "
    "model's own holdout error for the stat."
)

#: The grade labels in band order, tightest first -- the order a distribution
#: bar reads in.
GRADES = (GRADE_ON_THE_NUMBER, GRADE_CLOSE, GRADE_OFF, GRADE_WAY_OFF)


def grade_for(abs_delta: float, mae: float | None) -> str | None:
    """The grade for one miss, or None when there is no measured error to
    judge it against (no band is ever invented)."""
    if mae is None or mae <= 0:
        return None
    ratio = abs_delta / mae
    for multiple, label in _BANDS:
        if ratio <= multiple + _EPSILON:
            return label
    return GRADE_WAY_OFF


def prop_bands(models_dir: Path) -> dict[str, float] | None:
    """The props model's own holdout MAE per stat, or None when there is none.

    The band is what a projection has to miss by before it counts as wrong, so
    it is the model's measured error and it moves with each retrain. A
    made-up default would be a number nobody measured, and the grading would
    judge players against it -- so a missing manifest means no band, never a
    default. (This is the reader `api/routes.py::_prop_bands` has always
    used; it lives here now so the snapshot builder and the API read the
    manifest the same way.)
    """
    import json

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


def prop_conformal_cold_start(models_dir: Path) -> dict[str, dict] | None:
    """The manifest's published holdout conformal parameters per stat, or
    None when the manifest carries none.

    The retrain measures these on the same chronological holdout as the
    MAE bands (`pipeline/ingest.py::train_player_prop_models` publishing
    `metrics[stat]["conformal"]`): the scaled method's sigma bins and its
    normalized 80% quantile, for the cold start -- the days before the
    counted record holds enough of its own residuals to calibrate on
    (`tracking/conformal_intervals.py`, tier 3). A manifest written by a
    retrain from before the block existed parses to None, exactly as a
    missing MAE parses to no band: no parameter is ever invented to fill
    the gap. Only stats whose block is complete are returned.
    """
    import json

    path = Path(models_dir) / "player_props_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    out: dict[str, dict] = {}
    for stat, figures in (manifest.get("metrics") or {}).items():
        block = (figures or {}).get("conformal")
        if not isinstance(block, dict):
            continue
        sigma = block.get("sigma")
        q80 = block.get("q80")
        if not isinstance(sigma, dict) or not isinstance(q80, (int, float)):
            continue
        if not all(k in sigma for k in ("edges", "scales", "floor")):
            continue
        out[str(stat)] = {
            "q80": float(q80),
            "sigma": {
                "edges": [float(e) for e in sigma["edges"]],
                "scales": [float(s) for s in sigma["scales"]],
                "floor": float(sigma["floor"]),
            },
            "n_holdout": int(block.get("n_holdout", 0)),
        }
    return out or None


def build_interval_calibrator(
    db_path: Path,
    schedule: list[dict],
    cold_start: dict[str, dict] | None = None,
) -> PropIntervalCalibrator:
    """The production interval calibrator over this store's counted record.

    Built from the same `_graded_rows` population the grading reads (with
    no bands -- the calibrator needs residuals, not grades), so a surface
    that serves projections directly (the game page's box score) prices
    its ranges from exactly the record the grading payload calibrates
    on. Callers cache the result; building it is a bulk store read.
    """
    rows = _graded_rows(db_path, schedule, None)
    return PropIntervalCalibrator.from_graded_rows(rows, cold_start)


INTERVAL_NOTE = (
    "The range on each line is the model's 80% interval, built by scaled "
    "conformal prediction from the counted record itself: each game day's "
    "ranges are calibrated only on counted lines from game days before it, "
    "with the width scaled by the size of the projection -- bigger lines "
    "get wider ranges. An interval is never carried across from an older "
    "day's calibration."
)


def _attach_intervals(
    rows: list[dict], cold_start: dict[str, dict] | None
) -> dict[tuple[str, str], dict]:
    """Attach `interval_low` / `interval_high` / `inside_interval` to every
    graded row, and return the calibration info per (stat, game_date).

    The calibrator is built from the rows themselves (their pre-tip subset
    calibrates; every row is priced), so the day payload, the per-game
    payload and the aggregates all read the same intervals -- the coverage
    the aggregates publish counts THESE rows' inside flags, never a
    recomputation that could drift from what the review showed.
    """
    if not rows:
        return {}
    calibrator = PropIntervalCalibrator.from_graded_rows(rows, cold_start)
    infos: dict[tuple[str, str], dict] = {}
    for r in rows:
        hw, info = calibrator.half_width(r["stat"], r["predicted"], r["game_date"])
        infos[(r["stat"], r["game_date"])] = info
        if hw is None:
            r["interval_low"] = None
            r["interval_high"] = None
            r["inside_interval"] = None
        else:
            r["interval_low"] = _round1(r["predicted"] - hw)
            r["interval_high"] = _round1(r["predicted"] + hw)
            r["inside_interval"] = bool(r["_abs_delta"] <= hw + _EPSILON)
    return infos


def _tier_phrase(bases: set[str | None]) -> str:
    """One clause describing how a day was calibrated, from the set of
    tiers its stats used."""
    if bases == {BASIS_ROLLING}:
        return "the trailing 14 game days before it"
    if bases == {BASIS_SEASON}:
        return "every counted game day before it (the season was too young for a 14-day window)"
    if bases == {BASIS_HOLDOUT}:
        return "the model's holdout parameters (the counted record had not built up yet)"
    return "the counted game days before it"


def _interval_note_for_day(
    infos: dict[tuple[str, str], dict], game_date: str, stats: list[str]
) -> str | None:
    """The interval basis note for one game day, or None when no stat that
    day carries an interval at all (the section then says nothing)."""
    day_infos = [infos[(s, game_date)] for s in stats if (s, game_date) in infos]
    if not day_infos or all(i["basis"] is None for i in day_infos):
        return None
    bases = {i["basis"] for i in day_infos if i["basis"] is not None}
    ns = [i["n"] for i in day_infos if i["basis"] is not None and i["basis"] != BASIS_HOLDOUT]
    note = f"80% ranges calibrated on {_tier_phrase(bases)}"
    if ns:
        note += f" ({min(ns)}-{max(ns)} counted lines per stat)"
    return note + ". " + INTERVAL_NOTE


def _rolling_baselines(db_path: Path, schedule: list[dict]) -> dict[tuple[str, str], list[tuple[str, str, float]]]:
    """{(player_id, stat): [(game_date, game_id, actual), ...] sorted by date}.

    Every recorded outcome, deduped to the earliest recorded row per key
    (the reader half of the outcomes unique index), dated by the schedule.
    An outcome whose game the schedule cannot date cannot be ordered against
    a graded game, so it is left out rather than placed by guess.
    """
    dates = {g.get("game_id"): g.get("game_date") for g in schedule if g.get("game_id")}
    with store.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT game_id, player_id, stat, actual_value, recorded_at FROM game_player_outcomes"
        ).fetchall()
    series: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for row in earliest_recorded_outcome(rows):
        day = dates.get(row["game_id"])
        if day is None:
            continue
        series.setdefault((row["player_id"], row["stat"]), []).append(
            (day, row["game_id"], float(row["actual_value"]))
        )
    for entries in series.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return series


def _baseline_for(
    series: dict[tuple[str, str], list[tuple[str, str, float]]],
    player_id: str,
    stat: str,
    game_date: str,
) -> float | None:
    """The player's rolling average going INTO this game: the mean of his
    previous `BASELINE_WINDOW` recorded games for this stat, or None when the
    stored record holds no prior game for him. The graded game itself is
    excluded by identity, not by position, so a same-day row can never leak
    into its own baseline."""
    entries = series.get((player_id, stat))
    if not entries:
        return None
    # Only strictly earlier dates count: a same-day row for a different game
    # is not provably before this one, so it is never in the baseline.
    prior = [value for day, gid, value in entries if day < game_date]
    if not prior:
        return None
    window = prior[-BASELINE_WINDOW:]
    return sum(window) / len(window)


def _round1(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def _props_tables_present(db_path: Path) -> bool:
    """Whether the store carries the player-prop tables at all.

    A store whose `init_db` never ran -- the snapshot staleness tests drive
    `generate_snapshot` against exactly such a path -- or one created before
    player props existed holds no recorded props by definition. That is an
    empty grading, not an error: the snapshot must still publish, with the
    sections empty, rather than refuse over tables it never promised.
    """
    import sqlite3

    try:
        with store.get_connection(db_path) as conn:
            found = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('player_prediction_snapshots', 'game_player_outcomes')"
                )
            }
    except sqlite3.Error:
        return False
    return found == {"player_prediction_snapshots", "game_player_outcomes"}


def _publish(row_list: list[dict]) -> list[dict]:
    """The public shape of graded rows: the private ranking keys (``_abs_*``)
    never leave the builder."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in row_list]


def _graded_rows(
    db_path: Path,
    schedule: list[dict],
    bands: dict[str, float] | None,
) -> list[dict]:
    """Every counted player projection in the store, graded -- the one
    population the day payload, the per-game payload and the aggregates all
    read from, so the three can never disagree about which rows exist.

    A store without the props tables (or with no resolved picks) grades to
    an empty list, not an error: see `_props_tables_present`.
    """
    if not _props_tables_present(db_path):
        return []
    resolved = resolved_player_props(db_path, schedule)
    if not resolved:
        return []

    games = {g.get("game_id"): g for g in schedule if g.get("game_id")}
    names = store.load_player_names(db_path)
    series = _rolling_baselines(db_path, schedule)

    rows: list[dict] = []
    for pick in resolved:
        game = games.get(pick["game_id"]) or {}
        game_date = game.get("game_date")
        if game_date is None:
            continue
        predicted = float(pick["predicted_value"])
        actual = float(pick["actual_value"])
        delta = actual - predicted
        mae = (bands or {}).get(pick["stat"])
        baseline = _baseline_for(series, pick["player_id"], pick["stat"], game_date)
        rows.append(
            {
                "game_id": pick["game_id"],
                "game_date": game_date,
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "player_id": pick["player_id"],
                "player_name": names.get(pick["player_id"]),
                "stat": pick["stat"],
                "predicted": _round1(predicted),
                "actual": _round1(actual),
                "delta": _round1(delta),
                "grade": grade_for(abs(delta), mae),
                "made_before_tip": bool(pick["made_before_tip"]),
                "baseline": _round1(baseline),
                "departure": _round1(predicted - baseline) if baseline is not None else None,
                "_abs_delta": abs(delta),
                "_abs_departure": abs(predicted - baseline) if baseline is not None else None,
            }
        )
    return rows


def _prop_aggregates(rows: list[dict], bands: dict[str, float] | None) -> dict:
    """How the prop model is tracking, over the whole counted record.

    **Pre-tip rows only.** The day payload grades every counted row and
    labels the ones that cannot prove they were made in time; the model's
    record may not do that -- an aggregate over rows made after tip-off
    would grade the model on answers it had already seen. So the tracking
    counts only rows whose own timestamp proves they were made before
    tip-off, and says so in its note.

    Per stat: how many, the average miss (MAE), the average signed miss
    (positive = players beat the projections on average -- the projections
    run low), how the grades fall across the bands as shares of the banded
    rows, and the stat's holdout MAE beside them as the reference the live
    figure is tracking against. Plus the daily MAE per stat, so the record
    reads as form over time rather than one settled number.
    """
    pre_tip = [r for r in rows if r["made_before_tip"]]

    per_stat: list[dict] = []
    for stat in STATS:
        srows = [r for r in pre_tip if r["stat"] == stat]
        if not srows:
            continue
        banded = [r for r in srows if r["grade"] is not None]
        shares = [
            {
                "grade": label,
                "n": sum(1 for r in banded if r["grade"] == label),
                "pct": round(100.0 * sum(1 for r in banded if r["grade"] == label) / len(banded), 1),
            }
            for label in GRADES
        ] if banded else []
        # Live coverage of the shipped 80% intervals over the same rows:
        # the share of the stat's counted pre-tip lines whose actual
        # landed inside the range that day's own calibration produced.
        # Rows with no interval (no tier could serve their day) are
        # counted in neither side; interval_n says how many were judged.
        interval_rows = [r for r in srows if r.get("inside_interval") is not None]
        per_stat.append(
            {
                "stat": stat,
                "n": len(srows),
                "mae": round(sum(abs(r["delta"]) for r in srows) / len(srows), 3),
                "bias": round(sum(r["delta"] for r in srows) / len(srows), 3),
                "holdout_mae": (bands or {}).get(stat),
                "band_n": len(banded),
                "bands": shares,
                "interval_coverage": (
                    round(
                        100.0
                        * sum(1 for r in interval_rows if r["inside_interval"])
                        / len(interval_rows),
                        1,
                    )
                    if interval_rows
                    else None
                ),
                "interval_n": len(interval_rows),
            }
        )

    by_day: dict[str, dict[str, list[float]]] = {}
    for r in pre_tip:
        by_day.setdefault(r["game_date"], {}).setdefault(r["stat"], []).append(abs(r["delta"]))
    daily = [
        {
            "date": date,
            "stats": {
                stat: {"n": len(misses), "mae": round(sum(misses) / len(misses), 3)}
                for stat, misses in sorted(stats.items())
            },
        }
        for date, stats in sorted(by_day.items())
    ]

    dates = [r["game_date"] for r in pre_tip]
    return {
        "basis": AGGREGATES_NOTE,
        "from_date": min(dates) if dates else None,
        "to_date": max(dates) if dates else None,
        "n_days": len(daily),
        "total": len(pre_tip),
        "per_stat": per_stat,
        "daily": daily,
    }


def build_prop_grading(
    db_path: Path,
    schedule: list[dict],
    bands: dict[str, float] | None,
    *,
    limit: int = SECTION_LIMIT,
    cold_start: dict[str, dict] | None = None,
) -> dict:
    """The graded player props for the most recent completed game day, plus
    the model's tracking aggregates over the whole counted record.

    Computed from stored data only: recorded predictions, recorded outcomes,
    the schedule, and the manifest's published MAEs. Nothing is recomputed
    from model files and nothing is fetched; callers cache or snapshot the
    result (the public snapshot carries it, and the hub route serves it
    behind the props cache's TTL discipline).

    `cold_start` is the manifest's holdout conformal block
    (`prop_conformal_cold_start`), the intervals' tier-3 calibration; the
    intervals themselves are attached from the counted rows before any
    section reads them, so the day, the rankings and the aggregates all
    carry the same ranges.
    """
    rows = _graded_rows(db_path, schedule, bands)
    infos = _attach_intervals(rows, cold_start)
    aggregates = _prop_aggregates(rows, bands)
    pre_tip_rows = [r for r in rows if r["made_before_tip"]]
    if pre_tip_rows:
        # The aggregates' own population is pre-tip; the calibration note
        # describes that population's latest day, not a later day graded
        # only on rows that cannot prove they were made in time.
        latest_date = max(r["game_date"] for r in pre_tip_rows)
        day_stats = sorted({r["stat"] for r in pre_tip_rows if r["game_date"] == latest_date})
        aggregates["interval_method"] = INTERVAL_METHOD
        aggregates["interval_level"] = INTERVAL_LEVEL
        aggregates["interval_calibration"] = _interval_note_for_day(
            infos, latest_date, day_stats
        )
    payload: dict = {
        "bands": bands,
        "counting": COUNTING_NOTE,
        "confidence_basis": CONFIDENCE_NOTE,
        "grade_basis": GRADE_NOTE,
        "latest_day": None,
        "most_confident": [],
        "biggest_surprises": [],
        "n_with_baseline": 0,
        "aggregates": aggregates,
    }
    if not rows:
        return payload

    latest = max(r["game_date"] for r in rows)
    day_rows = [r for r in rows if r["game_date"] == latest]

    by_game: dict[str, dict] = {}
    for r in day_rows:
        entry = by_game.setdefault(
            r["game_id"],
            {
                "game_id": r["game_id"],
                "game_date": latest,
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "rows": [],
            },
        )
        entry["rows"].append(r)
    stat_order = {stat: i for i, stat in enumerate(STATS)}
    for entry in by_game.values():
        entry["rows"].sort(key=lambda r: ((r["player_name"] or r["player_id"]), stat_order.get(r["stat"], 99)))
    games_out = [by_game[g["game_id"]] for g in schedule if g.get("game_id") in by_game]

    ranked = [r for r in day_rows if r["_abs_departure"] is not None]
    most_confident = sorted(
        ranked, key=lambda r: (-r["_abs_departure"], -r["_abs_delta"], r["player_name"] or r["player_id"])
    )[:limit]
    surprises = sorted(
        day_rows, key=lambda r: (-r["_abs_delta"], r["player_name"] or r["player_id"])
    )[:limit]

    payload["latest_day"] = {
        "date": latest,
        "n_graded": len(day_rows),
        "n_pre_tip": sum(1 for r in day_rows if r["made_before_tip"]),
        "games": [
            {**g, "rows": _publish(g["rows"])} for g in games_out
        ],
    }
    payload["most_confident"] = _publish(most_confident)
    payload["biggest_surprises"] = _publish(surprises)
    payload["n_with_baseline"] = len(ranked)
    return payload


def build_game_prop_grading(
    db_path: Path,
    schedule: list[dict],
    bands: dict[str, float] | None,
    game_id: str,
    *,
    limit: int = SECTION_LIMIT,
    cold_start: dict[str, dict] | None = None,
) -> dict | None:
    """One game's complete counted grading, for its post-game review.

    Every counted row for the game -- the review is where a reader asks how
    the props did in THIS match, so the day payload's latest-day window is
    the wrong shape for it. Rows that cannot prove they were made before
    tip-off are included and labelled, exactly as the day payload treats
    them. None when the game has no counted rows at all: an unplayed game,
    an unknown id, and a played game nobody projected are all the same
    absence to the review, and the review shows no section for it.

    Intervals are attached from the FULL counted population before the
    game is filtered out of it (the calibrator needs the record, not the
    game), so this game's ranges are the same ones the day payload and
    the aggregates carry.
    """
    all_rows = _graded_rows(db_path, schedule, bands)
    infos = _attach_intervals(all_rows, cold_start)
    rows = [r for r in all_rows if r["game_id"] == game_id]
    if not rows:
        return None
    stat_order = {stat: i for i, stat in enumerate(STATS)}
    rows.sort(key=lambda r: ((r["player_name"] or r["player_id"]), stat_order.get(r["stat"], 99)))
    surprises = sorted(
        rows, key=lambda r: (-r["_abs_delta"], r["player_name"] or r["player_id"])
    )[:limit]
    first = rows[0]
    return {
        "game_id": game_id,
        "game_date": first["game_date"],
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "n_graded": len(rows),
        "n_pre_tip": sum(1 for r in rows if r["made_before_tip"]),
        "rows": _publish(rows),
        "biggest_surprises": _publish(surprises),
        "bands": bands,
        "counting": COUNTING_NOTE,
        "grade_basis": GRADE_NOTE,
        "interval_method": INTERVAL_METHOD,
        "interval_level": INTERVAL_LEVEL,
        "interval_basis": _interval_note_for_day(
            infos, first["game_date"], sorted({r["stat"] for r in rows})
        ),
    }
