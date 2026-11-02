"""Player-prop grading: how close each counted projection landed.

The track record already grades player props as one aggregate MAE per stat
(`services/hub_service.py::_summarize_player_props`). This is the per-pick
counterpart: every counted projection for the most recent completed game
day, laid against the player's actual box score, and graded in units of the
model's own error -- the PL predictor's fixture-grading shape, applied to
players instead of matches.

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
"""

from pathlib import Path

from nba_predictor.tracking import store
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


def build_prop_grading(
    db_path: Path,
    schedule: list[dict],
    bands: dict[str, float] | None,
    *,
    limit: int = SECTION_LIMIT,
) -> dict:
    """The graded player props for the most recent completed game day.

    Computed from stored data only: recorded predictions, recorded outcomes,
    the schedule, and the manifest's published MAEs. Nothing is recomputed
    from model files and nothing is fetched; callers cache or snapshot the
    result (the public snapshot carries it, and the hub route serves it
    behind the props cache's TTL discipline).
    """
    payload: dict = {
        "bands": bands,
        "counting": COUNTING_NOTE,
        "confidence_basis": CONFIDENCE_NOTE,
        "grade_basis": GRADE_NOTE,
        "latest_day": None,
        "most_confident": [],
        "biggest_surprises": [],
    }
    resolved = resolved_player_props(db_path, schedule)
    if not resolved:
        return payload

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

    def publish(row_list: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in row_list]

    payload["latest_day"] = {
        "date": latest,
        "n_graded": len(day_rows),
        "n_pre_tip": sum(1 for r in day_rows if r["made_before_tip"]),
        "games": [
            {**g, "rows": publish(g["rows"])} for g in games_out
        ],
    }
    payload["most_confident"] = publish(most_confident)
    payload["biggest_surprises"] = publish(surprises)
    payload["n_with_baseline"] = len(ranked)
    return payload
