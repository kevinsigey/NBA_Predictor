"""Player props: the COUNTED pick per player+stat, and the resolved MAE.

The MAE is computed from RESOLVED rows only -- rows with both a prediction and
a recorded actual. It grades the same pick the track record counts: the
EARLIEST recorded one for that (game, player, stat), whenever it was made
(`track-record-counts-every-pick`, predictor-hub #66, 2026-10-01). A backtest
row -- a prediction written after the game was played and scored against that
game's result -- is a recorded pick, so it counts, and it is labelled
`made_before_tip: False` rather than dropped.

Before that decision this module graded "the newest row made before tip-off,
else the newest, flagged rebuilt" and skipped the rebuilt ones outright. The
docstring here used to argue for that exclusion by appealing to CFB's track
record. It no longer does, because CFB's track record reversed too (CFB #27):
the exclusion was not accuracy, it was a record that stopped tracking.

**What did not change, deliberately:** `picks_by_player_stat` -- what the
site shows beside a box score -- still serves the newest pre-tip row, because
a projection printed next to a game's result has to be the freshest thing the
model said about it, not the first. The record and the display answer two
different questions and are built from two different helpers, each documented
at its own definition.
"""

from pathlib import Path

from nba_predictor.tracking import store
from nba_predictor.tracking.timing import (
    earliest_recorded,
    earliest_recorded_outcome,
    latest_by_instant,
    latest_pre_tip,
    made_before_tip,
)


def _earliest_outcome(rows: list) -> dict[tuple[str, str], float]:
    """{(player_id, stat): actual_value}, the EARLIEST recorded row per key.

    The reader half of `uq_game_player_outcomes_key`. It has to key on exactly
    the triple the unique index enforces -- (game_id, player_id, stat), and
    `_counting_key` is the shared definition -- because a reader deduping on a
    key the schema does not cover is a silent correctness hole, not a
    second layer of safety.

    Before this it was a bare dict comprehension over the rows as SQLite
    returned them: no dedupe and no ORDER BY, so on a database holding two rows
    for one key the MAE was graded against whichever came back last. One row
    per key is the normal case and this returns the input unchanged there, so
    the fix changes nothing on a clean file; it only makes the legacy,
    pre-unique-index case deterministic instead of arbitrary.
    """
    return {(row["player_id"], row["stat"]): row["actual_value"]
            for row in earliest_recorded_outcome(rows)}


def _picks_from_rows(rows, game: dict) -> dict[tuple[str, str], dict]:
    """{(player_id, stat): (pick_row, rebuilt)} from one game's snapshot rows.

    The serving rule, unchanged: the newest row made before tip-off, else the
    newest row at all with `rebuilt` set. `rebuilt` means "this row was written
    after the game started", which is a statement about time, not about
    counting -- the track record counts rebuilt rows and says so per pick.
    """
    by_key: dict[tuple[str, str], list] = {}
    for row in rows:
        by_key.setdefault((row["player_id"], row["stat"]), []).append(row)

    picks = {}
    for key, key_rows in by_key.items():
        pick = latest_pre_tip(key_rows, game)
        rebuilt = pick is None
        picks[key] = (pick if pick is not None else latest_by_instant(key_rows), rebuilt)
    return picks


def picks_by_player_stat(db_path: Path, game_id: str, game: dict) -> dict[tuple[str, str], dict]:
    """{(player_id, stat): (pick_row, rebuilt)} for one game -- the DISPLAYED pick."""
    return _picks_from_rows(store.get_player_predictions_for_game(db_path, game_id), game)


def actuals_by_player_stat(db_path: Path, game_id: str) -> dict[tuple[str, str], float]:
    return _earliest_outcome(store.get_player_outcomes_for_game(db_path, game_id))


# SQLite caps the number of bound parameters per statement; stay well under it.
_IN_CHUNK = 500


def resolved_player_props(db_path: Path, schedule: list[dict]) -> list[dict]:
    """Every (stat, prediction, actual) that was actually gradeable, one per
    (game, player, stat) -- the EARLIEST recorded prediction, whenever it was
    made.

    The counting key is (game, PLAYER, stat), not (game, stat): a points pick
    is one pick per player per game, and collapsing a game's rotation into a
    single pick would delete the player record rather than deduplicate it.
    `test_player_props_are_keyed_by_game_player_and_stat_not_by_game_alone`
    puts two players on one stat in one game and requires both to survive,
    because the collapse is silent -- the MAE would still be a number, just
    the wrong one.

    A rerun of the same player prop on the same game is history: it stays in
    `player_prediction_snapshots` and is not graded. Otherwise re-running the
    models until the numbers looked right would be free.

    Every returned row carries `made_before_tip`, DERIVED from its own
    `created_at` against the game's tip-off as UTC instants and failing closed
    to False, so the caller can publish the headline MAE over every counted row
    and the honest pre-tip MAE over the subset without a second pass and
    without the two disagreeing about which rows are which.

    Only a game that already has a recorded outcome can contribute, and that is
    a handful of a season's ~1,760 scheduled games. So the outcomes are read
    once, and predictions are read only for those games, in bulk. The per-game
    version ran two queries for EVERY scheduled game (about 3,500) and took
    ~50 s on the live site.
    """
    games = {g.get("game_id"): g for g in schedule if g.get("game_id")}
    actuals_by_game: dict[str, dict[tuple[str, str], float]] = {}
    rows_by_game: dict[str, list] = {}
    with store.read_connection(db_path) as conn:
        # Grouped per game, then deduped by `_earliest_outcome`, so a duplicated
        # key in a file that predates the unique index resolves to the earliest
        # recorded row instead of to whatever SQLite scanned last.
        outcomes_by_game: dict[str, list] = {}
        for row in conn.execute(
            "SELECT game_id, player_id, stat, actual_value, recorded_at FROM game_player_outcomes"
        ):
            if row["game_id"] in games:
                outcomes_by_game.setdefault(row["game_id"], []).append(row)
        for game_id, outcome_rows in outcomes_by_game.items():
            actuals_by_game[game_id] = _earliest_outcome(outcome_rows)
        ids = list(actuals_by_game)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"SELECT * FROM player_prediction_snapshots WHERE game_id IN ({marks}) ORDER BY created_at",
                chunk,
            ):
                rows_by_game.setdefault(row["game_id"], []).append(row)

    resolved = []
    for game in schedule:  # schedule order, exactly as before
        game_id = game.get("game_id")
        if not game_id or game_id not in actuals_by_game:
            continue
        actuals = actuals_by_game[game_id]
        for pick in earliest_recorded(rows_by_game.get(game_id, [])):
            actual = actuals.get((pick["player_id"], pick["stat"]))
            if actual is None:
                continue
            resolved.append(
                {
                    "game_id": game_id,
                    "player_id": pick["player_id"],
                    "stat": pick["stat"],
                    "predicted_value": pick["predicted_value"],
                    "actual_value": actual,
                    "made_before_tip": made_before_tip(pick["created_at"], game),
                    "created_at": pick["created_at"],
                }
            )
    return resolved
