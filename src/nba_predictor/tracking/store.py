import logging
import os
import sqlite3
import threading
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Sequence

_CONNECT_TIMEOUT_SECONDS = 30

#: How long a READ connection waits for a lock, against the writer's 30s above.
#: A page read must not sit on a lock: `/games/week` reads once per game, and the
#: browser gives up at 15s (frontend REQUEST_TIMEOUT_MS), so the request has to
#: fail into "no pick" well before that. Writers keep the long timeout: a rescore
#: waiting out a reader is correct, a visitor waiting out a rescore is not.
_READ_TIMEOUT_SECONDS = 5.0

#: Set to a truthy value to open connections with SQLite's `nolock=1`, for the
#: Azure Files (SMB) mount where SQLite's fcntl locking is not honoured.
#:
#: It is OFF by default and must stay that way. Every connection used to set it
#: unconditionally, justified by "the container app is pinned to exactly one
#: replica, making this process the sole writer" -- which was false. The daily
#: `nba-rescore` systemd unit runs `docker exec ... refresh_full`, a SECOND
#: process in the same container writing the same file, and `_DB_LOCK` below is
#: a `threading.Lock`: threads, not processes. On 2026-10-26 that produced an
#: outage: the rescore died at `sqlite3.OperationalError: disk I/O error` during
#: a commit and left the file malformed, so every read raised "database disk
#: image is malformed" and the games page 500'd.
#:
#: The DB lives on a local ext4 volume now (compose binds ./volumes/nba-tracking),
#: where locking works. Turn this on only for SMB, and then only if you accept
#: that concurrent writers are your problem.
NOLOCK_ENV = "NBA_TRACKING_DB_NOLOCK"

_DB_LOCK = threading.Lock()


logger = logging.getLogger(__name__)


def _nolock_enabled() -> bool:
    return os.environ.get(NOLOCK_ENV, "").strip().lower() in {"1", "true", "yes"}


#: Set by API page reads (see `bounded_reads`): connections opened inside it wait
#: `_READ_TIMEOUT_SECONDS` for a lock instead of the writer's 30s.
_read_bound: ContextVar[bool] = ContextVar("tracking_read_bound", default=False)


@contextmanager
def bounded_reads():
    token = _read_bound.set(True)
    try:
        yield
    finally:
        _read_bound.reset(token)


def _connect(db_path: Path, timeout: float | None = None) -> sqlite3.Connection:
    if timeout is None:
        timeout = _READ_TIMEOUT_SECONDS if _read_bound.get() else _CONNECT_TIMEOUT_SECONDS
    query = "?nolock=1" if _nolock_enabled() else ""
    return sqlite3.connect(f"file:{db_path}{query}", uri=True, timeout=timeout)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    predicted_margin REAL NOT NULL,
    predicted_total REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS game_market_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    model_probability REAL NOT NULL,
    market_probability REAL,
    edge REAL,
    bookmaker TEXT,
    american_odds INTEGER,
    point REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_forecast_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_timing_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    american_odds INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS player_prediction_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    created_at TEXT NOT NULL,
    position TEXT
);

CREATE TABLE IF NOT EXISTS player_names (
    player_id TEXT PRIMARY KEY,
    player_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_player_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    actual_value REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS value_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    american_odds INTEGER NOT NULL,
    point REAL,
    model_probability REAL NOT NULL,
    market_probability REAL NOT NULL,
    edge REAL NOT NULL,
    snapshot_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS value_pick_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    outcome TEXT NOT NULL,
    settled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS value_closing_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    closing_american_odds INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);
"""

# The one row key this schema ENFORCES, and the only one it can.
#
# Every table above carries `id INTEGER PRIMARY KEY AUTOINCREMENT`. That is a
# PRIMARY KEY, but a surrogate one: it makes each ROW distinct and constrains
# nothing a reader can care about, so two rows can always claim the same
# business key. That was true of every table here until this index.
#
# It is applied to `game_player_outcomes` and ONLY to it, because that is the
# only one of the six whose row key is a FACT rather than a history:
#
#   * `game_player_outcomes` -- one player's actual stat line in one game. There
#     is exactly one truth, a rerun of the ingest cannot produce a second, and
#     the reader grades picks against it. It is the one table where a duplicate
#     silently moves an aggregate (see `player_props._earliest_outcome`, which
#     is the reader half of this same key).
#   * `predictions` and `player_prediction_snapshots` -- deliberately MANY rows
#     per key. Rule 2 of the 2026-10-01 spec (predictor-hub #66) keeps a later
#     rerun as history: it stays in the table and stays in the published
#     per-pick list, and `earliest_recorded` decides which one counts. The
#     scheduled refresh re-presents a rolling window of finished games every
#     day, so duplicates there are the product working, not a bug. A UNIQUE
#     index would not harden them, it would abort the daily ingest and take away
#     the history the rule reads.
#   * `game_market_predictions` -- several rows per (game, market) BY
#     CONSTRUCTION: `refresh_odds` writes both sides of every bookmaker at one
#     instant. Its counted unit is the run (`earliest_run`), not the row.
#   * `game_forecast_snapshots`, `odds_timing_snapshots` -- no writer and no
#     reader anywhere in `src/`, so there is nothing to protect and inventing a
#     key for them would be a claim about a table nothing uses.
#
# The reader key and this index key are the SAME triple, (game_id, player_id,
# stat). They must stay the same: the reader's dedupe is only a statement about
# the rule when the schema enforces the same key, and a reader keyed on a
# column the constraint does not cover is precisely the silent hole this closes
# (`_counting_key` in tracking/timing.py is the reader half).
_UNIQUE_INDEXES = (
    ("uq_game_player_outcomes_key", """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_game_player_outcomes_key
    ON game_player_outcomes (game_id, player_id, stat)
    """),
    ("uq_value_pick_outcome_key", """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_value_pick_outcome_key
    ON value_pick_outcomes (game_id, market, selection, bookmaker)
    """),
    ("uq_value_closing_line_key", """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_value_closing_line_key
    ON value_closing_lines (game_id, market, selection, bookmaker)
    """),
    # NOT indexed: value_picks. The odds refresher runs on a loop, so it
    # re-snapshots the same pick as often as the cadence allows. That is this
    # repo's standing decision for a prediction-shaped table -- keep the history,
    # count the earliest -- and the ledger reader does exactly what
    # `tracking.timing.earliest_recorded` does for the game-level tables. An
    # index here would have made the refresh loop fail once the loop was doing
    # its job.
)


class ConflictingOutcome(Exception):
    """A second, DIFFERENT actual value for a key already recorded.

    Raised instead of writing the row. Recorded outcomes are immutable (rule 1:
    recorded stays recorded) and an outcome is a fact, so a disagreeing value is
    not an update to apply -- it is a contradiction that a human has to see.
    Silently ignoring it (INSERT OR IGNORE) or silently letting scan order
    decide (no constraint, no dedupe on the read) are the two failures this
    module exists to prevent.
    """


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK, _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_point_column(conn)
        _ensure_player_position_column(conn)
        _ensure_unique_indexes(conn)


def _ensure_point_column(conn: sqlite3.Connection) -> None:
    """Adds `point` to a game_market_predictions table created before this
    column existed. CREATE TABLE IF NOT EXISTS above won't add it to an
    already-existing table, so this migration covers any DB file left over
    from a prior deploy."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(game_market_predictions)")}
    if "point" not in columns:
        conn.execute("ALTER TABLE game_market_predictions ADD COLUMN point REAL")
        conn.commit()


def _ensure_player_position_column(conn: sqlite3.Connection) -> None:
    """Adds `position` to a player_prediction_snapshots table created before
    this column existed. CREATE TABLE IF NOT EXISTS above won't add it to an
    already-existing table, so this migration covers any DB file left over
    from a prior deploy."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(player_prediction_snapshots)")}
    if "position" not in columns:
        conn.execute("ALTER TABLE player_prediction_snapshots ADD COLUMN position TEXT")
        conn.commit()


def _ensure_unique_indexes(conn: sqlite3.Connection) -> None:
    """Adds the enforced row key to any DB file created before it existed.

    SQLite has no `ALTER TABLE ... ADD CONSTRAINT`, so the constraint is a
    unique index, which enforces exactly what a UNIQUE table constraint does:
    a duplicate insert raises `sqlite3.IntegrityError` instead of landing.

    Two properties matter more than the happy path here, because `init_db` runs
    on EVERY boot (`api/app.py`) and in the public-snapshot build:

    * **Idempotent.** `IF NOT EXISTS`, so a second `init_db` is a no-op --
      `test_init_db_is_idempotent` already required that and still passes.
    * **It never takes the service down on a file it cannot fix.** A file
      written before this change can already hold two outcomes for one key, and
      the scheduled refresh re-presents a rolling window of finished games
      every day, so that is the expected shape of an old file rather than a
      corruption. `CREATE UNIQUE INDEX` refuses to build over existing
      conflicts, and rule 1 forbids deleting the rows that block it -- so the
      index is skipped, `conflicting_outcome_keys` still reports them, the
      reader dedupes, and a warning names the count instead of the constraint
      quietly pretending to be there. Silently skipping would be the exact
      degradation this closes: a schema that looks enforced and is not.
    """
    for name, sql in _UNIQUE_INDEXES:
        try:
            conn.execute(sql)
        except sqlite3.IntegrityError:
            conflicts = _conflicting_outcome_keys(conn)
            warnings.warn(
                f"could not build unique index {name}: {len(conflicts)} "
                "(game_id, player_id, stat) key(s) in game_player_outcomes "
                "already hold more than one row, so that index is NOT in "
                "place. Those rows are left alone (recorded stays recorded); "
                "the reader keeps the earliest one per key. "
                "`store.conflicting_outcome_keys` lists them.",
                RuntimeWarning,
                stacklevel=2,
            )
        conn.commit()


def _conflicting_outcome_keys(conn: sqlite3.Connection) -> list[tuple[str, str, str, int]]:
    """`conflicting_outcome_keys` on an already-open connection.

    Split out because `init_db` holds `_DB_LOCK` and `get_connection` takes it
    again -- `threading.Lock` does not re-enter, so the locked public wrapper
    must not be called from inside a locked migration.
    """
    return [
        (row[0], row[1], row[2], row[3])
        for row in conn.execute(
            """
            SELECT game_id, player_id, stat, COUNT(*) AS n_rows
            FROM game_player_outcomes
            GROUP BY game_id, player_id, stat
            HAVING COUNT(*) > 1
            ORDER BY game_id, player_id, stat
            """
        )
    ]


def conflicting_outcome_keys(db_path: Path) -> list[tuple[str, str, str, int]]:
    """`(game_id, player_id, stat, n_rows)` for every duplicated outcome key.

    Ordered so the answer is stable and quotable. Empty on a clean database,
    and empty on every database this version creates.
    """
    with read_connection(db_path) as conn:
        return _conflicting_outcome_keys(conn)


@contextmanager
def get_connection(db_path: Path):
    with _DB_LOCK:
        conn = _connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


@contextmanager
def read_connection(db_path: Path):
    """A connection for READS ONLY, without `_DB_LOCK`.

    `get_connection` holds the process-wide lock for the whole `with` body, so a
    reader that pulls a few hundred thousand rows (track record, prop grading)
    made every other DB read, `/games` included, queue behind it with no
    timeout: measured 10 s for a 0.04 s request. SQLite serialises readers and
    writers itself (the 5 s busy timeout in `bounded_reads`); the lock is only
    needed where this process does read-then-write. Never write through this.
    """
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_prediction(
    db_path: Path,
    *,
    game_id: str,
    created_at: str,
    model_version: str,
    home_win_prob: float,
    predicted_margin: float,
    predicted_total: float,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total),
        )
        conn.commit()
        return cur.lastrowid


def insert_predictions(db_path: Path, rows: Sequence[dict], attempts: int = 4) -> int:
    """Many team predictions in ONE transaction with a retry, like `insert_player_predictions` (same VPS disk-I/O failure)."""
    batch = [(r["game_id"], r["created_at"], r["model_version"], r["home_win_prob"], r["predicted_margin"], r["predicted_total"]) for r in rows]
    for attempt in range(attempts):
        try:
            with get_connection(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO predictions
                        (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
            return len(batch)
        except sqlite3.OperationalError as exc:
            if attempt == attempts - 1:
                raise
            logger.warning("bulk prediction insert failed (%s); retrying %d/%d", exc, attempt + 1, attempts - 1)
            time.sleep(2 * (attempt + 1))
    return 0


def get_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def insert_market_prediction(
    db_path: Path,
    *,
    game_id: str,
    market: str,
    selection: str,
    model_probability: float,
    market_probability: float | None,
    edge: float | None,
    bookmaker: str | None,
    american_odds: int | None,
    created_at: str,
    point: float | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_market_predictions
                (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_market_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_market_predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def insert_player_prediction(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    predicted_value: float,
    created_at: str,
    position: str | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO player_prediction_snapshots (game_id, player_id, stat, predicted_value, created_at, position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, predicted_value, created_at, position),
        )
        conn.commit()
        return cur.lastrowid


def insert_player_predictions(db_path: Path, rows: Sequence[dict], attempts: int = 4) -> int:
    """Many player projections in ONE transaction (one connection, one commit), returning the number written.

    The per-row `insert_player_prediction` opens a connection and commits for every row; a scoring run writes ~13,000,
    and on the VPS's overlay filesystem two runs died part-way with `sqlite3.OperationalError: disk I/O error`. One
    transaction is the same data with a thousandth of the syncs. A transient OperationalError is retried (the whole
    batch, nothing half-written); a persistent one is raised, never swallowed.
    """
    batch = [(r["game_id"], r["player_id"], r["stat"], r["predicted_value"], r["created_at"], r.get("position")) for r in rows]
    for attempt in range(attempts):
        try:
            with get_connection(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO player_prediction_snapshots (game_id, player_id, stat, predicted_value, created_at, position)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
            return len(batch)
        except sqlite3.OperationalError as exc:
            if attempt == attempts - 1:
                raise
            logger.warning("bulk player-prediction insert failed (%s); retrying %d/%d", exc, attempt + 1, attempts - 1)
            time.sleep(2 * (attempt + 1))
    return 0


def record_player_names(db_path: Path, box_rows) -> int:
    """Remember player_id -> displayName from every ESPN box row seen.

    Merge-only: a row with an empty id or name is skipped, so a blank
    displayName never overwrites a known name; nothing is ever deleted.
    Lives in the tracking DB so it persists exactly as long as the picks do.
    """
    pairs = [
        (str(r["player_id"]), r["player_name"].strip())
        for r in box_rows
        if r.get("player_id") and (r.get("player_name") or "").strip()
    ]
    if not pairs:
        return 0
    with _DB_LOCK, _connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO player_names (player_id, player_name) VALUES (?, ?) "
            "ON CONFLICT(player_id) DO UPDATE SET player_name = excluded.player_name",
            pairs,
        )
    return len(pairs)


def load_player_names(db_path: Path) -> dict[str, str]:
    try:
        with read_connection(db_path) as conn:
            return {r["player_id"]: r["player_name"] for r in conn.execute("SELECT player_id, player_name FROM player_names")}
    except sqlite3.OperationalError:  # table not created yet on an old DB
        return {}


def get_player_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM player_prediction_snapshots WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def get_latest_prediction_for_game(db_path: Path, game_id: str) -> sqlite3.Row | None:
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at DESC LIMIT 1",
            (game_id,),
        )
        return cur.fetchone()


def insert_player_outcome(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    actual_value: float,
    recorded_at: str,
) -> int:
    """Records one player's actual stat line, and returns its row id.

    Idempotent on the same fact, and LOUD on a contradicting one.

    The scheduled refresh (`refresh-data.yml`) runs the ingest over a rolling
    60-day window every day, so this is called again for every recently finished
    game on every run. That is why the check is here rather than left to the
    caller: a plain INSERT would pile a duplicate outcome on top of the one
    already recorded, and nothing downstream could tell which of them the MAE
    was graded against.

    * **Same (game, player, stat), same value** -> the recorded row's id is
      returned and nothing is written. The fact is already on the record;
      writing it twice would add a row that changes no number.
    * **Same key, DIFFERENT value** -> raises `ConflictingOutcome`. Outcomes are
      immutable, so this is not an update to apply; it is a contradiction
      (`store_player_outcomes` counts and logs it rather than letting it abort
      the ingest). `INSERT OR IGNORE` would swallow it and `INSERT OR REPLACE`
      would rewrite a recorded pick -- the two silent outcomes this avoids.
    * **New key** -> inserted.

    Read-then-write inside the connection, not `INSERT ... ON CONFLICT DO
    NOTHING`, and that is deliberate. `get_connection` holds `_DB_LOCK` across
    the whole block, and this module's header records why that lock is enough
    (one replica, so this process is the sole writer), so there is no in-process
    window between the SELECT and the INSERT for the index to close. More to
    the point, SQLite rejects an `ON CONFLICT (game_id, player_id, stat)` target
    that matches no unique index, raising `OperationalError: ON CONFLICT clause
    does not match any PRIMARY KEY or UNIQUE constraint`. That is precisely the
    legacy-file case this module deliberately supports: where existing
    conflicting rows block the index and rule 1 forbids removing them,
    `_ensure_unique_indexes` leaves the index out -- and an upsert would then
    break every write to that file. This form works with or without the index,
    and on a file that has none it is the pre-check that still stops a new
    duplicate.

    The reader is unchanged by any of this and stays load-bearing:
    `player_props._earliest_outcome` still keeps the earliest row per key, so a
    file written before this index existed keeps grading deterministically.
    """
    with get_connection(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id, actual_value FROM game_player_outcomes
            WHERE game_id = ? AND player_id = ? AND stat = ?
            ORDER BY id LIMIT 1
            """,
            (game_id, player_id, stat),
        ).fetchone()
        if existing is not None:
            if float(existing["actual_value"]) != float(actual_value):
                raise ConflictingOutcome(
                    f"game_player_outcomes already records "
                    f"{existing['actual_value']!r} for (game_id={game_id!r}, "
                    f"player_id={player_id!r}, stat={stat!r}) as row "
                    f"{existing['id']}; refusing to record {actual_value!r}. "
                    "Recorded outcomes are immutable."
                )
            return existing["id"]
        cur = conn.execute(
            """
            INSERT INTO game_player_outcomes (game_id, player_id, stat, actual_value, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, actual_value, recorded_at),
        )
        conn.commit()
        return cur.lastrowid


def get_player_outcomes_for_player(db_path: Path, player_id: str) -> list[sqlite3.Row]:
    """Every recorded outcome for one player, oldest row first.

    The drawer's last-ten-games chart reads the actuals from here: what a
    player actually did, as recorded, rather than what a cache says.
    """
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_player_outcomes WHERE player_id = ? ORDER BY recorded_at, id",
            (str(player_id),),
        )
        return cur.fetchall()


def get_player_outcomes_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with read_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_player_outcomes WHERE game_id = ? ORDER BY recorded_at, id",
            (game_id,),
        )
        return cur.fetchall()


def get_all_predictions(db_path: Path) -> list[sqlite3.Row]:
    """The latest prediction per game across the whole tracking DB (for settlement)."""
    with read_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT game_id, MAX(created_at) AS max_created_at
                FROM predictions
                GROUP BY game_id
            ) latest ON p.game_id = latest.game_id AND p.created_at = latest.max_created_at
            """
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Per-player prop ledger (spec section 7)
#
# `player_prediction_snapshots` and `game_player_outcomes` have existed in the
# schema with no writer and no reader, so the site has had no per-player graded
# record -- the prop counterpart of the game-level track record.
# ---------------------------------------------------------------------------


def write_prop_snapshot(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    predicted_value: float,
    created_at: str,
    position: str | None = None,
) -> bool:
    """Snapshot one pre-tip prop pick.

    Appends, and always returns True. This table deliberately carries **no**
    unique index: the daily refresh re-scores a rolling window on every run, so
    a repeated snapshot is written every day in production and is *not* a defect.
    It is kept as history and the earliest one is what counts --
    `read_prop_ledger` does the dedupe on the read, which is the convention
    `tests/test_tracking_unique_keys.py` pins for the game-level `predictions`
    table. Adding a unique index here would contradict that decision, so the
    immutability of a published pick is enforced by the reader, not the schema.

    `position` is recorded *here*, at snapshot time. A player's position can
    change between the pick and its settlement, and reading it later would
    attribute the pick to a role they did not hold when it was made.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO player_prediction_snapshots
                (game_id, player_id, stat, predicted_value, created_at, position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, predicted_value, created_at, position),
        )
        conn.commit()
        return True


def read_prop_ledger(db_path: Path, game_id: str) -> list[dict]:
    """Every snapshotted prop pick for a game, graded where an outcome exists.

    **The earliest snapshot per (player, market) wins.** The table keeps every
    snapshot as history, per the repo's standing decision that prediction tables
    carry no unique index; the pick that counts is the one published first,
    which is the prop counterpart of the earliest-pick rule the game-level track
    record already follows. Without this dedupe a rerun that re-scored a game
    would silently replace a published pre-tip line with a later one.

    An ungradeable pick reads `actual_value=None`, `hit=None`, `error=None`.
    Never `hit=False`: a pick with no recorded outcome is not a miss, and
    reporting it as one is the exact error the game-level track record is built
    to avoid.
    """
    with read_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.game_id, s.player_id, s.stat, s.predicted_value,
                   s.created_at, s.position, o.actual_value
            FROM player_prediction_snapshots AS s
            LEFT JOIN game_player_outcomes AS o
              ON o.game_id = s.game_id
             AND o.player_id = s.player_id
             AND o.stat = s.stat
            WHERE s.game_id = ?
            ORDER BY s.created_at, s.player_id, s.stat
            """,
            (game_id,),
        ).fetchall()

    earliest: dict[tuple[str, str], dict] = {}
    for row in rows:
        record = dict(row)
        actual = record["actual_value"]
        record["error"] = None if actual is None else abs(float(actual) - float(record["predicted_value"]))
        record["hit"] = None if actual is None else bool(actual >= record["predicted_value"])
        # Rows arrive in created_at order, so the first row seen for a key is the
        # earliest one and later snapshots do not displace it.
        earliest.setdefault((record["player_id"], record["stat"]), record)
    return list(earliest.values())


def normalise_player_name(name: str | None) -> str:
    """Casefolded, punctuation-stripped name for exact matching.

    Normalisation exists so that "J. Tatum" and "jayson tatum" are not two
    players. It is deliberately *not* fuzzy: no edit distance, no initials, no
    surname-only fallback.
    """
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.casefold() if ch.isalnum() or ch.isspace()).strip()


def join_player_ids(
    prop_names: Sequence[tuple[str, str]],
    espn_players: Sequence[dict],
) -> list[str]:
    """Map (name, team) prop rows onto ESPN player ids, dropping the unmatched.

    ESPN keys players by id; the sportsbook props feed keys them by name. That
    gap is closed here by exact match on the normalised (name, team) pair, and
    **only** that: an unmatched prop is logged and skipped, never fuzzy-matched.
    A wrong line attributed to a real player is worse than a missing line,
    which is the same asymmetry `api/availability.py` is built around.

    Team is part of the key, so a player who transferred does not inherit the
    row recorded under their old team.

    Returns ids in the order the prop rows were given.
    """
    index = {
        (normalise_player_name(p.get("player_name")), (p.get("team") or "").strip().upper()): p["player_id"]
        for p in espn_players
        if p.get("player_id")
    }
    resolved: list[str] = []
    for name, team in prop_names:
        key = (normalise_player_name(name), (team or "").strip().upper())
        player_id = index.get(key)
        if player_id:
            resolved.append(player_id)
        else:
            logger.warning(
                "props feed player %r/%r did not match any ESPN player id; skipped "
                "rather than fuzzy-matched", name, team,
            )
    return resolved


def get_recent_market_predictions(db_path: Path, since: str) -> list[sqlite3.Row]:
    """Market rows stamped at or after `since`, newest first.

    The freshness filter lives in the *reader* rather than only in the gate,
    because the odds refresher appends a new row for every book on every pass
    and the table has no uniqueness constraint: without this the gate would be
    handed a season of lines and would have to discard nearly all of them.
    `since` is an ISO-8601 timestamp string so this stays a plain SQL
    comparison, the same form `get_market_predictions_for_game` already uses.
    """
    with read_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM game_market_predictions
            WHERE created_at >= ?
            ORDER BY created_at DESC, id DESC
            """,
            (since,),
        )
        return cur.fetchall()
