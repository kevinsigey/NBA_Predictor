"""Refuse to publish a snapshot built on stale models (spec section 10, gap G6).

The snapshot carried whatever manifest it was handed. On 2026-10-04 it happily
published a snapshot stamped fresh while serving models trained on 2026-09-18 --
the reader sees "generated today" and reasonably concludes the numbers behind it
are current. They were not, and nothing on the page said so.

The fix is the NFL pattern (`public_snapshot.py` L226-272): compare the model's
`trained_at` against now at publish time and refuse when it is too old.

`MAX_MODEL_AGE_DAYS` is a **named constant because Kevin decides it**. It is not
tuned so a particular model passes; `tests/test_snapshot_staleness.py` pins the
Phase A measurement against it so loosening it fails a test rather than quietly
reversing a conclusion.

A refusal raises rather than warns. The alternative is a snapshot with a
staleness banner on it, which is one more thing a reader has to notice, and the
whole point is that they would not.
"""

from __future__ import annotations

import gzip
import json
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba_predictor import config
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache
from nba_predictor.services.schedule_repository import load_schedule
from nba_predictor.tracking import store
from nba_predictor.tracking.prop_grading import build_prop_grading, prop_bands
from nba_predictor.tracking.store import init_db

#: How old a model may be at publish time. Fourteen days: the daily refresh runs
#: every 24h, so a model older than a fortnight means the refresh is broken --
#: which is exactly what happened for the 16 days between 2026-09-18 and
#: 2026-10-04, and nothing noticed.
MAX_MODEL_AGE_DAYS = 14

#: Clock-skew tolerance for a `trained_at` in the future. A retrain and a
#: publish on machines whose clocks disagree can produce a stamp a few seconds
#: ahead; that is not the same as a stamp days ahead, which is either a clock
#: fault or a manifest written by something that never trained.
MAX_CLOCK_SKEW = timedelta(minutes=5)


class StaleModels(Exception):
    """The snapshot would publish numbers from a model that is too old.

    Carries the age so the log says how far past the line it is, which is the
    difference between "the refresh is broken" and "the threshold is wrong".
    """


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def model_age_days(manifest: dict, now: datetime | None = None) -> float | None:
    """Days since the manifest's `trained_at`, or None when there is no stamp.

    None is not 0.0: a manifest with no readable `trained_at` carries no evidence
    that it is current, and reporting it as brand new would wave exactly the
    failure this gate exists to catch straight through.
    """
    trained_at = _parse_ts((manifest or {}).get("trained_at"))
    if trained_at is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - trained_at).total_seconds() / 86400.0


def assert_models_fresh(
    manifest: dict | None,
    now: datetime | None = None,
    max_age_days: int = MAX_MODEL_AGE_DAYS,
) -> float:
    """Raise `StaleModels` unless the manifest's models are recent enough.

    Returns the age in days so a caller can log it, or None when there is no
    manifest at all.

    **No manifest is not stale.** A repository that has never trained publishes
    `"manifest": null` and the frontend shows a "not trained" state; refusing
    that would break a state the site legitimately has and that misleads no
    reader. What the gate exists for is the opposite case -- a manifest that
    looks current because the snapshot is fresh while the model behind it is
    not. An *undated* manifest is refused, because that one is indistinguishable
    from a fresh one.
    """
    if manifest is None:
        return None

    age = model_age_days(manifest, now)
    if age is not None and age < -MAX_CLOCK_SKEW.total_seconds() / 86400.0:
        # A future stamp yields a NEGATIVE age, which every "is it too old?"
        # comparison passes -- so a manifest dated next month would keep
        # publishing for a month plus the limit. Refuse rather than clamp.
        raise StaleModels(
            f"manifest trained_at={manifest.get('trained_at')} is in the future "
            f"({-age:.1f} days ahead). That is a clock fault or a manifest that "
            "was never really trained on; refusing rather than treating it as "
            "brand new."
        )
    if age is None:
        raise StaleModels(
            f"manifest has no readable trained_at (model_version="
            f"{manifest.get('model_version')!r}); refusing to publish, because a "
            "snapshot with an undated model is indistinguishable from a fresh one"
        )
    if age > max_age_days:
        raise StaleModels(
            f"models are {age:.1f} days old (trained_at="
            f"{manifest.get('trained_at')}), past the {max_age_days}-day limit; "
            "the daily refresh is not running. Refusing to publish a snapshot "
            "whose numbers would look current."
        )
    return age


def generate_snapshot(
    schedule_path: Path,
    hub_dir: Path,
    manifest_path: Path,
    db_path: Path,
    now: datetime | None = None,
    max_age_days: int = MAX_MODEL_AGE_DAYS,
) -> dict:
    """Build the snapshot, refusing to build one from stale models.

    The check is here rather than in `main()` so every caller gets it -- the CLI,
    and anything that imports this function.
    """
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    assert_models_fresh(manifest, now=now, max_age_days=max_age_days)
    schedule = load_schedule(schedule_path)

    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "schedule": schedule,
        "hub": {
            "teams": load_hub_cache(hub_dir / "teams.json"),
            "players": load_hub_cache(hub_dir / "players.json"),
            "rankings": load_hub_cache(hub_dir / "rankings.json"),
            "standings": load_hub_cache(hub_dir / "standings.json"),
        },
        "track_record": [record.model_dump() for record in compute_track_record(db_path, schedule)],
        # Per-pick player-prop grading for the most recent completed day,
        # computed here (publish time, stored data) rather than per request.
        # The props manifest sits beside the game manifest it was handed.
        "prop_grading": build_prop_grading(db_path, schedule, prop_bands(manifest_path.parent)),
        "manifest": manifest,
    }

def write_snapshot(snapshot: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))


def _manifest_total_sigma(manifest: dict | None) -> float | None:
    """The model's total residual sigma, per model version.

    The published manifest carries it under metrics.total.residual_sigma; it is
    the sigma the total market is priced from. Missing or unreadable means the
    total market cannot be priced from this snapshot.
    """
    try:
        return float((manifest or {}).get("metrics", {}).get("total", {}).get("residual_sigma"))
    except (TypeError, ValueError):
        return None


def _manifest_margin_sigma(manifest: dict | None) -> float | None:
    """The model's MARGIN residual sigma, per model version.

    The published manifest carries it under metrics.margin.residual_sigma. It is
    what a spread market is priced from: a spread rung needs P(margin > line),
    and that probability is computed from the margin's own sigma. A total's
    sigma cannot stand in for it -- they are different quantities measured on
    different scales -- so the margin's is published separately.

    Missing or unreadable means the spread market cannot be priced from this
    snapshot, and the field is left out rather than guessed.
    """
    try:
        return float((manifest or {}).get("metrics", {}).get("margin", {}).get("residual_sigma"))
    except (TypeError, ValueError):
        return None


def append_pick_history(
    schedule_path: Path,
    db_path: Path,
    history_path: Path,
    total_sigma: float | None,
    margin_sigma: float | None = None,
) -> int:
    """Append each game's current pick to the pick history.

    Append-only and never overwritten: a pick row is keyed by (game_id,
    created_at), so a later retrain adds a new row instead of replacing the one
    that existed when a price was logged. Every version the model has published
    stays in the file, which is what lets the ledger pair each price with the
    pick that preceded it.

    Returns the number of rows appended.
    """
    init_db(db_path)
    schedule = load_schedule(schedule_path)
    by_game = {g["game_id"]: g for g in schedule}

    existing: dict[tuple[str, str], dict] = {}
    if history_path.exists():
        try:
            with gzip.open(history_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("game_id") and row.get("created_at"):
                        existing[(str(row["game_id"]), str(row["created_at"]))] = row
        except (OSError, gzip.BadGzipFile):
            existing = {}

    new_rows: list[dict] = []
    with store.get_connection(db_path) as conn:
        for game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total in conn.execute(
            "SELECT game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total "
            "FROM predictions ORDER BY created_at"
        ):
            key = (str(game_id), str(created_at))
            if key in existing:
                continue
            game = by_game.get(game_id)
            if game is None:
                continue
            new_rows.append(
                {
                    "game_id": game_id,
                    "game_date": game.get("game_date"),
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "home_win_prob": home_win_prob,
                    "predicted_margin": predicted_margin,
                    "predicted_total": predicted_total,
                    "total_sigma": total_sigma,
                    # The margin's own sigma, for the spread market. Absent
                    # rather than zero when the manifest does not publish one.
                    **({"margin_sigma": margin_sigma} if margin_sigma is not None else {}),
                    "model_version": model_version,
                    "created_at": created_at,
                }
            )

    if not new_rows:
        return 0

    all_rows = list(existing.values()) + new_rows
    all_rows.sort(key=lambda r: str(r.get("created_at") or ""))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_name(history_path.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    tmp_path.replace(history_path)
    return len(new_rows)


INITIAL_LOOKBACK_DAYS = 14

PICK_FIELDS = (
    "game_id", "game_date", "home_team", "away_team", "home_win_prob", "predicted_margin", "predicted_total",
    "total_sigma", "margin_sigma", "model_version", "created_at",
)


def _read_history(history_path: Path) -> dict[tuple[str, str], dict]:
    existing: dict[tuple[str, str], dict] = {}
    if not history_path.exists():
        return existing
    try:
        with gzip.open(history_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("game_id") and row.get("created_at"):
                    existing[(str(row["game_id"]), str(row["created_at"]))] = row
    except (OSError, gzip.BadGzipFile):
        return {}
    return existing


def append_api_rows(history_path: Path, api_rows: list[dict]) -> int:
    """Append API pick rows to the history, append-only and keyed by (game_id, created_at); returns the rows added."""
    existing = _read_history(history_path)
    new_rows = []
    for row in api_rows:
        key = (str(row.get("game_id") or ""), str(row.get("created_at") or ""))
        if not key[0] or not key[1] or key in existing:
            continue
        new_rows.append({f: row[f] for f in PICK_FIELDS if row.get(f) is not None})
    if not new_rows:
        return 0
    all_rows = sorted(list(existing.values()) + new_rows, key=lambda r: str(r.get("created_at") or ""))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_name(history_path.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    tmp_path.replace(history_path)
    return len(new_rows)


#: The keys a served prop-grading payload must carry before it may stand in
#: for a locally computed one. Anything less is not the grading -- it is an
#: error page or a different endpoint wearing the grading's URL -- and
#: publishing it would replace an honest empty with a dishonest something.
_PROP_GRADING_KEYS = (
    "bands", "counting", "confidence_basis", "grade_basis",
    "latest_day", "most_confident", "biggest_surprises", "n_with_baseline",
)


def fetch_prop_grading(picks_api: str, timeout: float = 30.0) -> dict | None:
    """The live API's prop grading, or None when it has none to give.

    The grading is computed from the tracking store, and the snapshot build
    runs where that store does not exist (CI -- the real one lives on the
    VPS), so a locally built grading there is the empty payload forever:
    correct bands over no days, which is the failure this fetch exists to
    end. The pick history below solves the identical problem the identical
    way, by asking the live API for what only it can compute.

    A payload counts only if it carries every grading key AND a graded day:
    an API with no graded day yet has nothing better than the local empty,
    and swapping one empty for another would only disguise whose it was.
    `served_stale_seconds` is dropped: it describes the API process's cache
    state at fetch time, and the snapshot's own `generated_at` is the stamp
    a reader should date this copy by.
    """
    import urllib.request

    url = f"{picks_api.rstrip('/')}/hub/prop-grading"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    if not isinstance(data, dict) or any(key not in data for key in _PROP_GRADING_KEYS):
        return None
    if not isinstance(data.get("latest_day"), dict):
        return None
    if not isinstance(data.get("most_confident"), list) or not isinstance(data.get("biggest_surprises"), list):
        return None
    grading = dict(data)
    grading.pop("served_stale_seconds", None)
    return grading


def fill_prop_grading_from_api(snapshot: dict, picks_api: str) -> bool:
    """Replace the snapshot's grading with the API's, but only an empty one.

    A build that HAS the store (a VPS-local run) computed its grading from
    the same counted picks the track record counts, and that grading wins:
    the fetch fills a gap, it never overrules a measurement. Returns whether
    a substitution happened.
    """
    local = snapshot.get("prop_grading") or {}
    if local.get("latest_day") is not None:
        return False
    grading = fetch_prop_grading(picks_api)
    if grading is None:
        return False
    snapshot["prop_grading"] = grading
    return True


def fetch_and_append_picks(picks_api: str, history_path: Path, timeout: float = 30.0) -> int:
    """Fetch every pick the API has made since the last one in the history, and append them.

    `since` is the newest created_at already in the history (all of the API's rows when there is none): asking for
    rows since NOW, as an earlier version did, can only ever return nothing.
    """
    import urllib.parse
    import urllib.request

    existing = _read_history(history_path)
    # With no history yet the API would hand back every pick it ever stored (23k rows, mostly rebuilt backtests of
    # games long gone): start two weeks back, which is further than any price the ledger has seen.
    since = max((created for _, created in existing), default=(datetime.now(timezone.utc) - timedelta(days=INITIAL_LOOKBACK_DAYS)).isoformat())
    url = f"{picks_api.rstrip('/')}/picks/history" + (f"?since={urllib.parse.quote(since)}" if since else "")
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return append_api_rows(history_path, data.get("rows", []))


def main() -> None:
    init_db(config.TRACKING_DB_PATH)
    snapshot = generate_snapshot(
        schedule_path=config.DATA_DIR / "cache" / "schedule" / "games.json",
        hub_dir=config.DATA_DIR / "cache" / "hub",
        manifest_path=config.PROJECT_ROOT / "models" / "manifest.json",
        db_path=config.TRACKING_DB_PATH,
    )

    # In CI the tracking DB is empty (the real one lives on the VPS), so the
    # grading built above is the empty payload -- and before this fill, that
    # empty is what every snapshot published, forever, under correct bands
    # and the counting text. When PICKS_API names the live API, fill an
    # empty grading from the API's own served grading, the same remedy the
    # pick history below already uses. Fail soft: a failed fetch warns and
    # publishes the local grading as it was.
    picks_api = os.environ.get("PICKS_API")
    if picks_api and (snapshot.get("prop_grading") or {}).get("latest_day") is None:
        try:
            if fill_prop_grading_from_api(snapshot, picks_api):
                print("prop grading: filled from PICKS_API (the local store had no graded props)")
        except Exception as exc:  # noqa: BLE001 - warned, not raised
            warnings.warn(f"PICKS_API prop-grading fetch failed: {exc}")
            print(f"prop grading: PICKS_API fetch failed: {exc}")

    write_snapshot(snapshot, config.DATA_DIR / "public_snapshot.json")

    manifest = json.loads((config.PROJECT_ROOT / "models" / "manifest.json").read_text()) if (
        config.PROJECT_ROOT / "models" / "manifest.json"
    ).exists() else None
    appended = append_pick_history(
        schedule_path=config.DATA_DIR / "cache" / "schedule" / "games.json",
        db_path=config.TRACKING_DB_PATH,
        history_path=config.DATA_DIR / "picks_history.jsonl.gz",
        total_sigma=_manifest_total_sigma(manifest),
        margin_sigma=_manifest_margin_sigma(manifest),
    )

    # In CI the tracking DB is empty (the real one lives on the VPS), so the picks come from the live API's
    # /picks/history. Fail soft: a failed fetch warns and leaves the history as it was.
    if picks_api:
        try:
            added = fetch_and_append_picks(picks_api, config.DATA_DIR / "picks_history.jsonl.gz")
            print(f"picks history: appended {added} new pick rows from PICKS_API")
        except Exception as exc:  # noqa: BLE001 - warned, not raised
            warnings.warn(f"PICKS_API fetch failed: {exc}")
            print(f"picks history: PICKS_API fetch failed: {exc}")

    if appended:
        print(f"picks history: appended {appended} new pick rows")


if __name__ == "__main__":
    main()
