"""The track-record cache on compute_track_record.

/hub/track-record was paying the settle-everything build on every
request (12s cold in production). The record changes only when the
tracking database does, so the cache is keyed on the database's own
write marker -- (mtime_ns, size) of the file, the signal
`routes._db_state` keys the MAE cache on -- plus the inputs that
shape the payload. The TTL is a backstop, not the mechanism, and a
failed recompute drops the entry and propagates, as the MAE cache
does: an expired figure is never served quietly.

The compute is stubbed and the clock injected, so every property is
tested without a real database build and without sleeping.
"""

from datetime import date
from pathlib import Path

import pytest

from nba_predictor.api.schemas import TrackRecordOut
from nba_predictor.services import hub_service

TODAY = date(2026, 11, 17)


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


@pytest.fixture(autouse=True)
def _clean_cache():
    hub_service.clear_track_record_cache()
    yield
    hub_service.clear_track_record_cache()


@pytest.fixture
def db(tmp_path) -> Path:
    path = tmp_path / "tracking.db"
    path.write_bytes(b"v1")
    return path


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_compute(db_path, schedule=None, today=None, *, include_picks=True):
        seen.append(include_picks)
        return [TrackRecordOut(
            market="game_outcome", total_predictions=len(seen),
            correct_predictions=0, hit_rate=0.0, settled=True,
        )]

    monkeypatch.setattr(hub_service, "_compute_track_record", fake_compute)
    return seen


def test_a_second_call_on_an_unchanged_database_does_not_recompute(db, calls):
    clock = Clock()
    first = hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    second = hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    assert calls == [False]
    assert first == second


def test_a_database_write_invalidates_on_its_own_marker(db, calls):
    clock = Clock()
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    db.write_bytes(b"v2 -- the ingest landed")
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    assert calls == [False, False]


def test_the_ttl_is_only_a_backstop(db, calls):
    clock = Clock()
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    clock.advance(hub_service.TRACK_RECORD_CACHE_TTL_SECONDS + 1)
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    assert calls == [False, False]


def test_the_picks_and_summary_variants_cache_separately(db, calls):
    clock = Clock()
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    hub_service.compute_track_record(db, [], TODAY, include_picks=True, now=clock)
    hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    assert calls == [False, True]


def test_a_failed_recompute_propagates_and_is_not_cached(db, monkeypatch):
    def boom(db_path, schedule=None, today=None, *, include_picks=True):
        raise RuntimeError("db locked")

    monkeypatch.setattr(hub_service, "_compute_track_record", boom)
    with pytest.raises(RuntimeError):
        hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=Clock())
    assert hub_service._TRACK_RECORD_CACHE == {}


def test_a_caller_cannot_mutate_the_cached_rows(db, calls):
    clock = Clock()
    first = hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    first[0].total_predictions = 999999
    second = hub_service.compute_track_record(db, [], TODAY, include_picks=False, now=clock)
    assert second[0].total_predictions == 1
    assert calls == [False]


def test_the_warm_fills_the_summary_and_never_raises(db, calls, monkeypatch):
    assert hub_service.warm_track_record_cache(db, [], now=Clock()) is True
    assert calls == [False]

    def boom(db_path, schedule=None, today=None, *, include_picks=True):
        raise RuntimeError("db locked")

    monkeypatch.setattr(hub_service, "_compute_track_record", boom)
    hub_service.clear_track_record_cache()
    assert hub_service.warm_track_record_cache(db, [], now=Clock()) is False
