import gzip
import json
from datetime import datetime, timezone

from nba_predictor.data import kalshi_lines as kl

HURTS = {
    "ticker": "KXNBAPTS-26OCT22BOSNYK-NYKJBRUNSON11-25", "event_ticker": "KXNBAPTS-26OCT22BOSNYK",
    "title": "Jalen Brunson: 25+ points", "floor_strike": 24.5, "status": "active", "result": "",
    "yes_bid_dollars": "0.0500", "yes_ask_dollars": "0.0600", "no_bid_dollars": "0.9400", "no_ask_dollars": "0.9500",
    "last_price_dollars": "0.0000", "volume_fp": "12.00", "open_interest_fp": "3.00",
    "occurrence_datetime": "2026-10-18T20:00:00Z", "close_time": "2026-10-20T17:00:00Z", "expiration_value": "",
}


def test_normalize_keeps_the_real_threshold_and_prices_as_probabilities():
    row = kl.normalize(HURTS, "KXNBAPTS", "2026-10-15T12:00:00+00:00")
    assert row["player"] == "Jalen Brunson" and row["stat"] == "points"
    assert row["line"] == 24.5                       # "325+" is the over of 24.5, not of 325
    assert (row["yes_bid"], row["yes_ask"]) == (0.05, 0.06)
    assert row["result"] is None and row["status"] == "active"


def test_a_settled_market_carries_its_outcome():
    row = kl.normalize({**HURTS, "status": "settled", "result": "yes", "expiration_value": "331"}, "KXNBAPTS", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "331"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "A B: 5+ receptions", "floor_strike": 4.5}, "KXNBAREB", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None and row["player"] == "A B"


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params.get("cursor") == "p2":
            return _Resp({"markets": [HURTS], "cursor": ""})
        return _Resp({"markets": [HURTS], "cursor": "p2" if params["series_ticker"] == "KXNBAPTS" and params["status"] == "open" else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 15, 12, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXNBAPTS", "open") in statuses and ("KXNBAREB", "settled") in statuses
    assert any(c.get("cursor") == "p2" for c in s.calls)
    assert len(rows) == len(kl.SERIES) * 2 + 1                       # 4 series x (open, settled) + the extra page
    assert all(r["snapshot_ts"].startswith("2026-10-15") for r in rows)


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(HURTS, "KXNBAPTS", "t")]
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_nba_lines_2026-10-15.jsonl.gz"
    lines = gzip.open(p1, "rt").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["line"] == 24.5
    assert kl.write_snapshot([], tmp_path, now) is None


def test_game_level_rungs_keep_their_title_and_have_no_player():
    spread = {**HURTS, "title": "Oklahoma City wins by over 8.5 points", "floor_strike": 8.5, "ticker": "KXNBASPREAD-26OCT22OKCHOU-OKC8"}
    row = kl.normalize(spread, "KXNBASPREAD", "t")
    assert row["player"] is None and row["stat"] == "spread" and row["line"] == 8.5 and row["title"].startswith("Oklahoma City")
    total = kl.normalize({**HURTS, "title": "Full Game: Over 243.5 points scored", "floor_strike": 243.5}, "KXNBATOTAL", "t")
    assert total["player"] is None and total["stat"] == "total"


def test_normalize_records_the_size_resting_at_each_sides_ask():
    # Kalshi's public market list carries yes_ask_size_fp and yes_bid_size_fp;
    # NO's ask is the YES bid from the other side, so it rests on the YES bid size.
    market = {"title": "A B: 5+ receptions", "floor_strike": 4.5,
              "yes_ask_size_fp": "774.30", "yes_bid_size_fp": "1.00"}
    row = kl.normalize(market, "KXTEST", "2026-11-18T00:00:00+00:00")
    assert (row["yes_ask_size"], row["no_ask_size"]) == (774.3, 1.0)
    assert kl.normalize({"title": "A B: 5+ receptions"}, "KXTEST", "t")["yes_ask_size"] is None
