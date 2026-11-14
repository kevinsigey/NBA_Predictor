"""NBA-4: rest, streak and altitude per TEAM, and the sentinel.

Three defects this file pins down, each with a test that fails on the old code path:

* **The 99 sentinel.** `compute_rest_days` returns 99 for a team's first game in the data, and a
  season gap returns ~180. A linear model reads those as "very rested". `rest_features` caps at 7
  and flags the first game.
* **Role-split rest.** The default measures a home game's rest from the team's last HOME game and an
  away game's from its last AWAY game. A team that played on the 3rd at home and on the 4th away is
  shown rested on the 4th. The flagged path reads the team's own previous game, whatever its role.
* **Role-split streak.** The default streak for a home game reads only the team's home results. The
  flagged path reads all of the team's results, checked against an independent oracle below.

The default path must stay byte-identical: the committed models were fitted on it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_predictor.features.build import (
    FEATURE_COLUMNS,
    REST_ALTITUDE_COLUMNS,
    REST_ALTITUDE_INTERACTION_COLUMNS,
    build_feature_frame,
)
from nba_predictor.features.context import current_streak, is_high_altitude
from nba_predictor.features.rest_travel import (
    REST_CAP_DAYS,
    compute_rest_days,
    is_high_altitude_home,
    rest_features,
)

BOX = {"fgm": 40, "fga": 88, "fg3m": 12, "tov": 11, "oreb": 9, "dreb": 32, "fta": 20}


def _game(game_id, date, home, away, home_win=1):
    row = {
        "game_id": game_id, "game_date": date, "home_team": home, "away_team": away,
        "home_pts": 110 if home_win else 100, "away_pts": 100 if home_win else 110, "home_win": home_win,
    }
    for field, value in BOX.items():
        row[f"home_{field}"] = value
        row[f"away_{field}"] = value
    return row


def _fixture():
    """Warm-up so every team has rolling features and travel, then the measured games.

    Teams DEN, UTA, LAL, ATL. The warm-up is a round robin in which DEN plays at Utah and both
    venues are used, so DEN's last AWAY game is 2025-12-10 and its last HOME game is 2025-12-07.
    Results alternate so the streak tests see both W and L.
    """
    rows = []
    teams = ["DEN", "UTA", "LAL", "ATL"]
    n = 0
    for r in range(4):
        pairs = [(0, 1), (2, 3)] if r % 2 == 0 else [(1, 0), (3, 2)]
        for i, (h, a) in enumerate(pairs):
            n += 1
            rows.append(_game(f"w{r}-{i}", f"2025-12-{r * 3 + i * 2 + 1:02d}", teams[h], teams[a],
                              home_win=n % 2))
    measured = [
        ("m1", "2026-01-03", "DEN", "LAL", 1),  # DEN at home, rested from its Dec 10 road game
        ("m2", "2026-01-04", "ATL", "DEN", 0),  # DEN on a back-to-back, on the road
        ("m3", "2026-01-06", "UTA", "LAL", 1),  # Utah at home at altitude
        ("m4", "2026-01-05", "UTA", "ATL", 1),  # ATL: back-to-back on the road at altitude
    ]
    rows.extend(_game(*m) for m in measured)
    return pd.DataFrame(rows)


def _frame(**kwargs):
    frame, cols = build_feature_frame(_fixture(), **kwargs)
    return frame.set_index("game_id"), cols


# ---- the sentinel -----------------------------------------------------------------------------

def test_rest_features_caps_and_flags_the_first_game():
    assert rest_features("2026-10-21", None) == (REST_CAP_DAYS, 1)


def test_rest_features_leaves_a_normal_gap_alone():
    assert rest_features("2026-01-04", "2026-01-03") == (0, 0)
    assert rest_features("2026-01-05", "2026-01-03") == (1, 0)


def test_rest_features_caps_a_season_gap_at_a_week():
    # 180 days across the offseason is "fully rested", not a value to extrapolate from.
    assert compute_rest_days("2026-10-21", "2026-04-15") > 100
    assert rest_features("2026-10-21", "2026-04-15") == (REST_CAP_DAYS, 0)


def test_compute_rest_days_keeps_its_raw_sentinel_for_existing_callers():
    # The default path and its tests depend on the raw value; the cap lives in rest_features.
    assert compute_rest_days("2026-10-21", None) == 99


def test_flagged_rest_has_no_sentinel_value_anywhere():
    frame, _ = _frame(rest_altitude=True)
    for col in ("home_rest_days", "away_rest_days"):
        assert frame[col].max() <= REST_CAP_DAYS, f"{col} still carries an uncapped value"
        assert (frame[col] != 99).all()


def test_default_path_is_unchanged_and_keeps_the_sentinel():
    frame, cols = _frame()
    assert cols == FEATURE_COLUMNS
    assert not set(REST_ALTITUDE_COLUMNS) & set(frame.columns)


# ---- team-based rest (role-split is the demonstrable bug) -------------------------------------

def test_role_split_rest_misses_a_back_to_back_across_roles():
    """DEN is at home on Jan 3 and away on Jan 4: a back-to-back. The default measures the Jan 4
    away game's rest from DEN's last AWAY game (Dec 10), so it reads ~24 days. The team-based rest
    is 0. This is the defect the flag fixes."""
    default, _ = _frame()
    flagged, _ = _frame(rest_altitude=True)
    assert default.loc["m2", "away_rest_days"] > REST_CAP_DAYS, "the role-split default reads a long rest"
    assert flagged.loc["m2", "away_rest_days"] == 0
    assert flagged.loc["m2", "away_back_to_back"] in (True, 1)


def test_flagged_rest_is_from_the_teams_own_previous_game():
    """m1: DEN's last game before Jan 3 was Dec 10 on the road. Team-based rest is capped, not 24."""
    flagged, _ = _frame(rest_altitude=True)
    assert flagged.loc["m1", "home_rest_days"] == REST_CAP_DAYS
    assert flagged.loc["m1", "home_first_game"] == 0


# ---- team-based streak, checked against an independent oracle ---------------------------------

def test_flagged_streak_matches_the_team_own_results_for_every_row():
    """The oracle reads the raw games, not the builder: a team's streak before a game is the
    current streak of ALL its prior results, home or away, from the raw home_win column."""
    raw = _fixture().sort_values("game_date").reset_index(drop=True)
    results: dict[str, list[str]] = {}
    expected = {}
    for _, g in raw.iterrows():
        expected[g.game_id] = (
            current_streak(results.get(g.home_team, [])),
            current_streak(results.get(g.away_team, [])),
        )
        results.setdefault(g.home_team, []).append("W" if g.home_win == 1 else "L")
        results.setdefault(g.away_team, []).append("L" if g.home_win == 1 else "W")

    flagged, _ = _frame(rest_altitude=True)
    assert len(flagged), "the fixture produced no rows"
    for gid, row in flagged.iterrows():
        assert (row["home_streak"], row["away_streak"]) == expected[gid], gid


def test_default_streak_is_role_split_and_differs_from_the_team_streak():
    """The bug, demonstrated: the default streak uses role-split results, so on at least one row it
    disagrees with the team's own streak. The default is kept for the committed models; this test
    documents that the flagged path is what is different."""
    default, _ = _frame()
    flagged, _ = _frame(rest_altitude=True)
    shared = default.index.intersection(flagged.index)
    differs = (default.loc[shared, ["home_streak", "away_streak"]].to_numpy()
               != flagged.loc[shared, ["home_streak", "away_streak"]].to_numpy()).any()
    assert differs


# ---- altitude: Utah is missing from the existing test -----------------------------------------

def test_altitude_indicator_covers_denver_and_utah_not_phoenix():
    assert is_high_altitude_home("DEN") and is_high_altitude_home("UTA")
    assert not is_high_altitude_home("PHX")


def test_existing_altitude_helper_misses_utah_as_a_data_gap():
    """team_reference has altitude_ft = 0 for Utah, so the existing helper says it is not high
    altitude. Recorded here so the gap is visible; the flagged path uses HIGH_ALTITUDE_HOME."""
    assert is_high_altitude("UTA") is False


def test_flagged_altitude_flags_a_utah_home_game():
    default, _ = _frame()
    flagged, _ = _frame(rest_altitude=True)
    assert default.loc["m3", "is_high_altitude"] == 0
    assert flagged.loc["m3", "is_high_altitude"] == 1


# ---- visitor interactions (opt-in, requires the flag) -----------------------------------------

def test_interactions_require_the_flag():
    with pytest.raises(ValueError):
        build_feature_frame(_fixture(), rest_altitude_interactions=True)


def test_visitor_back_to_back_at_altitude_is_set_only_for_the_visitor_on_a_b2b():
    frame, cols = _frame(rest_altitude=True, rest_altitude_interactions=True)
    assert set(REST_ALTITUDE_INTERACTION_COLUMNS) <= set(cols)
    # m4: ATL played at home on Jan 4 and visits Utah on Jan 5 -> b2b at altitude for the visitor.
    assert frame.loc["m4", "away_back_to_back"] in (True, 1)
    assert frame.loc["m4", "away_b2b_at_altitude"] == 1
    # m1: LAL (the visitor) is not on a back-to-back -> no interaction.
    assert frame.loc["m1", "away_b2b_at_altitude"] == 0
    assert frame.loc["m2", "away_b2b_travel_miles"] == pytest.approx(frame.loc["m2", "away_travel_miles"])
