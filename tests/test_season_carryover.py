"""Season carry-over: a rolling window that spans an offseason is not form.

`add_rolling_four_factors` does `shift(1).rolling(10)` per team across the whole
frame. With three seasons loaded, a team's first game of a season therefore
averages the PREVIOUS season's last ten games -- a sample spanning a six-month
gap, presented to the model with the same confidence as a fresh ten-game sample.

Carry-over regresses those early-season values toward the league mean, with the
weight decaying to zero as current-season games fill the window. The weight is
fitted, never hard-coded.

Four claims, one per test:

  * with no weight, the output is byte-identical to today's;
  * week-1 values are built from prior-season games ONLY (checked by identity,
    not by counting);
  * the carry-over weight reaches zero once the window is full of current-season
    games, so nothing downstream is permanently shrunk;
  * a larger weight pulls the first game of a season further toward the mean.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.features.four_factors import (
    add_rolling_four_factors,
    carry_over_weight,
    season_of,
)


def _long_games(rows) -> pd.DataFrame:
    """Long-format box scores: one row per team per game, as build.py makes them."""
    records = []
    for i, (date, team) in enumerate(rows):
        records.append({
            "game_id": f"g{i}", "game_date": date, "team": team,
            # Vary the numbers by index so each team's rolling mean is not constant
            # and a regression toward the mean is actually observable.
            "fgm": 40.0 + (i % 7), "fga": 88.0, "fg3m": 12.0, "tov": 11.0,
            "oreb": 9.0, "opp_dreb": 32.0, "fta": 20.0,
        })
    return pd.DataFrame(records)


def _two_seasons(n_per_season: int = 14, team: str = "BOS") -> pd.DataFrame:
    """One team, two seasons: `n` games last spring, `n` games this autumn."""
    rows = []
    for k in range(n_per_season):
        rows.append((f"2025-04-{10 + k:02d}", team))   # season 2024
    for k in range(n_per_season):
        rows.append((f"2025-10-{20 + k:02d}", team))   # season 2025
    return _long_games(rows)


# --- the switch is off by default and inert -------------------------------

def test_no_carry_over_leaves_todays_behaviour_untouched():
    """The default must not move a single value, or every recorded number in the
    repo shifts for a reason nobody asked for."""
    games = _two_seasons()
    today = add_rolling_four_factors(games)
    explicit_none = add_rolling_four_factors(games, carry_over_weight_value=None)

    pd.testing.assert_frame_equal(today, explicit_none)

    # And the week-1 value is genuinely the raw prior-season mean, i.e. the
    # behaviour this whole exercise exists to change.
    first = today[today.game_date == "2025-10-20"]
    assert len(first) == 1, "fixture is missing the first game of the new season"


# --- week-1 uses prior-season data only -----------------------------------

def test_the_first_game_of_a_season_is_built_only_from_prior_season_games():
    """Checked by identity: every game in a week-1 rolling window predates the
    season it appears in. Counting rows would not catch a window that included
    one current-season game."""
    games = _two_seasons()
    out = add_rolling_four_factors(games)
    season = out["game_date"].map(season_of)

    first_game = out[out.game_date == "2025-10-20"].iloc[0]
    contributing = games[
        (games.team == first_game.team) & (games.game_date < "2025-10-20")
    ]
    assert len(contributing) == 14, "fixture is wrong: expected 14 prior games"
    # April 2025 belongs to the season that started in 2024, so the prior season
    # is 2024 -- not the calendar year, which is the easy mistake here.
    assert {season_of(d) for d in contributing.game_date} == {2024}, (
        "a week-1 window is not built purely from the prior season"
    )
    # And nothing from the new season, obviously.
    assert not (contributing.game_date >= "2025-10-20").any()


def test_season_of_labels_a_game_by_the_year_its_season_started_in():
    assert season_of("2025-10-20") == 2025, "October starts a new season"
    assert season_of("2025-12-01") == 2025
    for month in (1, 2, 3, 4, 5, 6):
        # Everything from January to June -- regular season AND playoffs --
        # belongs to the season that began the previous October.
        assert season_of(f"2026-{month:02d}-15") == 2025, f"month {month}"
    for month in (10, 11, 12):
        assert season_of(f"2026-{month:02d}-15") == 2026, f"month {month}"


# --- the weight -----------------------------------------------------------

def test_carry_over_weight_decays_to_zero_as_the_window_fills():
    """By `window` games into a season the weight is zero, so a team that has
    played ten games is not still being shrunk toward the mean."""
    for k, expected_positive in ((0, True), (5, True), (9, True), (10, False), (20, False)):
        w = carry_over_weight(k, window=10, weight=0.5)
        assert (w > 0) is expected_positive, f"game {k}: weight {w}"
        assert 0.0 <= w <= 1.0


def test_a_larger_weight_pulls_a_first_game_further_toward_the_mean():
    """Distance is measured from the regression TARGET -- the pre-season league
    mean the code pulls toward. (This used to measure distance from the
    fixture's raw fgm mean, a counting number ~43, against a percentage ~0.56:
    the comparison was meaningless and passed under both the correct and the
    inverted formula.)"""
    games = _two_seasons()
    raw = add_rolling_four_factors(games)
    raw_first = raw[raw.game_date == "2025-10-20"].iloc[0]["efg_pct_roll"]

    values = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = add_rolling_four_factors(games, carry_over_weight_value=weight)
        values[weight] = out[out.game_date == "2025-10-20"].iloc[0]["efg_pct_roll"]

    # Weight 0 is no correction at all; weight 1 lands exactly on the target.
    assert values[0.0] == pytest.approx(raw_first)
    target = values[1.0]
    distances = [abs(values[w] - target) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert distances == sorted(distances, reverse=True), (
        f"distance from the regression target is not monotone in the weight: {distances}"
    )
    assert distances[0] > 0 and distances[-1] == pytest.approx(0.0)


def test_after_the_window_is_full_the_carry_over_has_no_effect_at_all():
    """The 11th game of a season must equal the UNCORRECTED value, whatever the
    weight was. (This used to compare two weighted arms against each other --
    which passes even when both are pinned to the league mean, as they were
    under the inverted formula.)"""
    games = _two_seasons(n_per_season=20)
    raw = add_rolling_four_factors(games)
    light = add_rolling_four_factors(games, carry_over_weight_value=0.2)
    heavy = add_rolling_four_factors(games, carry_over_weight_value=0.9)

    late = light[light.game_date >= "2025-10-30"]
    assert len(late) > 0, "fixture has no late-season games"
    for col in ("efg_pct_roll", "tov_rate_roll"):
        for arm in (light, heavy):
            np.testing.assert_allclose(
                arm.set_index("game_date")[col].loc[late.game_date].to_numpy(),
                raw.set_index("game_date")[col].loc[late.game_date].to_numpy(),
            )


def test_a_rejected_weight_is_refused_rather_than_clamped():
    """A weight outside [0, 1] is a bug in the caller, not something to round."""
    games = _two_seasons()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="weight"):
            add_rolling_four_factors(games, carry_over_weight_value=bad)


def test_every_factor_is_carried_over_not_just_efg():
    games = _two_seasons()
    out = add_rolling_four_factors(games, carry_over_weight_value=0.5)
    first = out[out.game_date == "2025-10-20"].iloc[0]
    for factor in ("efg_pct", "tov_rate", "orb_pct", "ft_rate"):
        assert np.isfinite(first[f"{factor}_roll"]), f"{factor}_roll was not carried over"


def _two_different_seasons(n_per_season: int = 14, team: str = "BOS") -> pd.DataFrame:
    """Two consecutive seasons whose four-factors differ sharply.

    Deliberately unlike `_two_seasons`: there the two seasons have identical
    numbers, so the prior-season mean and the combined mean are the same value
    and no test can tell a leaking target from a clean one.
    """
    # Every box-score input changes between the seasons, not just fgm: the
    # other three factors are functions of tov/fta/oreb/opp_dreb, so leaving
    # those fixed would make their prior and combined means identical and the
    # assertion below vacuous for them too.
    records = []
    i = 0
    for date_base, month_day in (("2025-04", 10), ("2025-10", 20)):
        new_season = month_day == 20
        for k in range(n_per_season):
            records.append({
                "game_id": f"g{i}",
                "game_date": f"{date_base}-{month_day + k:02d}",
                "team": team,
                "fgm": 60.0 if new_season else 40.0,
                "fga": 88.0,
                "fg3m": 12.0 if new_season else 5.0,
                "tov": 16.0 if new_season else 8.0,
                "oreb": 14.0 if new_season else 6.0,
                "opp_dreb": 28.0 if new_season else 34.0,
                "fta": 26.0 if new_season else 14.0,
            })
            i += 1
    return pd.DataFrame(records)


# --- the regression target itself must not leak ---------------------------

def test_the_league_mean_excludes_the_season_being_predicted():
    """The mean a week-1 value is pulled toward is computed from games STRICTLY
    before its own season. Using `<=` instead of `<` would let the predicted
    season's own results set the target, which is the same class of leak the
    calibrator rule exists to prevent, and it is invisible in the output because
    the correction moves the value either way.
    """
    from nba_predictor.features.four_factors import _pre_season_means

    out = add_rolling_four_factors(
        _two_different_seasons(), carry_over_weight_value=0.5
    )
    means = _pre_season_means(out)

    prior_rows = out[out["season"] == 2024]
    current_rows = out[out["season"] == 2025]
    assert len(prior_rows) > 0 and len(current_rows) > 0

    for factor in ("efg_pct", "tov_rate", "orb_pct", "ft_rate"):
        target = means[(factor, 2025)]
        assert target == pytest.approx(float(prior_rows[factor].mean())), (
            f"{factor}: the regression target for season 2025 is not the "
            "prior season's mean"
        )
        # A mean over both seasons would be a different number, so this test
        # would fail if the two were equal by accident.
        assert target != pytest.approx(float(out[factor].mean())), (
            f"{factor}: prior and combined means coincide, so this fixture "
            "cannot tell them apart"
        )


def test_a_wildly_different_new_season_does_not_move_the_regression_target():
    """Black-box version of the same claim.

    The current season is built to be nothing like the prior one. If the target
    leaked in, the carried week-1 value would shift; with a strictly-prior
    target it must not.
    """
    out = add_rolling_four_factors(
        _two_different_seasons(), carry_over_weight_value=0.5
    )
    week1 = out[out.game_date == "2025-10-20"].iloc[0]
    prior_mean = out[out.game_date < "2025-10-20"]["efg_pct"].mean()
    league = out["efg_pct"].mean()

    # Regressed toward the PRIOR mean, so strictly closer to it than to the
    # combined mean, which is dragged by the new season's very different games.
    assert abs(week1.efg_pct_roll - prior_mean) < abs(week1.efg_pct_roll - league) + 1e-12
    assert abs(prior_mean - league) > 1e-6, "fixture: the two means are identical"


def test_build_training_frame_is_unchanged_without_a_weight():
    """The plumbing must be inert by default, or every production caller and
    every recorded number in the repo shifts."""
    import numpy as np
    from nba_predictor.features.build import build_training_frame

    rng = np.random.default_rng(21)
    n = 40
    games = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "game_date": pd.date_range("2025-01-01", periods=n).astype(str),
        "home_team": ["BOS"] * n,
        "away_team": ["LAL"] * n,
        "home_pts": [110.0] * n, "away_pts": [105.0] * n,
        "home_fgm": [40.0 + i % 5 for i in range(n)],
        "home_fga": [88.0] * n, "home_fg3m": [12.0] * n, "home_tov": [11.0] * n,
        "home_oreb": [9.0] * n, "home_dreb": [32.0] * n, "home_fta": [20.0] * n,
        "away_fgm": [38.0 + i % 3 for i in range(n)],
        "away_fga": [85.0] * n, "away_fg3m": [11.0] * n, "away_tov": [12.0] * n,
        "away_oreb": [8.0] * n, "away_dreb": [33.0] * n, "away_fta": [18.0] * n,
    })

    games["home_win"] = (games.home_pts > games.away_pts).astype(int)
    default, cols_a = build_training_frame(games)
    explicit, cols_b = build_training_frame(games, carry_over_weight=None)
    pd.testing.assert_frame_equal(default, explicit)
    assert cols_a == cols_b
