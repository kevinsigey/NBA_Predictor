import logging

import pandas as pd
from datetime import date

from nba_predictor.features.context import current_streak, game_flags, is_high_altitude
from nba_predictor.features.four_factors import add_rolling_four_factors
from nba_predictor.features.rest_travel import (
    compute_rest_days,
    congestion_flags,
    fatigue_index,
    is_back_to_back,
    is_high_altitude_home,
    rest_features,
    rolling_travel_miles,
    timezone_change_count,
)
from nba_predictor.features.availability import AVAILABILITY_COLUMNS, compute_availability_by_season
from nba_predictor.features.team_features import compute_h_features

logger = logging.getLogger(__name__)


#: How many days of travel a team is carrying into a game. Seven is the week,
#: which is what a road trip is: two games in three nights, a flight home, a
#: home stand. A longer window makes this a season total with a week's name.
TRAVEL_WINDOW_DAYS = 7

#: The travel / congestion / fatigue CANDIDATE (audit follow-up Task 4(b)). They are computed for every frame so
#: the comparison tool can use them, but they are NOT in the default model contract: the candidate loses its
#: evaluation (docs/nba-parity-evaluation-2026-10.md), so a retrain must not pick them up by default.
#: `build_feature_frame(..., include_travel_fatigue=True)` opts in.
TRAVEL_FATIGUE_COLUMNS = [
    "home_travel_miles", "away_travel_miles",
    "home_timezone_changes", "away_timezone_changes",
    "home_three_in_four", "away_three_in_four",
    "home_four_in_six", "away_four_in_six",
    "home_fatigue_index", "away_fatigue_index",
]

#: NBA-4 rest / streak / altitude candidate. OFF by default: the committed models were fitted on the role-split
#: rest, the 99 sentinel and the role-split streak, and switching them silently would move live inputs without a
#: retrain. `build_feature_frame(..., rest_altitude=True)` opts in; the evaluation is docs/nba-4-rest-altitude.md.
REST_ALTITUDE_ENABLED = False

#: Columns the flagged path adds. Rest, streak and is_high_altitude keep their names; only their values change.
REST_ALTITUDE_COLUMNS = ["home_first_game", "away_first_game"]
#: Visiting-team load interactions (rest_altitude only): back-to-back x 7-day travel miles, and a visitor on a
#: back-to-back playing at altitude.
REST_ALTITUDE_INTERACTION_COLUMNS = [
    "home_b2b_travel_miles", "away_b2b_travel_miles", "away_b2b_at_altitude",
]

FEATURE_COLUMNS = [
    "home_efg_pct_roll", "home_tov_rate_roll", "home_orb_pct_roll", "home_ft_rate_roll",
    "away_efg_pct_roll", "away_tov_rate_roll", "away_orb_pct_roll", "away_ft_rate_roll",
    "elo_diff_s",
    "home_adj_off_s", "away_adj_off_s",
    "home_adj_def_s", "away_adj_def_s",
    "home_pace_roll_s", "away_pace_roll_s",
    "home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back",
    "home_streak", "away_streak",
    "is_high_altitude", "conference_game", "division_game",
    # Lagged rotation availability (features/availability.py); needs player box scores.
    *AVAILABILITY_COLUMNS,
]

#: Columns that are absent from the incoming frame and are meant to be, filled
#: with 0.0 so a caller that does not have the data still gets a usable frame.
#:
#: `home_fatigue_index` was in this list AND in FEATURE_COLUMNS, which is how it
#: stayed a constant zero for so long: the zero-fill made the column exist, and
#: nothing ever checked that it moved. It is computed now, so it is not here.
_OPTIONAL_COLUMNS_DEFAULT_ZERO = []


def _long_format_box_scores(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = games.rename(
        columns={
            "home_team": "team", "home_fgm": "fgm", "home_fga": "fga", "home_fg3m": "fg3m",
            "home_tov": "tov", "home_oreb": "oreb", "away_dreb": "opp_dreb", "home_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    away_rows = games.rename(
        columns={
            "away_team": "team", "away_fgm": "fgm", "away_fga": "fga", "away_fg3m": "fg3m",
            "away_tov": "tov", "away_oreb": "oreb", "home_dreb": "opp_dreb", "away_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_feature_frame(
    games: pd.DataFrame, carry_over_weight: float | None = None, include_travel_fatigue: bool = False,
    rest_by_team: bool = False, players: pd.DataFrame | None = None,
    rest_altitude: bool = REST_ALTITUDE_ENABLED, rest_altitude_interactions: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Features for `games`.

    `players` is the per-player box-score frame (`ingest.to_player_training_frame`) behind the
    availability columns. None means no player history: those columns take the documented
    no-history default (0.0), with a warning in the log, so a caller that forgets it is visible.

    `carry_over_weight` is passed to `add_rolling_four_factors`; None (the
    default, and every production caller) is today's behaviour exactly.

    `rest_altitude=True` is the NBA-4 candidate: rest and streak per TEAM (from the team's own previous game,
    any role), rest capped at REST_CAP_DAYS with an is_first_game flag instead of the 99 sentinel, and the
    altitude indicator covering Utah as well as Denver. `rest_altitude_interactions=True` adds the visitor
    back-to-back interactions on top. Both are OFF by default and change nothing else.
    """
    if rest_altitude_interactions and not rest_altitude:
        raise ValueError("rest_altitude_interactions requires rest_altitude=True")
    games = games.copy()

    # Compute H team features (elo, adjusted ratings, pace) in a single chronological pass
    h_features = compute_h_features(games)
    
    # Only merge columns that don't already exist (preserve precomputed values)
    h_cols_to_merge = [c for c in h_features.columns if c not in games.columns or c == "game_id"]
    if len(h_cols_to_merge) > 1:  # more than just game_id
        games = games.merge(h_features[h_cols_to_merge], on="game_id", how="left")

    if not set(AVAILABILITY_COLUMNS) <= set(games.columns):
        if players is None:
            logger.warning("no player box scores supplied: availability features take their no-history default (0.0)")
        games = games.merge(compute_availability_by_season(games, players), on="game_id", how="left")

    long_form = add_rolling_four_factors(
        _long_format_box_scores(games), carry_over_weight_value=carry_over_weight
    )
    rolled = long_form.set_index(["game_id", "team"])[
        ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]
    ]

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        merged = games.merge(
            rolled.reset_index(),
            left_on=["game_id", team_col],
            right_on=["game_id", "team"],
            how="left",
        )
        for factor in ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]:
            games[f"{side}_{factor}"] = merged[factor].values

    games = games.sort_values("game_date").reset_index(drop=True)

    #: Per team, in date order: the dates it played and the ABBREVIATION of the
    #: venue it played in. A home game's venue is the team itself; a road game's
    #: is the opponent. `rolling_travel_miles` and `timezone_change_count` read
    #: consecutive pairs of these, so the order matters and must be date order.
    team_game_dates: dict[str, list[str]] = {}
    team_game_venues: dict[str, list[str]] = {}

    home_last_game: dict[str, str] = {}
    away_last_game: dict[str, str] = {}
    #: ONE last-game date per team, home or away. The fatigue candidate needs the team's real rest: DEN at home
    #: on the 3rd and away on the 4th is on zero days' rest, which the role-split dicts above cannot see.
    team_last_game: dict[str, str] = {}
    home_results: dict[str, list[str]] = {}
    away_results: dict[str, list[str]] = {}
    rest_days_home, rest_days_away = [], []
    streak_home, streak_away = [], []
    travel_home, travel_away = [], []
    tz_home, tz_away = [], []
    congestion_home, congestion_away = [], []
    fatigue_home, fatigue_away = [], []
    first_home, first_away = [], []
    #: ALL of a team's results in date order, whatever the venue. The flagged streak reads this; the default
    #: streak reads the role-split home_results / away_results above, which is what the committed models saw.
    team_results: dict[str, list[str]] = {}

    for _, row in games.iterrows():
        home, away, game_date = row["home_team"], row["away_team"], row["game_date"]

        if rest_altitude:
            # NBA-4 candidate: the team's own previous game, whatever its role; capped; first game flagged.
            rh, fh = rest_features(game_date, team_last_game.get(home))
            ra, fa = rest_features(game_date, team_last_game.get(away))
            rest_days_home.append(rh)
            rest_days_away.append(ra)
            first_home.append(fh)
            first_away.append(fa)
            streak_home.append(current_streak(team_results.get(home, [])))
            streak_away.append(current_streak(team_results.get(away, [])))
        elif rest_by_team:
            streak_home.append(current_streak(home_results.get(home, [])))
            streak_away.append(current_streak(away_results.get(away, [])))
            # Rest from the team's last game whatever its role. OPT-IN: the default below is what every committed
            # model was fitted on, so changing it silently would shift live inputs before it has been evaluated.
            rest_days_home.append(compute_rest_days(game_date, team_last_game.get(home)))
            rest_days_away.append(compute_rest_days(game_date, team_last_game.get(away)))
        else:
            # Role-split: a home game's rest is measured from the team's last HOME game, an away game's from its
            # last AWAY game, so DEN at home on the 3rd and away on the 4th looks rested on the 4th.
            rest_days_home.append(compute_rest_days(game_date, home_last_game.get(home)))
            rest_days_away.append(compute_rest_days(game_date, away_last_game.get(away)))
            streak_home.append(current_streak(home_results.get(home, [])))
            streak_away.append(current_streak(away_results.get(away, [])))

        # Travel and congestion are built from each team's OWN games strictly
        # before this one -- the venue list is read before either side is
        # appended, so no game is ever its own history.
        for side, team, miles, tz, congested, fatigue in (
            ("home", home, travel_home, tz_home, congestion_home, fatigue_home),
            ("away", away, travel_away, tz_away, congestion_away, fatigue_away),
        ):
            rest = compute_rest_days(game_date, team_last_game.get(team))
            dates = team_game_dates.get(team, [])
            venues = team_game_venues.get(team, [])
            if len(dates) >= 1:
                window = [
                    venues[i] for i in range(len(dates))
                    if date.fromisoformat(dates[i]).toordinal()
                    >= date.fromisoformat(game_date).toordinal() - (TRAVEL_WINDOW_DAYS - 1)
                ]
                # The path ends at THIS game's venue (the home team's arena, known before tip-off, never
                # the result): the trip into the game being scored is part of the load it carries.
                window.append(home)
                miles.append(rolling_travel_miles(window))
                tz.append(timezone_change_count(window))
                flags = congestion_flags(dates, game_date)
                congested.append(flags)
                fatigue.append(
                    fatigue_index(miles[-1], tz[-1], rest)
                )
            else:
                miles.append(0.0)
                tz.append(0)
                congested.append({"three_in_four": False, "four_in_six": False})
                fatigue.append(fatigue_index(0.0, 0, rest))

        home_last_game[home] = game_date
        away_last_game[away] = game_date
        team_last_game[home] = game_date
        team_last_game[away] = game_date
        if pd.notna(row["home_win"]):
            home_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")
            away_results.setdefault(away, []).append("L" if row["home_win"] == 1 else "W")
            team_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")
            team_results.setdefault(away, []).append("L" if row["home_win"] == 1 else "W")

        # The venue of a game is the arena it is played in, which is the HOME
        # team's own. Recording the team instead would make every team's location
        # history a list of its own abbreviation, and haversine(X, X) is zero --
        # travel would compute as always 0 and look like a real measurement.
        team_game_dates.setdefault(home, []).append(game_date)
        team_game_dates.setdefault(away, []).append(game_date)
        team_game_venues.setdefault(home, []).append(home)
        team_game_venues.setdefault(away, []).append(home)

    games["home_rest_days"] = rest_days_home
    games["away_rest_days"] = rest_days_away
    games["home_back_to_back"] = [is_back_to_back(d) for d in rest_days_home]
    games["away_back_to_back"] = [is_back_to_back(d) for d in rest_days_away]
    games["home_streak"] = streak_home
    games["away_streak"] = streak_away
    games["home_travel_miles"] = travel_home
    games["away_travel_miles"] = travel_away
    games["home_timezone_changes"] = tz_home
    games["away_timezone_changes"] = tz_away
    games["home_fatigue_index"] = fatigue_home
    games["away_fatigue_index"] = fatigue_away
    games["home_three_in_four"] = [f["three_in_four"] for f in congestion_home]
    games["away_three_in_four"] = [f["three_in_four"] for f in congestion_away]
    games["home_four_in_six"] = [f["four_in_six"] for f in congestion_home]
    games["away_four_in_six"] = [f["four_in_six"] for f in congestion_away]

    if rest_altitude:
        games["home_first_game"] = first_home
        games["away_first_game"] = first_away
        games["is_high_altitude"] = games["home_team"].apply(is_high_altitude_home)
    else:
        games["is_high_altitude"] = games["home_team"].apply(is_high_altitude)
    flags = games.apply(lambda r: game_flags(r["home_team"], r["away_team"]), axis=1, result_type="expand")
    games["conference_game"] = flags["conference_game"]
    games["division_game"] = flags["division_game"]

    if rest_altitude_interactions:
        games["home_b2b_travel_miles"] = games["home_back_to_back"].astype(float) * games["home_travel_miles"]
        games["away_b2b_travel_miles"] = games["away_back_to_back"].astype(float) * games["away_travel_miles"]
        games["away_b2b_at_altitude"] = (
            games["away_back_to_back"].astype(float) * games["is_high_altitude"].astype(float))

    columns = FEATURE_COLUMNS + (TRAVEL_FATIGUE_COLUMNS if include_travel_fatigue else [])
    if rest_altitude:
        columns = columns + REST_ALTITUDE_COLUMNS
    if rest_altitude_interactions:
        columns = columns + REST_ALTITUDE_INTERACTION_COLUMNS
    games = games.dropna(subset=columns).reset_index(drop=True)
    return games, columns


def build_training_frame(
    games: pd.DataFrame, carry_over_weight: float | None = None, include_travel_fatigue: bool = False,
    rest_by_team: bool = False, players: pd.DataFrame | None = None,
    rest_altitude: bool = REST_ALTITUDE_ENABLED, rest_altitude_interactions: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    return build_feature_frame(games, carry_over_weight=carry_over_weight, include_travel_fatigue=include_travel_fatigue,
                               rest_by_team=rest_by_team, players=players, rest_altitude=rest_altitude,
                               rest_altitude_interactions=rest_altitude_interactions)
