from datetime import date

import numpy as np
import pandas as pd

def season_of(game_date: str) -> int:
    """The starting year of the season a game belongs to.

    `2025-10-20` -> 2025, `2026-04-02` -> 2025. Same convention as
    `pipeline/retrain.py`, so "this season" means the same thing everywhere.
    """
    year = int(str(game_date)[:4])
    month = int(str(game_date)[5:7])
    return year if month >= 10 else year - 1


def carry_over_weight(games_into_season: int, *, window: int, weight: float) -> float:
    """How strongly a stale rolling value is pulled toward the pre-season mean.

    The shrinkage applied to a team's rolling value on its (k+1)-th game of a
    season: `weight` on the first game, decaying linearly to zero by `window`
    games, and zero from then on -- by then the rolling window holds only
    current-season games and there is nothing stale left to correct.
    """
    if games_into_season >= window:
        return 0.0
    return weight * (1.0 - games_into_season / window)


def compute_four_factors(
    fgm: float,
    fga: float,
    fg3m: float,
    tov: float,
    oreb: float,
    opp_dreb: float,
    fta: float,
) -> dict:
    if fgm is None or fga is None or fg3m is None or tov is None or oreb is None or opp_dreb is None or fta is None:
        return {"efg_pct": float("nan"), "tov_rate": float("nan"), "orb_pct": float("nan"), "ft_rate": float("nan")}
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga else 0.0
    tov_rate = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0
    orb_pct = oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else 0.0
    ft_rate = fta / fga if fga else 0.0
    return {
        "efg_pct": efg_pct,
        "tov_rate": tov_rate,
        "orb_pct": orb_pct,
        "ft_rate": ft_rate,
    }


def add_rolling_four_factors(
    games: pd.DataFrame,
    window: int = 10,
    carry_over_weight_value: float | None = None,
    league_mean: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rolling four-factors per team.

    **`carry_over_weight_value` is the season-boundary correction.** Without it
    (the default, and the only behaviour before this change) a team's first game
    of a season takes the plain mean of the PREVIOUS season's last `window` games
    -- a sample spanning a six-month offseason, handed to the model with the same
    confidence as ten fresh games.

    With a weight, those early-season values are regressed toward the league mean
    by `carry_over_weight(...)`, which decays to zero by `window` games so the
    correction does not persist. The league mean for each factor is taken from
    games BEFORE the season in question (or supplied in `league_mean`), so a
    regression never pulls toward a number that includes the games being
    predicted.

    The weight is expected to be fitted in walk-forward, not chosen here: see
    `models/evaluate/carry_over.py`.
    """
    games = games.sort_values(["team", "game_date"]).reset_index(drop=True)

    factor_rows = games.apply(
        lambda row: compute_four_factors(
            row["fgm"], row["fga"], row["fg3m"], row["tov"], row["oreb"], row["opp_dreb"], row["fta"]
        ),
        axis=1,
        result_type="expand",
    )
    games = pd.concat([games, factor_rows], axis=1)

    for factor in ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]:
        games[f"{factor}_roll"] = games.groupby("team")[factor].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    if carry_over_weight_value is None:
        return games

    if not 0.0 <= carry_over_weight_value <= 1.0:
        raise ValueError(
            f"carry-over weight must be in [0, 1], got {carry_over_weight_value!r}"
        )

    # Which season each row is in, and how many of THAT season's games this team
    # has already played. Both are known before the game is predicted.
    games["season"] = games["game_date"].map(season_of)
    games["games_into_season"] = games.groupby(["team", "season"]).cumcount()

    # The mean to regress toward: each factor's mean over every game strictly
    # before the season being predicted, so the target never contains the games
    # it is correcting.
    means = league_mean if league_mean is not None else _pre_season_means(games)

    for factor in ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]:
        col = f"{factor}_roll"
        target = games["season"].map(
            lambda s: means.get((factor, int(s)), np.nan)
        )
        # A season with no earlier games to average has nothing to regress
        # toward. Do NOT fall back to the current season's mean: that would
        # leak the current game's own box score into its feature via the
        # groupby mean. Instead, leave those rows uncorrected (w=0 effectively).
        # The carry_over_weight for games_into_season=0 is `weight` (full), so
        # we must ensure target is NaN for those rows; the regression formula
        # will then keep the original value.
        w = games["games_into_season"].map(
            lambda k: carry_over_weight(k, window=window, weight=carry_over_weight_value)
        )
        # Only apply regression where target is available (has a prior-season mean).
        mask = target.notna()
        # Shrink the roll toward the target by w: corrected = roll + w*(target - roll).
        # (The previous form, target + w*(roll - target), applied 1-w instead: with
        # w decaying to zero by `window`, the shrinkage GREW over the first window
        # of games and the value ended up pinned to the mean from game `window`
        # onward -- the inversion the 2026-10-20 audit flagged, still live here
        # because the weak tests below could not see it.)
        games.loc[mask, col] = games.loc[mask, col] + w[mask] * (target[mask] - games.loc[mask, col])
        # Rows without a target keep their original rolling value (no regression).

    return games


def _pre_season_means(games: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Mean of each factor over all games strictly before each season.

    Keyed by (factor, season) so a season's own games can never contribute to
    the value its early games are pulled toward.
    """
    out: dict[tuple[str, int], float] = {}
    seasons = sorted(games["season"].unique())
    factors = ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]
    for season in seasons:
        earlier = games[games["season"] < season]
        for factor in factors:
            series = earlier[factor]
            if len(series):
                out[(factor, int(season))] = float(series.mean())
    return out
