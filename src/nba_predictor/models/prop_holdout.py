"""Chronological holdout for the four prop regressors (spec section 7, gap G4).

The repo shipped prop models whose only evidence was **in-sample** metrics —
`ingest.train_player_prop_models` sets `training={"in_sample_metrics": True}` and
says so in the manifest. A model scored on the rows it was fitted against cannot
fail its own evaluation. This is the holdout that has never existed.

Two things this gets right that a naive version does not, both borrowed from
`pipeline/retrain.py` rather than reinvented:

* **Holdout rows are built from train+holdout, then selected.** Rolling features
  are shift(1), so a holdout player's first game must see its *training* games.
  Building features from the holdout slice alone gives every holdout row a
  truncated history that no serving scenario would have.
* **Features are built once**, over the concatenation, and the target is then
  picked per market. Building them inside a four-market loop recomputes the same
  frame four times and invites the four markets to disagree about the split.

A market whose target column is absent raises rather than reporting `inf`. An
`inf` MAE reads as a real, terrible number in the report, and the honest
statement is "this market had no target column", which is a different fact.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.models.conformal_props import (
    conformal_half_width,
    fit_sigma_model,
    sigma_values,
)
from nba_predictor.models.evaluate.walk_forward import chronological_split

#: Below this many holdout rows the conformal cold-start block is not
#: measured: decile bins over a handful of residuals would publish a
#: volatility curve nobody should calibrate on. The block is None and the
#: serving cold start simply has one fewer tier until a retrain with a
#: real holdout publishes it.
CONFORMAL_MIN_HOLDOUT_ROWS = 100

#: market name -> the column it actually lives in. "threes" is the market's
#: public name; the box score calls it fg3m. One map, resolved once, rather than
#: a chain of conditionals re-deriving it per row.
MARKET_TARGETS = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "fg3m",
}

MARKETS = tuple(MARKET_TARGETS)


def _date_aligned_split(
    df: pd.DataFrame, holdout_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split whose cut lands **between** dates.

    `chronological_split` cuts on rows, which is wrong here: a player-game frame
    holds ~20 rows per game date, so a row-wise cut puts one game on both sides
    and a model can score a holdout game from a feature that saw it. This was not
    hypothetical -- on 3,104 real player-games the row cut produced
    `train_max == holdout_min == 2026-05-09`, and the assertion below refused it.

    Reuses `chronological_split` for the row target, then walks the cut back to
    the nearest preceding date boundary so no date is shared.
    """
    train, holdout = chronological_split(
        df, date_col="game_date", holdout_fraction=holdout_fraction
    )
    train_dates = set(train["game_date"])
    leaked = set(holdout["game_date"]) & train_dates
    if not leaked:
        return train, holdout

    # Move every shared date to the training side, so the holdout starts strictly
    # after. Loses those games from the holdout -- which is the point.
    return (
        df[df["game_date"].isin(train_dates | leaked)],
        df[~df["game_date"].isin(train_dates | leaked)],
    )


def prop_holdout_metrics(
    df: pd.DataFrame,
    factories: dict[str, Callable],
    holdout_fraction: float = 0.2,
    frame_builder: Callable = build_player_feature_frame,
) -> dict:
    """Holdout vs in-sample MAE per market on one chronological split.

    `df` is the player-game frame `ingest.to_player_training_frame` builds.
    `factories` maps market name to `(X, y) -> fitted model`.

    Every market is scored against the **same** split — one split, four targets.
    Four independent splits would let the four numbers describe four different
    periods and quietly stop being comparable to each other.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        raise ValueError("prop_holdout_metrics needs a player-game frame with at least 2 rows")
    if "game_date" not in df.columns:
        raise ValueError("player-game frame has no game_date; cannot split chronologically")

    train, holdout = _date_aligned_split(df, holdout_fraction)

    # One feature build over train+holdout, so holdout rolling features inherit
    # the training history a real prediction would have (see module docstring).
    combined = pd.concat([train, holdout], ignore_index=True)
    combined_frame, feature_cols = frame_builder(combined)
    holdout_ids = set(holdout["game_id"])

    train_frame = combined_frame[~combined_frame["game_id"].isin(holdout_ids)]
    holdout_frame = combined_frame[combined_frame["game_id"].isin(holdout_ids)]

    if len(train_frame) == 0 or len(holdout_frame) == 0:
        raise ValueError(
            f"split left {len(train_frame)} train / {len(holdout_frame)} holdout rows; "
            "each player needs at least one prior game for the rolling features"
        )

    train_max = str(train["game_date"].max())
    holdout_min = str(holdout["game_date"].min())
    if train_max >= holdout_min:
        raise AssertionError(
            f"holdout is not strictly after train: {train_max} >= {holdout_min}"
        )

    results: dict[str, dict] = {}
    for market in MARKETS:
        if market not in factories:
            raise KeyError(f"no factory supplied for market {market!r}")
        target = MARKET_TARGETS[market]
        if target not in combined_frame.columns:
            raise KeyError(
                f"market {market!r} needs column {target!r}, which the frame does not have"
            )

        model = factories[market](train_frame[feature_cols], train_frame[target])
        in_sample = float(
            np.abs(np.asarray(model.predict(train_frame[feature_cols]), dtype=float) - train_frame[target]).mean()
        )
        holdout_pred = np.asarray(model.predict(holdout_frame[feature_cols]), dtype=float)
        holdout_mae = float(np.abs(holdout_pred - holdout_frame[target]).mean())

        # The baseline any model has to beat: the player's own rolling mean (features are shift(1), so causal).
        naive_mae = float(np.abs(holdout_frame[f"{target}_roll"] - holdout_frame[target]).mean())

        # The conformal cold start, measured on these same holdout
        # residuals: the scaled method's sigma curve fitted on the
        # holdout and its normalized 80% quantile, plus the symmetric
        # 80% half-width beside them. Published into the manifest so the
        # served intervals have a measured calibration before the live
        # counted record builds up (tracking/conformal_intervals.py,
        # tier 3) -- the holdout is the only honest source for it.
        conformal = None
        if len(holdout_frame) >= CONFORMAL_MIN_HOLDOUT_ROWS:
            holdout_resid = pd.DataFrame(
                {
                    "projected": holdout_pred,
                    "abs_resid": np.abs(holdout_pred - holdout_frame[target].to_numpy(float)),
                }
            )
            sigma_model = fit_sigma_model(holdout_resid)
            scores = holdout_resid["abs_resid"].to_numpy(float) / sigma_values(
                sigma_model, holdout_resid["projected"]
            )
            conformal = {
                "half_width_80": conformal_half_width(holdout_resid["abs_resid"], 0.8),
                "q80": conformal_half_width(scores, 0.8),
                "sigma": {
                    "edges": sigma_model["edges"],
                    "scales": sigma_model["scales"],
                    "floor": sigma_model["floor"],
                },
                "n_holdout": int(len(holdout_frame)),
            }
        results[market] = {
            "holdout_mae": holdout_mae,
            "naive_mae": naive_mae,
            "in_sample_mae": in_sample,
            # The honest gap. Positive means the in-sample number was flattering,
            # which is the thing this whole task exists to measure.
            "generalisation_gap": holdout_mae - in_sample,
            "train_max_date": train_max,
            "holdout_min_date": holdout_min,
            "n_train": int(len(train_frame)),
            "n_holdout": int(len(holdout_frame)),
            "conformal": conformal,
            "n_players": int(combined_frame["player_id"].nunique()),
            "feature_cols": list(feature_cols),
        }

    return results