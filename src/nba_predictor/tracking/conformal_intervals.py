"""Production conformal intervals for the player-prop projections.

The shipped method is **scaled (normalized) rolling conformal**, selected
by the preregistered trial-2 race
(`docs/conformal-scaled-preregistration-2026-11.md`, results in
`docs/conformal-scaled-results-2026-11.md`): on the frozen holdout it and
the Mondrian construction both repaired the symmetric champion's shape
defect -- low lines over-covered at 92-95%, star lines under-covered at
61-67% at "80%" -- and the frozen selection rule (both fix it -> the
narrower pooled width ships) picked scaled, 6.49 against Mondrian's 6.70.
Trial 1 (PR #170) is why anything ships at all: rolling conformal covers
79.8-80.0% at 80% nominal on all four stats where the MAE bands a reader
could derive cover ~61% (+-1x) or ~88% (+-2x).

**How a shipped interval is built.** For a projection on game day D, per
stat:

* sigma(projection) -- the volatility curve -- is fitted on the counted
  record's residuals from game days **strictly before D** (decile bins of
  |residual| against the projection, isotonic-smoothed, floored: the
  trial-2 estimator in `models/conformal_props.py`, reused verbatim).
  The race froze sigma on a fixed calibration segment; production refits
  it on the expanding prior record, because a live season has no frozen
  segment to freeze on. What production freezes is the CONSTRUCTION.
* The conformal quantile q is taken over **normalized scores**
  |residual| / sigma(projected) from the trailing **14 game days**
  strictly before D, and the half-width is q x sigma(this projection).

**The fallback chain**, in order, with the tier named in every payload
that carries an interval (`basis`):

1. ``rolling_window`` -- the trailing 14 game days hold at least
   `MIN_RESIDUALS` (300) of the stat's counted residuals.
2. ``season_to_date`` -- the window is thinner than that, but the whole
   prior record holds at least 300: q is taken over all prior scores.
3. ``holdout`` -- the counted record before D is thinner still, but the
   props manifest publishes holdout conformal parameters (the retrain
   measures them on its own chronological holdout): those parameters,
   labelled holdout-based wherever they serve. A manifest from before
   the retrain learned to publish them simply has no tier 3.
4. ``season_to_date`` again, on whatever prior record exists (>= 1
   residual): a plain symmetric conformal quantile over a handful of
   real residuals is still a measured number, and its n is printed
   beside it. Only a day with NO prior counted residual and no holdout
   parameters gets no interval at all (None) -- an invented width is
   never served.

**The population is the counted record's, pre-tip only**: the earliest
recorded projection per (game, player, stat) whose own timestamp proves
it was made before tip-off -- the same rows the grading aggregates count
(`tracking/prop_grading.py`), so the calibration population and the
coverage population are one population. A game's intervals use only
residuals from game days strictly before the game's own day; the day
itself can never calibrate its own intervals, and neither can the
future. The chronology tests plant future residuals and assert nothing
moves.

**Calibration population vs displayed pick.** The box-score surface
displays the newest pre-tip pick (`tracking/player_props.py`'s serving
rule); intervals are calibrated on the counted record regardless of
which pick a surface displays, because the interval is a statement
about the model's error distribution, not about one pick's provenance.
"""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd

from nba_predictor.models.conformal_props import (
    conformal_half_width,
    fit_sigma_model,
    sigma_values,
)

#: The shipped interval level. One level, the verdict level both races
#: were judged at; the other levels were evidence, not product.
LEVEL = 0.8

#: The shipped method name, as payloads report it.
METHOD = "scaled"

#: Trailing game days the rolling quantile reads (frozen in both races).
WINDOW_DAYS = 14

#: A window (or a season) thinner than this per stat does not calibrate a
#: scaled tier on itself; the fallback chain takes over.
MIN_RESIDUALS = 300

#: Basis tier names, as they appear in payloads.
BASIS_ROLLING = "rolling_window"
BASIS_SEASON = "season_to_date"
BASIS_HOLDOUT = "holdout"


class PropIntervalCalibrator:
    """Conformal half-widths from one counted record's residuals.

    `residuals` is an iterable of dicts with `game_date`, `stat`,
    `projected`, `abs_resid` -- the counted, pre-tip population only;
    filtering is the caller's one job, and `from_graded_rows` does it for
    the grading population. `cold_start` is the manifest's published
    holdout conformal block per stat (see
    `tracking/prop_grading.py::prop_conformal_cold_start`), or None.

    The day fit -- sigma model, quantile, tier -- is computed once per
    (stat, game day) and memoized: an aggregate pass asks for a
    half-width per counted row, but the expensive part depends only on
    the day, never on the row asking.
    """

    def __init__(self, residuals, cold_start: dict | None = None):
        self._cold_start = cold_start or {}
        self._by_stat: dict[str, tuple[list[str], list[tuple[str, float, float]]]] = {}
        grouped: dict[str, list[tuple[str, float, float]]] = {}
        for row in residuals:
            grouped.setdefault(row["stat"], []).append(
                (row["game_date"], float(row["projected"]), float(row["abs_resid"]))
            )
        for stat, rows in grouped.items():
            rows.sort(key=lambda r: r[0])
            self._by_stat[stat] = ([r[0] for r in rows], rows)
        self._day_fits: dict[tuple[str, str], dict] = {}

    @classmethod
    def from_graded_rows(cls, rows, cold_start: dict | None = None) -> "PropIntervalCalibrator":
        """Build from `prop_grading._graded_rows` output: the pre-tip rows
        only (the aggregates' population), carrying their unrounded miss."""
        return cls(
            (
                {
                    "game_date": r["game_date"],
                    "stat": r["stat"],
                    "projected": r["predicted"],
                    "abs_resid": r["_abs_delta"],
                }
                for r in rows
                if r["made_before_tip"]
            ),
            cold_start,
        )

    def _prior(self, stat: str, game_date: str):
        """(prior rows, trailing-window rows, window day count) for a stat
        strictly before `game_date`. The positional cut on the sorted
        date list is the leakage proof: rows dated ON the day are past
        the cut, never in it."""
        dates, rows = self._by_stat.get(stat, ([], []))
        cut = bisect_left(dates, game_date)
        prior = rows[:cut]
        if not prior:
            return prior, prior, 0
        window_days = sorted({r[0] for r in prior})[-WINDOW_DAYS:]
        window = [r for r in prior if r[0] in set(window_days)]
        return prior, window, len(window_days)

    @staticmethod
    def _scaled_fit(prior_rows, score_rows) -> dict:
        """Sigma fitted on `prior_rows`; q over `score_rows`' normalized
        scores. (When the two are the same set, this is the season tier;
        when score_rows is the trailing window, the rolling tier.)"""
        frame = pd.DataFrame(prior_rows, columns=["game_date", "projected", "abs_resid"])
        sigma_model = fit_sigma_model(frame)
        scores = np.array([r[2] for r in score_rows], dtype=float) / sigma_values(
            sigma_model, [r[1] for r in score_rows]
        )
        return {"sigma_model": sigma_model, "q": conformal_half_width(scores, LEVEL)}

    def _day_fit(self, stat: str, game_date: str) -> dict:
        """The day's calibration for one stat: {basis, n, window_days} and
        the machinery the basis serves -- a sigma model + q for the scaled
        tiers, a constant half-width for the thin symmetric fallback, or
        nothing when no tier can serve."""
        key = (stat, game_date)
        if key in self._day_fits:
            return self._day_fits[key]
        prior, window, window_days = self._prior(stat, game_date)
        if len(window) >= MIN_RESIDUALS:
            fit = {
                "basis": BASIS_ROLLING,
                "n": len(window),
                "window_days": window_days,
                **self._scaled_fit(prior, window),
            }
        elif len(prior) >= MIN_RESIDUALS:
            fit = {
                "basis": BASIS_SEASON,
                "n": len(prior),
                "window_days": len({r[0] for r in prior}),
                **self._scaled_fit(prior, prior),
            }
        elif stat in self._cold_start:
            cold = self._cold_start[stat]
            fit = {
                "basis": BASIS_HOLDOUT,
                "n": int(cold.get("n_holdout", 0)),
                "window_days": 0,
                "sigma_model": cold["sigma"],
                "q": float(cold["q80"]),
            }
        elif prior:
            fit = {
                "basis": BASIS_SEASON,
                "n": len(prior),
                "window_days": len({r[0] for r in prior}),
                "constant_hw": conformal_half_width([r[2] for r in prior], LEVEL),
            }
        else:
            fit = {"basis": None, "n": 0, "window_days": 0}
        self._day_fits[key] = fit
        return fit

    def half_width(self, stat: str, projected: float, game_date: str) -> tuple[float | None, dict]:
        """The 80% half-width for one projection, and its calibration info
        ({"basis", "n", "window_days"}): the tier that served, how many
        counted residuals calibrated it, how many game days they span.
        Half-width None when no tier can serve -- never invented."""
        fit = self._day_fit(stat, game_date)
        info = {"basis": fit["basis"], "n": fit["n"], "window_days": fit["window_days"]}
        if fit["basis"] is None:
            return None, info
        if "constant_hw" in fit:
            return fit["constant_hw"], info
        hw = fit["q"] * float(sigma_values(fit["sigma_model"], [float(projected)])[0])
        return hw, info

    def calibration_info(self, stat: str, game_date: str) -> dict:
        """The day's calibration info for a stat, without naming a line:
        for payload notes that describe how a day was calibrated."""
        fit = self._day_fit(stat, game_date)
        return {"basis": fit["basis"], "n": fit["n"], "window_days": fit["window_days"]}


def interval_for(
    calibrator: PropIntervalCalibrator, stat: str, projected: float, game_date: str
) -> tuple[float | None, float | None, dict]:
    """(low, high, calibration info) for one projection; (None, None, info)
    when no tier can serve. Bounds are NOT rounded here -- the serving
    layer rounds for display the way it rounds the projection itself."""
    hw, info = calibrator.half_width(stat, projected, game_date)
    if hw is None:
        return None, None, info
    return float(projected) - hw, float(projected) + hw, info
