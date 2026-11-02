# Early-season carry-over evaluation, November 2026

Question (RESEARCH-MEMO Task 2): fitted on the slice the correction exists
for -- the first 15 games each team plays in a season -- does regressing the
stale four-factor rolls toward the pre-season league mean beat leaving them
alone? The October evaluation fitted on early games but SCORED on a
mid-season holdout, which the memo calls uninformative. This one scores on
early-season games only.

## First, a bug: the shipped formula was inverted

`add_rolling_four_factors` applied `target + w * (roll - target)` with `w`
decaying to zero by `window` games. That applies `1 - w` of the intended
shrinkage: the pull toward the mean GREW over the first ten games, and from
game `window` onward the four _roll features were pinned to the pre-season
league mean for the rest of the season. The October 2026 audit flagged this
inversion; the fix never landed. Two tests could not see it (one compared
two weighted arms to each other -- both pinned reads as "no effect"; the
other measured distance from a raw fgm mean, ~43, against a percentage).

Fixed 2026-11-01 (commit c05bc91): corrected = roll + w * (target - roll),
`w = carry_over_weight(k)` -- full pull at game 1, decaying to zero by game
10, nothing after. Both weak tests were rewritten to pin the semantics
(distance from the actual regression target; late games compared to the
uncorrected values). Production was never exposed: no production caller
passes a weight, so served features are unchanged by the fix.

Consequence: every earlier carry-over measurement -- October's included,
and a first run of this evaluation -- compared two broken variants and is
void. This document is the first valid measurement.

## Method

History frame (2018-19 .. 2024-25, full box scores), six season boundaries.
For each boundary S: the weight is fitted ONLY on the prior season's early
games (margin ridge on the carry columns, grid 0 / .25 / .5 / .75 / .9 / 1),
then the candidate factories are trained on all games before S and scored
on S's early games. Arms: fitted weight vs no correction (production
today). Pooled paired bootstrap (seed 20261007) over 1,412 early-season
games; same ship rule as everywhere else (any_improves_none_worse, AUC must
not decline, calibration gap <= 0.0591 and not widened).

## Result

Fitted weights: 2019: 0.0 (inner fit degenerate, fallback), 2020: 1.0,
2021: 0.0, 2022: 0.0, 2023: 0.0, 2024: 0.0. In every normal recent season
the inner MAE increases monotonically with the weight -- the correction
hurts margin prediction on the very games it exists for. The single 1.0 is
the 2020-21 boundary (fitted on the disrupted 2019-20 season's early
games).

Pooled, fitted vs none:
- win log-loss 0.6706 vs 0.6707 (flat, CI crosses zero)
- win Brier 0.2369 vs 0.2370 (flat)
- win AUC 0.6400 vs 0.6401 (flat)
- margin MAE 11.3300 vs 11.3232 (+0.0068, CI -0.0066..+0.0208: not an
  improvement; point estimate worse)
- total MAE 15.6903 vs 15.7125 (-0.0222, CI -0.0436..-0.0004: the one
  significant improvement, driven by the single COVID-era boundary)
- max calibration gap 0.0857 vs 0.0821 (both arms run uncalibrated in this
  harness; the fitted arm's is slightly wider)

## Verdict: measured, not shipped

The total-MAE gain is real but one-boundary-driven, margin does not
improve, and the fitted procedure itself answers "no correction" for every
normal recent season -- wiring it would set weight 0 for 2025-26 anyway.
Production keeps carry_over_weight=None. The corrected formula and its
tests stay: the machinery is now honest, and a future re-fit (e.g. once
2025-26 early games can be scored) starts from valid code. Revisit only if
a fitted weight comes back materially above zero on a normal season.
