# NBA props: shrink projections toward the mean, preregistration (2026-11-19)

Follows `2026-11-19-nba-props-tails-results.md` (C1 passed). Its calibration table still showed P(over) about 7
points below the YES rate in the 0.3-0.5 bucket.

**What the grading cache shows (descriptive, before this rule; 5,836 pre-tip rows per stat):** the projections are
not low on average. Mean residuals (actual - predicted) are -0.08 for points, +0.04 rebounds, +0.03 assists, +0.01
threes. They are too spread out. Projections of 25+ points finish 3.5 below on average, projections under 5 finish
1.4 above, and the same slope shows up in every stat. That is regression to the mean: a calibration slope below 1.
No number for the candidate below has been computed.

## Candidate (one, fixed)

**C2, linear recalibration plus C1 tails.** For test day D and stat s, fit ordinary least squares
actual = a + b * predicted on the training rows. Then pred' = a + b * pred. Residuals are z' = (actual - pred') / half,
with half from the served interval exactly as in C1. Then p_C2 = (k + 0.5) / (n + 1), where
k = #{training z' : pred' + z' * h > L}. Both the fit and the residuals use only training rows.

## Walk-forward, rows and data

Identical to `2026-11-19-nba-props-tails-prereg.md`:
- test days from 2026-10-27;
- training = grading rows with game_date < D, all players, with the same filters;
- skip a stat-day if it has under 500 training rows;
- test contracts = the eligible set of `scripts/nba_props_vs_kalshi.py`.

The baseline is p_C1 from `scripts/nba_props_tails.py`. That is the passed incumbent, not the raw normal.

## Ship rule (all must hold)

1. **Primary:** paired per-contract delta = LL(p_C1) - LL(p_C2), game-clustered bootstrap, 4000 draws, seed
   20261017. low95 > 0.
2. **Nothing else worse:**
   - no stat has mean delta with upper95 < 0;
   - point accuracy on every test-day grading row (all players, not only Kalshi-matched):
     MAE(predicted) - MAE(pred') must not have upper95 < 0, clustered by game.

**If it passes:** docs, plus a follow-up PR that applies the walk-forward shrink to served projections behind a
flag, default off, storing the raw projection beside it. That PR needs Kevin's go to switch on.
**If it fails:** docs only.

## Reported, not gating

- MAE delta per stat.
- The fitted a and b per stat on the last test day.
- LL mid - LL(p_C2).
- The 4c betting rerun with p_C2 (1c haircut).
- The calibration table by p_C2 bucket.
- Counts.
