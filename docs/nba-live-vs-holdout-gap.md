# NBA props: why live MAE is worse than holdout

Window 2026-10-03 to 2026-11-02, 173 completed games, 0 skipped after retries. Raw output: `nba-live-vs-holdout-gap.json`. Reproduce: `scripts/nba_live_gap.py`.
Paired comparison on the 2,360 rows scored by every method. Actuals match ESPN 98.3-100%.

| stat | holdout | live served | naive (live) | shipped pickle re-scored | served - naive (95% CI) |
|---|---|---|---|---|---|
| points | 4.63 | 5.79 | 4.99 | 4.67 | +0.80 [0.65, 0.96] |
| rebounds | 1.93 | 2.25 | 1.96 | 1.88 | +0.29 [0.22, 0.35] |
| assists | 1.36 | 1.62 | 1.39 | 1.33 | +0.24 [0.19, 0.28] |
| threes | 0.90 | 1.01 | 0.93 | 0.89 | +0.08 [0.06, 0.11] |

## Finding
- The served projections lose to a causal naive baseline on all four stats (CI excludes zero).
- The same shipped pickle, re-scored today on the same rows, beats naive and matches its holdout MAE. The model is not the problem; what was served differs from what the model gives on settled inputs.
- Likely cause: projections stored early, before lineup and minutes information settles. The API rows carry no projection timestamp, so this is not tested here.
- Gap grows through the window (points 5.53 in week 3 to 5.96 in week 5).

## Next
Log a projection timestamp, and re-score the served projection closer to tip-off. Fixing serving is worth more than any new model work.
