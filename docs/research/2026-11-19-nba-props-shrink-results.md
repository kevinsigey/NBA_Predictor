# NBA props projection shrink (C2): results (2026-11-19)

Preregistered in `2026-11-19-nba-props-shrink-prereg.md`, commit af9fb24.
Rows: `2026-11-19-nba-props-shrink-rows.csv`. Reproduce with
`PYTHONPATH=src:scripts uv run python scripts/nba_props_shrink.py`
(self-check: `--selfcheck`).

## Verdict: fails

Rule 1 fails: pooled delta LL(p_C1) - LL(p_C2) = -0.0119, clustered
low95 = -0.0153 < 0. C2 has higher log loss than the C1 incumbent.

Rule 2 fails on both sub-conditions:
- Per-stat LL deltas are all negative with upper95 < 0 (points -0.0079,
  rebounds -0.0060, assists -0.0067, threes -0.0031).
- MAE(predicted) - MAE(pred') on test-day grading rows has pooled upper95
  -0.0150 < 0 (points -0.0180, rebounds -0.0037, threes -0.0151;
  assists +0.0010 is the only cell not below zero).

Per the prereg, this is docs only. C1 remains the incumbent.

## Counts

Base eligible-contract pipeline (same as `nba_props_vs_kalshi.py`):

| Step | Contracts |
|---|---|
| Tickers in the four series | 48,013 |
| Settled yes/no | 45,171 |
| Matched to one schedule game | 44,246 |
| Matched to exactly one counted projection | 44,226 |
| Projection made before tip | 44,226 |
| Served 80% interval present | 44,226 |
| Two-sided quote before tip | 37,750 |
| Eligible (>= one side within 3c slippage cap) | 30,755 |
| Bets (edge >= 4c net of fee) | 17,897 |

Walk-forward restriction (test days >= 2026-10-27):

| | |
|---|---|
| Test contracts | 24,218 |
| Games | 165 |
| Stat-days skipped for < 500 training rows | 0 |
| Test-day grading rows (MAE) | 15,264 |
| Games (MAE) | 170 |

## Ship-rule cells

Paired per-contract delta = LL(p_C1) - LL(p_C2), game-clustered bootstrap,
4000 draws, seed 20261017. Positive would favour C2.

| Cell | Contracts | Delta LL(p_C1)-LL(p_C2) [95%] |
|---|---|---|
| Pooled | 24,218 | -0.0119 [-0.0153, -0.0081] |
| Points | 6,287 | -0.0130 [-0.0182, -0.0079] |
| Rebounds | 6,318 | -0.0119 [-0.0178, -0.0060] |
| Assists | 4,833 | -0.0125 [-0.0180, -0.0067] |
| Threes | 6,780 | -0.0103 [-0.0169, -0.0031] |

MAE(predicted) - MAE(pred'), test-day grading rows, clustered by game
(positive would mean shrink reduces error):

| Cell | Rows | MAE delta [95%] |
|---|---|---|
| Pooled | 15,264 | -0.0293 [-0.0436, -0.0150] |
| Points | 3,816 | -0.0608 [-0.1025, -0.0180] |
| Rebounds | 3,816 | -0.0222 [-0.0416, -0.0037] |
| Assists | 3,816 | -0.0115 [-0.0232, +0.0010] |
| Threes | 3,816 | -0.0228 [-0.0305, -0.0151] |

## Reported, not gating

Fitted a, b per stat on the last test day (2026-11-18), training rows
n = 5,557 each:

| Stat | a | b |
|---|---|---|
| Points | 1.643819 | 0.824788 |
| Rebounds | 0.952811 | 0.759524 |
| Assists | 0.424832 | 0.819661 |
| Threes | 0.294967 | 0.745091 |

LL market mid - LL(p_C2), clustered (negative = market still better):

| Cell | Contracts | LL mid - LL(p_C2) [95%] |
|---|---|---|
| Pooled | 24,218 | -0.0884 [-0.1022, -0.0755] |
| Points | 6,287 | -0.1073 [-0.1311, -0.0853] |
| Rebounds | 6,318 | -0.0780 [-0.0957, -0.0610] |
| Assists | 4,833 | -0.0919 [-0.1107, -0.0746] |
| Threes | 6,780 | -0.0781 [-0.0927, -0.0645] |

4c-edge betting rerun with p_C2 (1c haircut):

| Cell | Bets | Net c/contract [95%] |
|---|---|---|
| Pooled | 14,922 | -3.5743 [-4.6258, -2.4972] |
| Points | 4,260 | -2.4852 [-4.1803, -0.7800] |
| Rebounds | 4,032 | -3.0070 [-4.7924, -1.1610] |
| Assists | 2,868 | -3.5221 [-5.1928, -1.8312] |
| Threes | 3,762 | -5.4555 [-7.1502, -3.7861] |

Calibration by p_C2 bucket:

| p_C2 bucket | n | mean p_C2 | mean mid | YES rate |
|---|---|---|---|---|
| < .1 | 4,954 | 0.056 | 0.125 | 0.131 |
| .1-.3 | 9,273 | 0.188 | 0.265 | 0.268 |
| .3-.5 | 5,483 | 0.399 | 0.526 | 0.526 |
| .5-.7 | 2,996 | 0.590 | 0.728 | 0.724 |
| .7-.9 | 1,350 | 0.779 | 0.855 | 0.855 |
| >= .9 | 162 | 0.942 | 0.927 | 0.926 |

## Notes

- Training uses grading rows with `game_date` strictly less than the test
  contract's game date. No stat-day was skipped for having fewer than 500
  training rows.
- The served intervals, projections and projection rows are unchanged by
  this script; C2 recomputes P(over) only.
- p_C1 is recomputed exactly as in `scripts/nba_props_tails.py`.
- Raw `p_yes` is retained upstream; this test compares C1 (incumbent) to C2.
