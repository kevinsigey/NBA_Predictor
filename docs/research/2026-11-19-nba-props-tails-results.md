# NBA props fatter tails (C1 empirical residuals): results (2026-11-19)

Preregistered in `2026-11-19-nba-props-tails-prereg.md`, commit 18a6b63,
PREREG_AT = 2026-11-19T14:42:20Z. Rows: `2026-11-19-nba-props-tails-rows.csv`.
Reproduce with `PYTHONPATH=src uv run python scripts/nba_props_tails.py`
(self-check: `--selfcheck`).

## Verdict: PASS

All three ship-rule conditions hold on the walk-forward sample:

1. Primary pooled delta LL(raw p_yes) - LL(p_C1) = +0.0315, clustered
   low95 = +0.0245 > 0.
2. No stat has mean delta with upper95 < 0 (points +0.0395, rebounds +0.0333,
   assists +0.0488, threes +0.0471). Both arms keep the 0.001 log-loss clip.
3. Served intervals are not touched by the script.

Per the prereg, NBA_Predictor may compute P(over) with C1 behind a flag
(default off), storing raw `p_yes` beside `p_C1`. Promotion to default and
to the paper ledger shadow is a separate step needing Kevin's go. C1 does
not close the gap to the market (LL mid - LL(C1) stays negative), so this
is an improvement over the raw model's calibration, not an edge over Kalshi.

## Counts

Base eligible-contract pipeline (same as `nba_props_vs_kalshi.py`):

| Step | Contracts |
|---|---|
| Tickers in the four series | 47,221 |
| Settled yes/no | 44,222 |
| Matched to one schedule game | 43,297 |
| Matched to exactly one counted projection | 43,277 |
| Projection made before tip | 43,277 |
| Served 80% interval present | 43,277 |
| Two-sided quote before tip | 36,971 |
| Eligible (>= one side within 3c slippage cap) | 29,981 |
| Bets (edge >= 4c net of fee) | 17,501 |

Walk-forward restriction (test days >= 2026-10-27):

| | |
|---|---|
| Test contracts | 23,444 |
| Games | 164 |
| Stat-days skipped for < 500 training rows | 0 |

## Ship-rule cells

Paired per-contract delta = LL(raw p_yes) - LL(p_C1), game-clustered
bootstrap, 4000 draws, seed 20261017. Positive favours C1.

| Cell | Contracts | Delta LL(raw)-LL(C1) [95%] |
|---|---|---|
| Pooled | 23,444 | +0.0315 [+0.0245, +0.0392] |
| Points | 6,150 | +0.0300 [+0.0211, +0.0395] |
| Rebounds | 6,085 | +0.0244 [+0.0166, +0.0333] |
| Assists | 4,731 | +0.0372 [+0.0258, +0.0488] |
| Threes | 6,478 | +0.0356 [+0.0246, +0.0471] |

## Reported, not gating

LL market mid - LL(p_C1), clustered (negative = market still better):

| Cell | Contracts | LL mid - LL(C1) [95%] |
|---|---|---|
| Pooled | 23,444 | -0.0765 [-0.0900, -0.0639] |
| Points | 6,150 | -0.0954 [-0.1187, -0.0746] |
| Rebounds | 6,085 | -0.0660 [-0.0832, -0.0498] |
| Assists | 4,731 | -0.0803 [-0.0982, -0.0623] |
| Threes | 6,478 | -0.0656 [-0.0806, -0.0516] |

4c-edge betting rerun with p_C1 (1c haircut):

| Cell | Bets | Net c/contract [95%] |
|---|---|---|
| Pooled | 13,164 | -3.8006 [-4.9079, -2.7165] |
| Points | 3,892 | -2.8609 [-4.7041, -1.0889] |
| Rebounds | 3,500 | -3.7691 [-5.8070, -1.7279] |
| Assists | 2,581 | -3.7736 [-5.6638, -1.9409] |
| Threes | 3,191 | -5.0033 [-6.8245, -3.1784] |

Calibration by p_C1 bucket:

| p_C1 bucket | n | mean p_C1 | mean mid | YES rate |
|---|---|---|---|---|
| < .1 | 4,310 | 0.056 | 0.125 | 0.132 |
| .1-.3 | 8,324 | 0.194 | 0.246 | 0.250 |
| .3-.5 | 4,832 | 0.396 | 0.459 | 0.458 |
| .5-.7 | 2,960 | 0.599 | 0.637 | 0.628 |
| .7-.9 | 2,215 | 0.789 | 0.790 | 0.797 |
| >= .9 | 803 | 0.941 | 0.876 | 0.898 |

## Notes

- Training uses grading rows with `game_date` strictly less than the test
  contract's game date (leakage guard). No stat-day was skipped for having
  fewer than 500 training rows.
- The served intervals, projections and projection rows are unchanged; only
  P(over) changes. The script never writes intervals.
- Raw `p_yes` is retained (as `p_raw` in the rows CSV) beside `p_C1`.
