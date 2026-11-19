# NBA props fatter tails (C1 empirical residuals): results (2026-11-19)

Preregistered in `2026-11-19-nba-props-tails-prereg.md`, commit 18a6b63,
PREREG_AT = 2026-11-19T14:42:20Z. Rows: `2026-11-19-nba-props-tails-rows.csv`.
Reproduce with `PYTHONPATH=src uv run python scripts/nba_props_tails.py`
(self-check: `--selfcheck`).

## Verdict: PASS

All three ship-rule conditions hold on the walk-forward sample:

1. Primary pooled delta LL(raw p_yes) - LL(p_C1) = +0.0327, clustered
   low95 = +0.0255 > 0.
2. No stat has mean delta with upper95 < 0 (points +0.0419, rebounds +0.0335,
   assists +0.0498, threes +0.0488). Both arms keep the 0.001 log-loss clip.
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
| Pooled | 23,444 | +0.0327 [+0.0255, +0.0406] |
| Points | 6,150 | +0.0320 [+0.0228, +0.0419] |
| Rebounds | 6,085 | +0.0246 [+0.0168, +0.0335] |
| Assists | 4,731 | +0.0381 [+0.0267, +0.0498] |
| Threes | 6,478 | +0.0370 [+0.0258, +0.0488] |

## Reported, not gating

LL market mid - LL(p_C1), clustered (negative = market still better):

| Cell | Contracts | LL mid - LL(C1) [95%] |
|---|---|---|
| Pooled | 23,444 | -0.0753 [-0.0887, -0.0629] |
| Points | 6,150 | -0.0934 [-0.1162, -0.0727] |
| Rebounds | 6,085 | -0.0657 [-0.0830, -0.0495] |
| Assists | 4,731 | -0.0794 [-0.0971, -0.0615] |
| Threes | 6,478 | -0.0642 [-0.0790, -0.0505] |

4c-edge betting rerun with p_C1 (1c haircut):

| Cell | Bets | Net c/contract [95%] |
|---|---|---|
| Pooled | 13,162 | -3.8430 [-4.9413, -2.7551] |
| Points | 3,890 | -2.9680 [-4.7680, -1.2010] |
| Rebounds | 3,493 | -3.9351 [-5.9759, -1.8967] |
| Assists | 2,585 | -3.7075 [-5.5552, -1.8936] |
| Threes | 3,194 | -4.9178 [-6.7192, -3.1162] |

Calibration by p_C1 bucket:

| p_C1 bucket | n | mean p_C1 | mean mid | YES rate |
|---|---|---|---|---|
| < .1 | 4,213 | 0.057 | 0.124 | 0.132 |
| .1-.3 | 8,300 | 0.195 | 0.243 | 0.245 |
| .3-.5 | 4,777 | 0.396 | 0.453 | 0.454 |
| .5-.7 | 3,115 | 0.598 | 0.633 | 0.625 |
| .7-.9 | 2,218 | 0.791 | 0.789 | 0.795 |
| >= .9 | 821 | 0.942 | 0.876 | 0.898 |

## Notes

- Training uses grading rows with `game_date` strictly less than the test
  contract's game date (leakage guard). No stat-day was skipped for having
  fewer than 500 training rows.
- The served intervals, projections and projection rows are unchanged; only
  P(over) changes. The script never writes intervals.
- Raw `p_yes` is retained (as `p_raw` in the rows CSV) beside `p_C1`.
