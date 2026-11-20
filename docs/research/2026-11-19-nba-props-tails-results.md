# NBA props fatter tails (C1 empirical residuals): results (2026-11-19)

Preregistered in `2026-11-19-nba-props-tails-prereg.md`, commit 18a6b63,
PREREG_AT = 2026-11-19T14:42:20Z. Rows: `2026-11-19-nba-props-tails-rows.csv`.
Reproduce with `PYTHONPATH=src uv run python scripts/nba_props_tails.py`
(self-check: `--selfcheck`).

## Verdict: PASS

All three ship-rule conditions hold on the walk-forward sample:

1. Primary pooled delta LL(raw p_yes) - LL(p_C1) = +0.0305, clustered
   low95 = +0.0235 > 0.
2. No stat has mean delta with upper95 < 0 (points +0.0388, rebounds +0.0315,
   assists +0.0492, threes +0.0455). Both arms keep the 0.001 log-loss clip.
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
| Pooled | 23,444 | +0.0305 [+0.0235, +0.0380] |
| Points | 6,150 | +0.0290 [+0.0201, +0.0388] |
| Rebounds | 6,085 | +0.0229 [+0.0155, +0.0315] |
| Assists | 4,731 | +0.0373 [+0.0257, +0.0492] |
| Threes | 6,478 | +0.0340 [+0.0228, +0.0455] |

## Reported, not gating

LL market mid - LL(p_C1), clustered (negative = market still better):

| Cell | Contracts | LL mid - LL(C1) [95%] |
|---|---|---|
| Pooled | 23,444 | -0.0776 [-0.0912, -0.0648] |
| Points | 6,150 | -0.0964 [-0.1198, -0.0753] |
| Rebounds | 6,085 | -0.0675 [-0.0852, -0.0510] |
| Assists | 4,731 | -0.0802 [-0.0980, -0.0624] |
| Threes | 6,478 | -0.0673 [-0.0822, -0.0531] |

4c-edge betting rerun with p_C1 (1c haircut):

| Cell | Bets | Net c/contract [95%] |
|---|---|---|
| Pooled | 13,215 | -3.6827 [-4.7702, -2.5828] |
| Points | 3,910 | -2.7456 [-4.5406, -0.9644] |
| Rebounds | 3,518 | -3.5522 [-5.5755, -1.5787] |
| Assists | 2,579 | -3.7083 [-5.5703, -1.8597] |
| Threes | 3,208 | -4.9476 [-6.6767, -3.2021] |

Calibration by p_C1 bucket:

| p_C1 bucket | n | mean p_C1 | mean mid | YES rate |
|---|---|---|---|---|
| < .1 | 4,447 | 0.057 | 0.126 | 0.135 |
| .1-.3 | 8,339 | 0.194 | 0.250 | 0.253 |
| .3-.5 | 4,788 | 0.394 | 0.465 | 0.461 |
| .5-.7 | 3,029 | 0.596 | 0.643 | 0.637 |
| .7-.9 | 2,133 | 0.791 | 0.799 | 0.809 |
| >= .9 | 708 | 0.943 | 0.880 | 0.898 |

## Notes

- Training uses grading rows with `game_date` strictly less than the test
  contract's game date (leakage guard). No stat-day was skipped for having
  fewer than 500 training rows.
- The served intervals, projections and projection rows are unchanged; only
  P(over) changes. The script never writes intervals.
- Raw `p_yes` is retained (as `p_raw` in the rows CSV) beside `p_C1`.
