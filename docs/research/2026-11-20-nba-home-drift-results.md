# NBA in-season home-margin drift (R1): results (2026-11-20)

Preregistered in `2026-11-20-nba-home-drift-prereg.md`, commit 5facbc7.
Rows: `2026-11-20-nba-home-drift-rows.csv`. Reproduce with
`PYTHONPATH=src uv run python scripts/nba_home_drift.py`
(self-check: `--selfcheck`).

## Verdict: fails

Rule 1 fails: pooled LL(p) - LL(p') = -0.0004, game bootstrap low95 =
-0.0013 < 0. R1 does not reduce log loss; the in-season bias correction
does not help.

Rule 2 holds: margin MAE(base) - MAE(margin') pooled upper95 = +0.0072
(not < 0), and no single held-out season has mean LL delta with upper95 < 0.
Per the prereg, this is docs only.

## Counts

| | |
|---|---|
| Held-out seasons | 2021, 2022, 2023, 2024 |
| Games (pooled) | 5,061 |
| Per season | 2021: 1,295; 2022: 1,304; 2023: 1,231; 2024: 1,231 |
| Descriptive 2026-27 served games | 288 |

## Ship-rule cells

Primary pooled LL(p) - LL(p'), game bootstrap 4000 draws, seed 20261017,
LL clipped at 0.001. Positive would favour R1.

| Cell | Games | LL delta [95%] |
|---|---|---|
| Pooled | 5,061 | -0.0004 [-0.0013, +0.0005] |
| 2021 | 1,295 | -0.0003 [-0.0017, +0.0011] |
| 2022 | 1,304 | -0.0003 [-0.0025, +0.0020] |
| 2023 | 1,231 | -0.0016 [-0.0035, +0.0003] |
| 2024 | 1,231 | +0.0005 [-0.0012, +0.0022] |

Margin MAE(base) - MAE(margin'), game bootstrap (positive = R1 reduces error):

| Cell | Games | MAE delta [95%] |
|---|---|---|
| Pooled | 5,061 | -0.0090 [-0.0249, +0.0072] |
| 2021 | 1,295 | -0.0026 [-0.0264, +0.0213] |
| 2022 | 1,304 | +0.0051 [-0.0350, +0.0461] |
| 2023 | 1,231 | -0.0332 [-0.0668, +0.0008] |
| 2024 | 1,231 | -0.0066 [-0.0384, +0.0242] |

Delta at season end:

| Season | delta |
|---|---|
| 2021 | +0.2972 |
| 2022 | -0.3980 |
| 2023 | -0.0406 |
| 2024 | +0.7416 |

## Reported, not gating

Descriptive 2026-27 served predictions (R1 applied to served rows):

| | |
|---|---|
| Games | 288 |
| LL delta | +0.0027 [-0.0026, +0.0082] |

## Notes

- Each fold trains the served margin model (candidate race winner) on all
  earlier seasons only; sigma is fitted on training residuals via
  walk-forward regression, as in `pipeline/retrain.py`. Max training date <
  min test date asserted per fold.
- Delta for a game uses only games dated strictly before that game's date;
  fewer than 100 prior games gives delta 0.
- The 2026-27 cell is descriptive only (served predictions, not walk-forward).
- The served intervals, totals and other outputs are untouched.
