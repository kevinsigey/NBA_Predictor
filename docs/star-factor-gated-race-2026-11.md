# Star factor trial 2 (star_out-gated): race results

Preregistered in `docs/star-factor-gated-preregistration-2026-11.md`. Raw numbers: `docs/star-factor-gated-race-2026-11.json`.

Holdout: 36551 rows (train through 2024-02-04, holdout from 2024-02-05); star_out == 1 on 10607 rows, star_out == 0 on 25944. Paired bootstrap 2000 resamples, seed 20261102.

Replication anchor (trial 1's base/naive MAEs reproduced): **True**. Per-row path vs harness max MAE difference: 0.00e+00.

| stat | naive MAE | base MAE | candidate MAE | delta (cand-base) | paired 95% CI |
|---|---|---|---|---|---|
| points | 4.696 | 4.606 | 4.607 | +0.0010 | [-0.0011, +0.0031] |
| rebounds | 1.956 | 1.926 | 1.927 | +0.0004 | [-0.0003, +0.0010] |
| assists | 1.374 | 1.355 | 1.356 | +0.0005 | [-0.0003, +0.0012] |
| threes | 0.896 | 0.893 | 0.894 | +0.0010 | [+0.0006, +0.0013] |

## Slices (transparency, not the gate)

| stat | slice | n | MAE base -> candidate | bias base -> candidate |
|---|---|---|---|---|
| points | star_out == 1 | 10607 | 4.6917 -> 4.7078 | -0.2646 -> -0.0952 |
| points | star_out == 0 | 25944 | 4.5715 -> 4.5663 | -0.0373 -> -0.0770 |
| rebounds | star_out == 1 | 10607 | 1.9651 -> 1.9704 | -0.1038 -> -0.0648 |
| rebounds | star_out == 0 | 25944 | 1.9103 -> 1.9087 | -0.0192 -> -0.0289 |
| assists | star_out == 1 | 10607 | 1.3674 -> 1.3750 | -0.1105 -> -0.0441 |
| assists | star_out == 0 | 25944 | 1.3505 -> 1.3480 | -0.0118 -> -0.0299 |
| threes | star_out == 1 | 10607 | 0.8833 -> 0.8878 | -0.0414 -> -0.0255 |
| threes | star_out == 0 | 25944 | 0.8968 -> 0.8963 | -0.0404 -> -0.0437 |

## Gate (frozen in the preregistration)

- (a) >=1 stat improves and none worsens beyond noise: **False** (improved: none; worsened beyond noise: ['threes'])
- (b) the improving stat's CI excludes zero: **False** (confirmed: none)
- (c) branch CI fully green: decided on the PR.

**Gate before clause (c): FAIL**

