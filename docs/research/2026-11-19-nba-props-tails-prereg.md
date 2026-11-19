# NBA props: fatter tails for P(over line), preregistration (2026-11-19)

Follows `2026-11-18-nba-props-vs-kalshi-results.md`. That test found the props model 0.12 nats worse than the
Kalshi mid, because it puts 3.5% on outcomes that happen 14% of the time.

**Why not just widen the ranges.** The served 80% intervals already cover 80% of results for every stat. Over
4,612 pre-tip rows per stat in the local grading cache, coverage is points 0.802, rebounds 0.803, assists 0.803
and threes 0.792. Widening them would break coverage. The fault is the shape used to turn the range into
P(stat > line): `p_yes` assumes a normal with sigma = half-width / 1.2816, which is too thin past the 80% band.

Disclosure: this rule was written after seeing the calibration table in the 2026-11-18 results. No number for the
candidate below has been computed. The committed time of this file is PREREG_AT.

## Candidate (one, fixed)

**C1, empirical residual tails.** For stat s, take every training grading row that has `made_before_tip`, a
served interval and an `actual`. Compute its scaled residual z = (actual - predicted) / half, where half is the
same as in `p_yes` (half = (hi - lo)/2; if lo == 0 and predicted < half, half = hi - predicted). For a test
contract with projection pred, half h and line L:

    k = #{training z of stat s : pred + z * h > L}
    p_C1 = (k + 0.5) / (n + 1)

The served interval, the projection and the projection's rows are unchanged. Only P(over) changes.

## Walk-forward

- **Test days:** every game date from 2026-10-27 to the last date in the data, in order. 2026-10-20 to 2026-10-26
  are burn-in only.
- **Training for test day D:** grading rows from games with `game_date` < D, all players (not only those with a
  Kalshi market). Training needs at least 500 rows for stat s, or that stat-day is skipped and counted.
- **Test contracts:** exactly the eligible set of `scripts/nba_props_vs_kalshi.py` (same matching, last two-sided
  pre-tip quote, 3c slippage cap), restricted to test days. The comparison is the raw `p_yes` against p_C1 on the
  same contracts.

## Ship rule (all must hold)

1. **Primary:** paired per-contract delta = LL(raw p_yes) - LL(p_C1), game-clustered bootstrap, 4000 draws, seed
   20261017. low95 > 0.
2. **Nothing else worse:** no stat has mean delta with an upper95 < 0. Both arms keep the 0.001 log-loss clip.
3. The served intervals are not touched, so coverage cannot move. The test checks that the script never writes
   them.

**If it passes:** NBA_Predictor computes P(over) with C1 behind a flag, default off. It stores raw `p_yes` beside
`p_C1`. Promotion to default and to the paper ledger shadow is a separate step that needs Kevin's go.
**If it fails:** docs only.

## Reported, not gating

- LL market mid - LL(p_C1), clustered CI. Says whether C1 closes the gap to the market.
- The 4c-edge betting rule rerun with p_C1: bets and net c/contract with a 1c haircut. This is retrospective and
  cannot promote anything.
- Calibration table by p_C1 bucket, with the same buckets as the 2026-11-18 results.
- Counts: test contracts, games, and stat-days skipped for having under 500 training rows.
