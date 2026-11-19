# NBA player props vs Kalshi prop prices: results (2026-11-18)

Preregistered in `2026-11-18-nba-props-vs-kalshi-prereg.md`, commit 72aa301, pushed at
**PREREG_AT = 2026-11-18T20:02:41Z**, before any projection was compared with any settlement. Rows:
`2026-11-18-nba-props-vs-kalshi-rows.csv`. Reproduce with `PYTHONPATH=src python scripts/nba_props_vs_kalshi.py`
(self-check: `--selfcheck`).

## Verdict: no edge shown

Under the frozen rule, the props model loses money against Kalshi prop prices and is clearly less calibrated than
the Kalshi mid. Every cell has its CI below zero. Nothing ships, and the forward shadow below is optional.

## Counts (Kalshi tickers in KXNBAPTS / REB / AST / 3PT, 2026-10-15 to 2026-11-18 snapshots)

| Step | Contracts |
|---|---|
| Tickers in the four series | 44,336 |
| Settled yes/no | 42,092 |
| Matched to one schedule game | 41,167 |
| Matched to exactly one counted projection (player + stat) | 41,147 |
| Projection made before tip (server's `made_before_tip`) | 41,147 |
| Served 80% interval present | 41,147 |
| Two-sided quote before tip | 35,170 |
| At least one side within the 3c slippage cap (eligible) | 28,197 (204 games) |
| Bets (edge >= 4c net of fee) | 16,559 |

Almost every settled contract had a pre-tip model projection: 41,147 of 41,167 game-matched (99.95%). The losses
come from the quote rules, not the model join.

## Results

Net cents per contract over the bets, and log loss on all eligible contracts (LL market minus LL model, so
positive would favour the model). 95% CIs come from a game-clustered bootstrap, 4000 draws, seed 20261017, with
the same resampled games for both metrics.

**Pre-#173 (retrospective, 1c haircut, cannot promote alone):**

| Cell | Contracts | Games | Bets | Net c/contract [95%] | LL market - model [95%] |
|---|---|---|---|---|---|
| **Pooled** | 28,196 | 203 | 16,558 | **-3.43 [-4.44, -2.41]** | **-0.12 [-0.14, -0.10]** |
| Points | 7,447 | 203 | 4,945 | -3.31 [-4.92, -1.63] | -0.14 [-0.17, -0.11] |
| Rebounds | 7,210 | 203 | 4,271 | -4.40 [-6.30, -2.58] | -0.11 [-0.13, -0.08] |
| Assists | 5,705 | 203 | 3,153 | -2.58 [-4.47, -0.72] | -0.13 [-0.16, -0.11] |
| Threes | 7,834 | 203 | 4,189 | -3.22 [-5.00, -1.41] | -0.10 [-0.12, -0.08] |
| YES bets | 6,112 | 201 | 6,112 | -3.53 [-5.05, -2.01] | -0.07 [-0.09, -0.06] |
| NO bets | 10,446 | 203 | 10,446 | -3.37 [-4.73, -1.98] | -0.26 [-0.31, -0.23] |
| Pooled, no haircut | 28,196 | 203 | 16,558 | -2.43 [-3.44, -1.41] | -0.12 [-0.14, -0.10] |

The YES/NO log-loss cells are computed on the bets in that cell only.

**Post-#173: 1 contract, 1 game** (a rebounds NO bet, -16.95c). That is not a sample. As expected, no game that
tipped after the merge (2026-11-18T02:44:51Z) had settled in the data on disk, apart from one late tip.

Mean log loss: model 0.607 vs market mid 0.488 (pooled, both splits).

### Descriptive only (not part of the verdict)

Calibration by model p bucket, all eligible contracts:

| Model p bucket | n | mean model p | mean mid | YES rate |
|---|---|---|---|---|
| < 0.10 | 7,980 | 0.035 | 0.147 | 0.143 |
| 0.10-0.30 | 6,586 | 0.199 | 0.271 | 0.269 |
| 0.30-0.50 | 4,646 | 0.394 | 0.413 | 0.412 |
| 0.50-0.70 | 4,629 | 0.595 | 0.578 | 0.574 |
| 0.70-0.90 | 3,785 | 0.791 | 0.757 | 0.764 |
| >= 0.90 | 571 | 0.925 | 0.864 | 0.872 |

The mid tracks the YES rate within about 1 point in every bucket. The model is far too extreme in the tails: it
puts 3.5% on outcomes that happen 14% of the time. The normal built from the served 80% range has tails that are
too thin for player stats, which have a fat upper tail (blowouts, minutes, hot nights). So the "edge" the rule
finds is mostly the model betting NO on ladder rungs the market correctly prices higher. A fatter-tailed
distribution might narrow the log-loss gap. It would be a new test with its own preregistration, and it could not
be tuned on these rows.

## Deviations and what is unverified

- **Endpoint path.** The prereg names `/api/hub/prop-grading/games/{id}`. The live API serves the same route at
  the root (`/hub/prop-grading/games/{id}`; `/api/...` returns the SPA HTML). Same data, no rule changed.
- **No `created_at` for prop rows.** The public endpoint does not publish it. So I could not check that the
  projection preceded the quote. All 41,147 rows report `made_before_tip: true`, a share I could not cross-check
  against the DB. The pre/post-#173 split is by tip-off, not by recording time.
- **Model rows come from the live site as of 2026-11-18.** They are what the tracking DB holds now. If the counted
  rows were ever rewritten, I could not see it from here. Raw API responses are cached locally (gitignored) and are
  not committed.
- **Depth.** Fills are assumed at top-of-book for any size, with the fee at 100 contracts.
- **The probe C overlap.** This window (204 games) overlaps the probe C sample (180 games) but is not identical,
  and the snapshot files have been extended since.

## Forward-shadow spec for the paper ledger (spec only, no code)

The aim is to log every NBA prop decision forward, with the timestamps the retrospective lacked, so that a
post-#173 sample can be judged on its own.

**Row:** a predictor-hub paper `candidate` row with `kind: "shadow"` and `stake: 0`, reusing the existing fields:
`sport: "nba"`, `market` = stat (`points|rebounds|assists|threes`), `game_id`, `event_ticker`, `ticker`, `player`
(the Kalshi name), `side` (`yes|no`), `line`, `model_p` (P(side wins)), `market_price` (that side's ask, dollars),
`price_observed_at`, `kickoff`, `expected_end`, `fee_cents` / `fee` (`kalshi_fee_cents` at 100 contracts),
`edge`, `yes_bid`, `no_bid`, `spread_points`, `volume`, `open_interest`, `model_fingerprint`, `logged_at`,
`status`, `outcome`, `pnl_per_contract`.

New fields a prop row needs:

- `player_id` (the NBA_Predictor id the projection is keyed on) and `player_match`
  (`exact|normalised`; any other result is not logged).
- `projection` (predicted value), `projection_created_at` (the tracking row's own `created_at`), and
  `projection_counted: true`. It must be the EARLIEST recorded row for (game, player, stat), the counted pick.
- `interval_low`, `interval_high`, `interval_method`, `sigma`, `p_basis: "normal_from_served_80pct_interval"`.
  If the basis ever changes, it gets a new value here and a new prereg, never a silent swap.
- `market_mid` and `p_yes_market`, logged on every eligible contract, bet or not. The log-loss comparison needs the
  non-bets too, so non-bets go in as `record: "score"` rows.
- `model_ordering_proven`: true only when `projection_created_at <= price_observed_at < kickoff`. Rows failing that
  are logged and excluded from the test.
- `nba173: true` (recorded under the 30 h window / 6 h refresh).

**Pricing:** the decision is made once, at the first collector snapshot after `projection_created_at` and before
`kickoff`, at that snapshot's ask. The price is then frozen. Same 4c net-edge threshold, same 3c slippage cap,
taker fee.

**Promotion test (all must hold; evaluated by game-clustered bootstrap, 4000 draws, seed 20261017):**

1. n >= 150 bets with `model_ordering_proven`, all recorded after the shadow went live.
2. low95 of net cents per contract > 0. No haircut, because this is forward data.
3. Paired test against the market-only baseline: low95 of per-contract (LL_market_mid - LL_model) > 0 on the same
   contracts (bets plus score rows).
4. No per-stat cell is reported as an edge unless it passes 1-3 on its own. The pooled result is the gate.

Given the retrospective above (log loss 0.12 nats worse than the mid, CI well clear of zero), the prior on passing
is low. The shadow is worth running only if a better-tailed P(stat > line) is preregistered first.
