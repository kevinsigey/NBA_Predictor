# NBA player props vs Kalshi prop prices: preregistration (2026-11-18)

Approved by Kevin on Nov 18. This is predictor-hub `docs/research/2026-11-18-data-coverage-and-angles.md`, angle 2.
Probe C there found Kalshi NBA prop mids calibrated (outcome minus mid +0.49 pts [-0.42, +1.48], 38,998 contracts,
180 games), so any edge has to come from the model. This file is committed before any model projection is compared
with any Kalshi settlement. Its commit time is `PREREG_AT`. Nothing below changes after that commit. If a rule turns
out to be unworkable, the results doc says so and reports the rule as frozen anyway.

## Data

- **Kalshi.** Every `data/lines/kalshi_nba_lines_*.jsonl.gz` on NBA_Predictor origin/main at run time (the
  "Collect Kalshi NBA lines" workflow, 15:30 and 23:15 UTC plus manual runs). Series used: `KXNBAPTS` (points),
  `KXNBAREB` (rebounds), `KXNBAAST` (assists), `KXNBA3PT` (threes). Those are the four stats the props model serves.
  PRA, steals and blocks have no served model and are out of scope. They are counted as unmatched, not hidden.
- **Model.** The counted pick: the EARLIEST recorded projection per (game, player, stat), as the live site serves it
  at `GET /api/hub/prop-grading/games/{game_id}` (`predicted`, `made_before_tip`, `interval_low`, `interval_high`).
  That is the only public read of the tracking DB. There is no SSH and no DB copy. The endpoint does not publish a
  prop row's `created_at`, only `made_before_tip`, which the server derives from `created_at` against tip-off and
  which fails closed to False.
- **Schedule / tip-off.** `data/public_snapshot.json` `schedule[].tip_off` (UTC) at run time.

## Join (Kalshi ticker to game, player, stat, line)

1. **Game.** Take the event code from `event_ticker`, the part after the series (e.g. `KXNBAPTS-26OCT20OKCSAS` gives
   `26OCT20OKCSAS`). Drop the 7-character date prefix to get the team string. It matches a schedule game when it
   equals `away_team + home_team` or `home_team + away_team`, and that game's `tip_off` is within 30 h of the
   contract's `occurrence` (the same `MATCH_HOURS` rule as `pipeline/refresh_kalshi.py`). No match, or more than one
   match, means unmatched.
2. **Stat.** Taken from the row's `stat` field (`points`, `rebounds`, `assists`, `threes`).
3. **Line.** Taken from the row's `line` field (e.g. 34.5 for "35+"). YES pays when stat > line, which is the same as
   stat >= line + 0.5.
4. **Player.** Kalshi `player` is matched to the grading row's `player_name` within the same game, after
   normalising both names: NFKD, ASCII, lowercase, punctuation dropped, and the suffixes jr/sr/ii/iii/iv dropped.
   A match needs exactly one grading row with that name and stat. Zero or more than one means unmatched.
5. **Outcome.** The latest snapshot row of the ticker with `status` finalized and `result` yes or no. `scalar`,
   void and unresolved tickers are dropped and counted.

## Model probability

The model publishes no probability for props. P(YES) = P(stat > line) is built from the projection and the
model's own served 80% range: the scaled rolling conformal interval in the grading row, calibrated on counted game
days strictly before the game's day. So it is as-of, not a refit.

- `half = (interval_high - interval_low) / 2`. If `interval_low` is 0 and `predicted - 0 < half`, the range was
  clipped at zero, and `half = interval_high - predicted` is used instead.
- `sigma = half / 1.2816`. That is a normal whose central 80% equals the served range.
- `p_yes = 1 - Phi((line - predicted) / sigma)`. The line is always x.5, so no continuity correction is needed.
- No interval (`interval_low` is None) means the row is excluded and counted.
- p is clipped to [0.001, 0.999] for log loss only.

This is a normal approximation to the served interval, not a distribution the model fitted. Threes and low-count
assists are discrete and skewed, so the per-stat cells are where that shows up.

## Timing (only quotes and projections recorded before tip)

- **Projection.** `made_before_tip` must be True. Rows that cannot prove it are excluded and counted.
- **Quote, and the frozen-price rule.** For each ticker, use the LAST snapshot with `snapshot_ts` before the game's
  `tip_off` whose quote is two-sided (0 < yes_bid < yes_ask < 1). That snapshot's `yes_ask` / `no_ask` /
  mid = (yes_bid + yes_ask) / 2 are frozen. A later price is never substituted, a better earlier price is never
  picked, and the decision is never re-made.
- The public endpoint gives no `created_at`, so the projection cannot be proven to precede the quote. Using the
  last pre-tip quote is the conservative choice: the market has the most information it will have before tip. A
  projection recorded after the last pre-tip snapshot (possible under the 6 h refresh) would give the model
  information the quote lacks. That direction favours the model, and the results doc names it as unverified.

## Decision rule (fixed now)

- Fee: predictor-hub `services/paper/paper/fees.py` `kalshi_fee_cents` (taker, `ceil(7 * C * P * (100 - P) / 10000)`
  cents, whole-cent price), at C = 100 contracts, divided by 100. That is the per-contract basis probe C used.
- `edge_yes = 100 * p_yes - yes_ask_c - fee(yes_ask_c)` and `edge_no = 100 * (1 - p_yes) - no_ask_c - fee(no_ask_c)`,
  all in cents.
- **Threshold: bet the side whose edge is >= 4.0 cents** (the hub paper ledger's `DEFAULT_EDGE_POINTS`). If both
  sides clear it, take the larger edge. One contract per ticker.
- **3-cent slippage cap.** A side is eligible only if its ask is at most 3 cents above the mid (`ask - mid <= 0.03`).
  A wider quote is not a price we could expect to fill near.
- **Depth caveat.** Snapshots carry no order-book depth, only top-of-book, volume and open interest. Every fill is
  assumed at the top-of-book ask for any size. That is optimistic for size, and the per-contract figure is not a
  capacity estimate.
- `pnl_c = 100 * won - ask_c - fee(ask_c)`, the same arithmetic as the hub's `settle.pnl_per_contract`.

## Metrics

- **Primary: net cents per contract** over the bets, with a 95% CI from a game-clustered bootstrap: resample whole
  games, take the per-contract mean of the resampled bets, 4000 draws, seed 20261017 (the hub's constants), and
  read the percentile interval.
- **Secondary: log loss of model p_yes vs market mid** on ALL eligible matched contracts, bet or not. The paired
  per-contract difference `LL_market - LL_model` (positive means the model is better) uses the same clustered
  bootstrap, and the same resampled games feed both metrics in each draw.
- Cells: pooled, per stat (points, rebounds, assists, threes), and YES vs NO bets.
- Counts reported honestly: Kalshi prop tickers in the four series, settled, game-matched, player-matched,
  pre-tip projection, interval present, pre-tip two-sided quote, eligible, and bets.

## Split by NBA#173

NBA#173 (merge commit f9b7226, 2026-11-18T02:44:51Z) limited recording to a 30 h lead window with a 6 h refresh.
That changed when the counted (earliest) pick is recorded.

- **Pre-#173: games with tip_off before 2026-11-18T02:44:51Z.** These are a retrospective. Each bet's pnl takes a
  **1-cent haircut**, and the haircut figure is the one the verdict reads. This sample **cannot promote alone**.
- **Post-#173: games with tip_off at or after that instant.** Reported separately. Without `created_at` a post-#173
  game's earliest row can still predate the merge. The cell is labelled with that, and it is expected to be close
  to empty on the data available on 11-18.

## Verdict rule

- Pre-#173 haircut net cents low95 > 0 AND paired log-loss low95 > 0: **"retrospective positive"**. The forward
  shadow runs. Nothing is promoted on this.
- Pre-#173 haircut net cents low95 > 0 with log loss not better: **"net positive, not better calibrated"**. The
  forward shadow runs, flagged as likely selection noise.
- Otherwise: **"no edge shown"**. A forward shadow is still allowed, since it is cheap, but nothing ships.
- Per-stat and YES/NO cells are descriptive. They do not change the verdict, and no cell is picked after the fact.

## Outputs

- `docs/research/2026-11-18-nba-props-vs-kalshi-rows.csv`: one row per eligible matched contract (ticker, game,
  player, stat, line, predicted, sigma, p_yes, mid, yes_ask, no_ask, side, edge, outcome, pnl, split).
- `docs/research/2026-11-18-nba-props-vs-kalshi-results.md`: counts, cells, CIs, verdict, unverified items, and
  the forward-shadow spec.
- Script: `scripts/nba_props_vs_kalshi.py`.
