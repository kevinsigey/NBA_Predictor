# Tonight's injury report as the availability input: serving replay (negative, not shipped)

Date: 2026-11-17. Verdict: **does not pass the ship rule. Nothing changes in serving.**

## What the input is today

- **Game models** (win / margin / total): the four `*_absent_{min,prod}_share` columns
  (`features/availability.py`). Of a team's top-8 rotation by cumulative minutes, the share held by players who
  **did not play the team's previous game**. No injury feed is read. The models were trained on this definition.
- **Player props**: no availability or teammate feature at all (own rolling means, minutes last/EWMA, matchup,
  home). At serving, a player ESPN's roster lists as `Out` already gets no projection
  (`ingest.to_player_scoring_frame`, `rosters` filter). There is no teammate-minutes redistribution anywhere in the
  repo, so there was nothing on the props side to change.

## Sources that exist (no new source added)

- `data.espn.get_injuries()`: live ESPN league report, already read by the API gates.
- `data/injuries/*.jsonl.gz`: the collector (`.github/workflows/collect-injuries.yml`), four snapshots a day, stored
  **from 2026-10-20 only**. Vocabulary in every stored snapshot is exactly `Out` and `Day-To-Day`. ESPN's feed
  publishes no doubtful / questionable / probable status, so those levels cannot be used.

Training cannot use reports: one month of history is not a training set. The candidate was therefore serving-only.

## Candidate (pre-registered before the replay ran)

For a game with a pre-tip report, a top-8 player counts as missing when the report says `Out`; `Day-To-Day` falls
back to the previous-game rule; a player the report does not list counts as available. Games without a report keep
today's definition. Primary metric: win log-loss; the repo's `ship_decision` (any improves, none worse, AUC must
exclude a decline, calibration gap) applied on top.

## Replay

`tools/replay_injury_report.py` (on branch `claude/nba-injury-avail-impl`). The committed game models
(trained 2026-10-19, before every scored game) score every settled regular-season game from 2026-10-20 to
2026-11-16 twice through the served feature path: baseline vs. features from the latest snapshot seen before tip-off
and no more than 30h before it. Paired bootstrap, 2000 resamples over games, seed 20261117.

n = 204 games with a pre-tip report (of 205 settled); the report changed the features in 201.

| metric | baseline | candidate | diff | 95% CI | |
|---|---|---|---|---|---|
| win log-loss (primary) | 0.6011 | 0.5963 | -0.0048 | [-0.0146, +0.0051] | n.s. |
| win Brier | 0.2078 | 0.2054 | -0.0024 | [-0.0069, +0.0021] | n.s. |
| win AUC | 0.7382 | 0.7516 | +0.0134 | [-0.0041, +0.0315] | n.s. |
| margin MAE | 11.768 | 11.726 | -0.042 | [-0.219, +0.142] | n.s. |
| total MAE | 14.675 | 14.821 | +0.146 | [-0.046, +0.348] | n.s. (leans worse) |

Calibration gap 0.113 -> 0.129 (paired bootstrap: no real widening). No metric improves with an interval excluding
zero, and AUC's interval does not exclude a decline: **does not ship**. Raw per-game baseline and candidate
predictions and features: `docs/nba-injury-report-availability-2026-11.csv`.

## Why, and what would fix it

The report-based shares average 0.15 (home absent-minutes) against 0.33 for the proxy on the same games: the proxy
counts every rest day and DNP as an absence, the report does not. The models' coefficients were fitted on the proxy's
scale, so swapping the input at serving shifts the feature distribution rather than sharpening it. The win-side point
estimates lean the right way; the total leans worse. The honest next step is to retrain on report-based features once
the collector has a season of pre-tip snapshots (2027-28 at the earliest), then re-run this gate as a walk-forward.

The implementation (flag `INJURY_REPORT_AVAILABILITY`, default off), its test and the replay tool are kept on branch
`claude/nba-injury-avail-impl` so this result can be reproduced; per the ship rule this PR is docs-only.
