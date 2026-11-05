# Conformal intervals for player props — preregistration

Written and committed BEFORE the race runs. The spec below is frozen: no edits
after the first race run starts, and no edit may rescue a marginal result.
This is a measurement, not a ship: the PR stays open and unmerged whatever the
outcome, and Kevin decides afterwards whether any of this reaches a reader.

## The question

The props models publish point projections. The only uncertainty a reader can
currently derive from them is the grade bands (`tracking/prop_grading.py`):
multiples of the stat's holdout MAE — within 0.5× is "On the number", 1× is
"Close", 2× is "Off". Those bands are descriptive: nobody ever promised that
±1× MAE contains any particular share of outcomes, and the live counted record
says what it actually contains (from the committed public snapshot's
aggregates, 13,004 pre-tip lines, 2026-10-03 → 2026-11-02): ±1× MAE covers
52.3% of points lines, 55.2% of rebounds, 55.5% of assists, 57.1% of threes.

Conformal prediction is the standard way to turn a point forecaster's past
residuals into intervals with a coverage target: take the (1−α) quantile of
past |residuals| and publish projection ± that. The question this race
answers:

**Do conformal intervals deliver empirical coverage materially closer to
their nominal level than the MAE-multiple bands, at comparable or narrower
width, under strictly chronological walk-forward evaluation — and does a
trailing-window (rolling) calibration track early-season drift better than a
frozen (split) calibration?**

## Population and harness (frozen)

Identical track to the star-factor races (PRs #164/#165), because the
residuals must come from the production model, not a stand-in:

- Data: the frozen history `data/history/nba_history.json.gz` via
  `pipeline/history.py::load_history`, player rows with minutes > 0.
- Frame: `features/prop_model_frame.py::build_prop_model_frame` — the 12
  production features, built once over train+holdout so holdout rolling
  features inherit training history (the harness's standing discipline).
- Split: `models/prop_holdout.py::_date_aligned_split`, chronological
  last-20%, date-aligned. Train ≤ 2024-02-04 (147,618 rows); holdout
  2024-02-05 → 2025-04-13, **36,551 rows over 226 game days**, every row
  carrying all four targets. One split for all four stats.
- Models: the production `train_player_stat_model` per stat, fitted once on
  the train segment. The model is never refitted inside the holdout and never
  sees a holdout row in training. Residual = actual − predicted on holdout
  rows.
- Replication anchor: the fitted models' full-holdout MAEs must reproduce the
  star-factor races' base arm to rounding — points 4.606, rebounds 1.926,
  assists 1.355, threes 0.893 (naive rolling-mean: 4.696 / 1.956 / 1.374 /
  0.896). If they do not, the race is stopped and reported as a setup
  failure, not raced on a different track.

The holdout spans the end of the 2023-24 season and the whole 2024-25 season
to 2025-04-13, so it contains a season boundary and an early-season stretch
(Oct–Nov 2024) — the drift regime the rolling method claims to handle.

## The within-holdout split (frozen)

- **Calibration segment:** the first 56 of the holdout's 226 game days
  (~9,000 rows per stat).
- **Evaluation segment:** the remaining 170 game days (~27,500 rows per
  stat). Every reported coverage/width number is computed on evaluation rows
  only, in per-game-day blocks, aggregated.
- Nothing is shuffled and nothing is randomly split, anywhere. A calibration
  row dated on or after an evaluated row is a defect the tests must make
  impossible, not a caveat.

This mirrors production: a model and its error figures are fitted before a
season and then lived with. The frozen calibration faces the same regime as
the published MAE bands.

## Methods (frozen)

All intervals are symmetric around the projection, per stat.

1. **Status quo — MAE bands.** Projection ± k × (manifest holdout MAE),
   k ∈ {1, 2}, with the MAEs exactly as published in
   `models/player_props_manifest.json` at preregistration time: points
   4.6286, rebounds 1.9315, assists 1.3548, threes 0.8986. These arms have no
   nominal level; their empirical coverage IS the finding. (The 0.5× band is
   recorded in the raw JSON for completeness; the verdict comparison uses
   {1×, 2×} only.)
2. **Split conformal.** Projection ± q, where q is the conformal quantile of
   |residuals| over the calibration segment, computed once and frozen for the
   whole evaluation segment.
3. **Rolling conformal.** Same construction, but for each evaluation game
   day the calibration set is the trailing **14 game days** strictly before
   that day (the window reaches back into the calibration segment and into
   earlier evaluation days — always strictly before the day being scored).
   14 days at the holdout's density (median 160 rows/day) is ~2,270
   residuals per stat: enough that the 90% order statistic is stable, short
   enough to follow form on the ~2-week timescale the live record's daily
   MAEs move on. A 5,000-row window (~31 days) would adapt at half that
   speed; a 7-day window halves the quantile's support. 14 days is the
   frozen compromise. (Implementation guard: a day is scored for rolling only
   if its window holds ≥ 500 residuals per stat; in this population the
   window is always full, so the guard is a tripwire, not a filter.)

**Nominal levels:** 50%, 80%, 90% for both conformal methods.

**Quantile convention (finite-sample correction):** for calibration size n
and level α, q is the k-th smallest |residual| with
k = ⌈(n+1)(1−α)⌉, capped at n. No interpolation, no off-by-one variants.

## Verdict rule (frozen)

Per stat, at **80% nominal**, each conformal method separately:

- (a) its empirical coverage on the evaluation segment is within ±3 points
  of nominal — in [77%, 83%]; AND
- (b) its mean width is ≤ the width of the narrowest MAE band (among ±1×,
  ±2×) whose empirical coverage is ≥ the conformal method's coverage. If no
  MAE band's coverage reaches the conformal method's, (b) is judged against
  the widest band (±2× MAE). A method may not win by simply being wider than
  everything that covers as much.

A stat where both clauses hold is a **win** for that method; anything else is
not. The per-stat verdicts are the result. Pooled (all-stats) numbers are
reported second and cannot rescue a method that wins no stat: a method that
wins pooled but fails every stat individually has not won. The 50% and 90%
levels are supporting evidence, reported in full, and are not gated.

## Diagnostics (frozen, NOT gates)

- **Conditional coverage:** per stat, evaluation rows split into terciles of
  the projected value (cut points from the evaluation segment itself);
  coverage at 80% for split, rolling, and the ±1× band per tercile. This is
  where a method that covers on average but abandons stars or scrubs is
  visible.
- **Coverage over time:** per stat, split vs rolling coverage at 80% over
  the early / middle / late thirds of the evaluation segment, plus pooled
  monthly coverage. The drift-adaptation claim lives or dies here: if
  rolling does not hold coverage better than split in the early-season
  stretch (the first evaluation third contains Oct–Nov 2024), it has no
  advantage to sell.

## Live-record cross-check (bounded, stated plainly)

The full live counted record (the grading population: earliest recorded
projection per game/player/stat in the tracking store) lives on the VPS; the
store is not in this repo and this race does not touch the VPS. Two
cross-checks are possible from what the repo does carry, and only these two
will be reported:

1. **Status-quo coverage on the whole live record**, from the committed
   public snapshot's `prop_grading.aggregates` band shares (13,004 pre-tip
   rows, 2026-10-03 → 2026-11-02): ±1× coverage = On-the-number + Close
   share; ±2× adds Off. Directly comparable to the eval frame's status-quo
   coverage.
2. **Transported conformal quantiles on the latest live day**: the snapshot's
   `prop_grading.latest_day` carries every counted row for 2026-11-02
   (1,332 rows) with projection and actual. The eval frame's final split
   quantiles and final rolling quantiles are applied to those rows as-is.
   One day, a different model vintage (the live projections came from the
   production models of Oct–Nov 2026, not this race's fit) — directional
   evidence only, labelled as such wherever it appears.

If a cross-check cannot be computed from committed data, the results doc
says so rather than reaching for the VPS.

## What does not change

No production behaviour: no route, payload, manifest, model file, or UI is
touched. The race adds an evaluation module, a race script, tests, and these
docs. Nothing a reader sees changes whatever the verdict.

## Family accounting (multiple testing)

This is trial 1 of a new family (interval construction), and it is not a
model race: the fitted model is fixed across all arms — only the interval
construction varies — so the star-factor family's "the holdout is not unseen"
selection concern applies weakly here (no variant of the *model* was chosen
on this holdout by this race). What the holdout cannot do is certify
transport to the live season; that is what the bounded cross-check is for,
and any reader-facing adoption would still need Kevin's call on live
evidence. Three nominal levels × two conformal methods × four stats are all
reported in full; the verdict rule was frozen at one level (80%) and one
comparison (per stat) precisely so the rest is evidence, not a search for a
flattering cell.

## Log

- Trial 1 — MAE bands vs split conformal vs rolling (14-day) conformal
  (this document): result in `docs/conformal-props-results-2026-11.md`,
  whatever it is.
