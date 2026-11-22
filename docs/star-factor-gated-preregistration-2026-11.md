# Star factor, trial 2: the star_out-GATED variant — preregistration

Written and committed BEFORE the race runs. The spec below is frozen: no edits
after the first race run starts, and no edit may rescue a marginal result.

## The variant (frozen)

Trial 1's seven star-factor columns (`vac_pts`, `vac_reb`, `vac_ast`,
`vac_fg3m`, `vac_min`, `vac_n`, `star_out` from
`features/star_factor.py`), with one change: the five magnitude columns
(`vac_pts`, `vac_reb`, `vac_ast`, `vac_fg3m`, `vac_min`) are **zeroed on every
row where `star_out == 0`**. Vacated production only registers when the absent
player is the team's #1 usage option. `vac_n` and `star_out` themselves stay
ungated.

The gating lives inside the star-factor feature path
(`features/star_factor.py::apply_star_out_gate`, applied by
`build_prop_model_frame`), so training and serving compute identical columns
through the same function — the no-skew property trial 1 shipped with. The
base arm is untouched: it is the 12 production features
(`points_roll … min_ewm5`) with no star columns at all.

Motivation (from trial 1, stated before this race): the ungated features moved
the absence-slice bias the right way but converted to no MAE, and trial 1's
slice was dilute — `vac_n > 0` covers 95.8% of holdout rows because sticky
top-8 rotations almost always have *someone* absent, while the causal study's
measured effect (+2.17 pts / +0.61 reb / +0.55 ast / +0.18 threes) lives in the
~12% of team-games where the absent player is a star. The gate concentrates
the columns on that population.

## Harness (frozen, identical to trial 1)

- Data: the frozen history `data/history/nba_history.json.gz` via
  `pipeline/history.py::load_history`, player rows with minutes > 0 — the
  same input trial 1 raced on.
- Harness: `models/prop_holdout.py::prop_holdout_metrics`, chronological
  last-20% holdout, date-aligned split (the harness's own), one split for all
  four markets.
- Factories: the production `train_player_stat_model` per market, unchanged.
- Arms: base = 12 production features; candidate = base 12 + the 7 gated star
  columns.
- Comparison: per-stat holdout MAE (base, candidate, naive rolling-mean MAE),
  Δ = candidate − base, and a paired bootstrap over holdout rows — 2,000
  resamples, seed **20261102**, percentile 95% CI on the per-row Δ|error|.
- Replication anchor: the base arm must reproduce trial 1's base MAEs
  (points 4.606, rebounds 1.926, assists 1.355, threes 0.893) and naive MAEs
  (4.696 / 1.956 / 1.374 / 0.896) to rounding. If it does not, the race is
  stopped and reported as a setup failure, not raced on a different track.

## Slices (transparency, NOT the gate)

Per stat, on `star_out == 1` rows and on `star_out == 0` rows of the holdout:
MAE and signed bias (mean of prediction − actual), base vs candidate, with
row counts. The study's effect lives in the star-out slice; that is where the
candidate must show its work if the effect is real.

## Ship gate (frozen)

Merge only if ALL hold:

(a) at least one stat's holdout MAE improves AND no stat worsens beyond noise
    (any worsening stat's paired-bootstrap 95% CI must include zero) —
    trial 1's gate;
(b) the improving stat's paired-bootstrap 95% CI excludes zero. A borderline
    pass whose CI touches zero counts as NOT confirmed: the PR stays open
    and is reported that way;
(c) branch CI fully green.

Any other outcome: the PR stays open and unmerged with the full race table
and this family log in its body.

## Family accounting (multiple testing)

This is trial 2 in the star-factor family, and it is not independent of
trial 1:

- Trial 1 (ungated, PR #164): FAILED its gate. No stat improved; rebounds and
  threes worsened ~0.001 with CIs entirely above zero.
- Trial 2 (this variant): motivated by trial 1's slice analysis **on the same
  holdout**. The holdout is therefore not unseen for this family, and the
  variant choice itself is a selection on trial 1's results. Stated plainly
  so the reader can discount accordingly.

Chance accounting: the gate needs ≥1 of 4 stats improving with its 95% CI
excluding zero (a one-sided 2.5% tail under the null) — ≈ 9.6% for this trial
alone under a full null, ≈ 18% across the family's 8 stat-tests if both
trials were null. A pass here **nominates** the gated features; it does not
prove them. No third variant in this family may be raced without a new
preregistration that names this family count (2 trials, 8 stat-tests) and
uses data this family has not selected on.

## Log

- Trial 1 — ungated vacated production (PR #164): gate FAILED (see PR body
  for the full table).
- Trial 2 — star_out-gated (this document): result appended in the race
  report `docs/star-factor-gated-race-2026-11.md`, whatever it is.
