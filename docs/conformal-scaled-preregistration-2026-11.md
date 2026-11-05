# Conformal intervals for player props, trial 2 — fixing the shape: preregistration

Written and committed BEFORE the race runs. The spec below is frozen: no
edits after the first race run starts, and no edit may rescue a marginal
result. Unlike trial 1, this trial ends in a ship: Kevin has approved
shipping conformal intervals for player props ("fix the terrible shape and
then ship it since it improves"), so the winner selected here — or the
trial-1 champion, if neither challenger fixes the shape — is the method
Stage 2 builds into production. Production calibration is a separate,
live-store rolling construction (Stage 2's own spec); this race selects
the *method*, on the same frozen track trial 1 used, so its table reads
directly against trial 1's.

## The question

Trial 1 (`docs/conformal-props-preregistration-2026-11.md`, results in
`docs/conformal-props-results-2026-11.md`, PR #170) established that
conformal intervals deliver their nominal coverage where the MAE bands do
not, and that rolling calibration (trailing 14 game days) is the most
faithful construction. It also measured the defect this trial exists to
fix: **coverage is an average that hides the shape.** At 80% nominal, every
symmetric per-stat interval over-covers low-projection lines (92–95%) and
under-covers high-projection lines (61–67%) — star performances are
intrinsically more volatile, and one half-width per stat cannot be both
the scrub's interval and the star's. Trial 1's champion numbers, the ones
a challenger must beat on shape without giving back:

| Stat | Rolling cov @80 | Rolling width @80 | Low / Mid / High tercile cov |
|---|---|---|---|
| Points | 79.8% | 14.37 | 92.5 / 80.3 / 66.7% |
| Rebounds | 79.8% | 5.91 | 91.8 / 81.4 / 66.2% |
| Assists | 80.0% | 4.13 | 93.5 / 85.4 / 61.0% |
| Threes | 79.8% | 2.86 | 95.0 / 81.6 / 62.8% |

**Can a projection-aware conformal construction flatten that shape —
every tercile covered near nominal — without surrendering the marginal
coverage or paying for it in width?**

## Population and harness (frozen, identical to trial 1)

Same track, same split, same calibration anchor, so the numbers are
comparable to trial 1's table cell for cell:

- Data: the frozen history `data/history/nba_history.json.gz` via
  `pipeline/history.py::load_history`, player rows with minutes > 0.
- Frame: `features/prop_model_frame.py::build_prop_model_frame`, built
  once over train+holdout.
- Split: `models/prop_holdout.py::_date_aligned_split`, chronological
  last-20%. Holdout 2024-02-05 → 2025-04-13, **36,551 rows over 226 game
  days**, every row carrying all four targets.
- Models: the production `train_player_stat_model` per stat, fitted once
  on the train segment, never refitted inside the holdout. Residual =
  actual − predicted on holdout rows.
- **Calibration segment:** the first 56 of the holdout's 226 game days
  (8,896 rows per stat). **Evaluation segment:** the remaining 170 game
  days (27,655 rows per stat), scored in per-game-day blocks. Nothing is
  shuffled and nothing is randomly split, anywhere.
- **Anchor check:** the champion arm (below) must reproduce trial 1's
  rolling numbers to rounding — coverage within ±0.15 points and mean
  width within ±0.05 of the table above, per stat — and the fitted
  models' full-holdout MAEs must again reproduce the star-factor base arm
  (points 4.606, rebounds 1.926, assists 1.355, threes 0.893). If they do
  not, the race is stopped and reported as a setup failure, not raced on
  a different track.
- Rolling window: trailing **14 game days** strictly before the day being
  scored, same as trial 1, with the same tripwire: a day is scored only
  if its window holds ≥ 500 residuals per stat (in this population the
  window is always full; the guard is a tripwire, not a filter).
- Quantile convention: identical to trial 1 — the k-th smallest score
  with k = ⌈(n+1)(1−α)⌉, capped at n. No interpolation.

## Methods (frozen)

All intervals are per stat, centred on the projection. One champion, two
challengers — exactly these, no others.

**Champion — rolling symmetric conformal (trial 1's winner).**
Half-width for a day = the conformal quantile of |residuals| over the
trailing 14 game days, one half-width per stat per day. Reproduced by the
trial-1 harness unchanged (`models/conformal_props.py::evaluate_stat`);
its numbers are the anchor check and the bar.

**Challenger A — scaled (normalized) rolling conformal.**
The half-width scales with a per-row volatility estimate σ̂(projection):

- σ̂ is fitted **on the calibration segment only**, per stat, and frozen
  for the whole evaluation segment — the production analogue of a
  pre-season fit. Construction: calibration rows are binned into 10
  equal-frequency bins by projected value (bin edges = the calibration
  segment's decile cut points; duplicate edges collapse, so a degenerate
  stat may have fewer bins). Each bin's raw scale is its mean |residual|.
  The bin scales are then passed through **isotonic regression**
  (non-decreasing in bin order, weighted by bin counts,
  pool-adjacent-violators) so the scale cannot decrease as the projection
  grows. σ̂(row) is the isotonic value of the bin the row's projection
  falls into; a projection outside the calibration range takes the
  nearest edge bin's value. A floor keeps degenerate low bins from
  producing zero-width intervals: σ̂ is at least **10% of the stat's
  calibration-segment mean |residual|**.
- Score: s = |residual| / σ̂(projected). For each evaluation day, q is
  the conformal quantile of s over the trailing 14 game days strictly
  before it (the window rows' scores use the frozen σ̂). A row's
  half-width that day is q × σ̂(its projected).

**Challenger B — Mondrian rolling conformal by projection tercile.**
A separate rolling quantile per projection bin:

- Bin edges are the calibration segment's projected-value terciles per
  stat, frozen. A row belongs to the low bin when projected ≤ edge 1,
  the mid bin when ≤ edge 2, the high bin above — the same boundary
  convention as trial 1's tercile diagnostic.
- For each evaluation day and each bin, the half-width is the conformal
  quantile of |residuals| over the trailing 14 game days strictly before
  it, restricted to window rows in that bin. **Frozen fallback:** if a
  bin's window holds fewer than **100** residuals, that bin that day
  uses the day's pooled rolling half-width (the champion's) instead.
  Fallback invocations are counted per stat and reported; a challenger
  that leans on the fallback is visible as one.

**Nominal levels:** 50%, 80%, 90% for all three methods. The verdict is
judged at 80% only; the other levels are supporting evidence, reported
in full.

**Reporting terciles (frozen).** Shape is reported on the
**evaluation segment's own projected-value terciles**, constructed
exactly as trial 1's diagnostic constructed them, so this trial's shape
table reads against trial 1's cell for cell. (The Mondrian method's
*assignment* bins are the calibration edges above; the *reporting*
terciles are the evaluation segment's, for comparability. The two are
close by construction and any gap between them is itself reported by
the Mondrian fallback/bin-count accounting.)

## Verdict rule (frozen)

A challenger **fixes the shape** at 80% nominal if and only if ALL of:

1. **Shape:** for at least 3 of the 4 stats, every reporting tercile's
   coverage at 80% lies in **[74%, 86%]**.
2. **No abandoned stat:** no stat's worst reporting-tercile coverage at
   80% is below **70%**.
3. **Marginal held:** the challenger's marginal (whole-stat) coverage
   at 80% lies in **[77%, 83%]** for all four stats — trial 1's clause
   (a), unchanged.
4. **Width paid for honestly:** the challenger's mean width at 80% is
   at most **1.10 × the champion's** mean width at 80%, per stat, all
   four stats.

Selection among challengers:

- Exactly one fixes the shape → that method ships.
- Both fix the shape → the one with the lower **pooled mean width at
  80%** (pooled = all four stats' evaluation rows together) ships. A
  tie within 0.5% of pooled width goes to the **scaled** method, whose
  half-widths vary continuously with the projection rather than in
  three steps.
- **Neither fixes the shape → the champion (rolling symmetric) ships.**
  Kevin approved shipping regardless, because the champion already
  improves on the status quo; in that case the results doc states in
  plain words that the shape remains unfixed, and by how much.

The winner must not sacrifice trial 1's temporal stability to win: the
time-thirds diagnostic (below) is reported for all three methods, and
if the selected method's worst third on any stat sags more than 3
points below the champion's same-third coverage on that stat, the
results doc says so in its verdict paragraph — the selection rule does
not change, but the cost is printed next to the win.

## Diagnostics (frozen, NOT gates)

- **Per stat × tercile coverage at 80%** for champion and both
  challengers (the shape table; the verdict reads clauses 1–2 off it).
- **Marginal coverage and mean width at 50% / 80% / 90%** for all three
  methods, per stat and pooled.
- **Coverage over time:** per stat, all three methods' coverage at 80%
  over the early / middle / late thirds of the evaluation segment
  (trial 1's construction), plus pooled monthly coverage for the
  selected method.
- **Mondrian fallback count** per stat (bin-days served by the pooled
  fallback), and the window bin sizes behind it.
- **Scaled σ̂ summary** per stat: the fitted bin edges and isotonic
  scales, so the shape of the volatility curve is on the record.

## What does not change in this stage

No production behaviour: no route, payload, manifest, model file, or UI
is touched by the race itself. Stage 2 (the ship) is a separate body of
work in the same PR, built on whichever method this race selects, with
its own tests.

## Family accounting (multiple testing)

Trial 2 of the interval-construction family, on the same holdout trial 1
used. The selection concern is stronger here than in trial 1 and is
stated plainly: two challenger constructions are being compared on a
holdout that has already selected the champion, so this holdout cannot
certify that the winner's coverage transports to the live season. What
carries the shipped claim is the production design — calibration from
the live tracking store's own current-season residuals, rolling — and
the aggregate live-coverage figure Stage 2 publishes beside the bands,
which keeps measuring the shipped intervals against the counted record
in public. The verdict rule above is frozen at one level (80%), one
shape band, and one width cap precisely so the rest of the table is
evidence, not a search for a flattering cell. All numbers — including a
challenger that fixes nothing — are reported in full.

## Log

- Trial 1 — MAE bands vs split vs rolling conformal: rolling won all
  four stats; shape defect measured (stars 61–67% at "80%"). PR #170,
  `docs/conformal-props-results-2026-11.md`.
- Trial 2 — champion vs scaled vs Mondrian (this document): result in
  `docs/conformal-scaled-results-2026-11.md`, whatever it is.
