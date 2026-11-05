# Conformal intervals for player props — race results

Preregistered in `docs/conformal-props-preregistration-2026-11.md` (committed
before the race ran; the spec was not edited afterwards). Raw numbers:
`docs/conformal-props-race-2026-11.json`. Reproduce with
`PYTHONPATH=src python scripts/conformal_props_race.py`.

## Verdict

**Both conformal methods win all four stats under the frozen rule.** At 80%
nominal, per stat:

| Stat | Split cov | Split width | Rolling cov | Rolling width | ±1× MAE band cov (width) | ±2× MAE band cov (width) |
|---|---|---|---|---|---|---|
| Points | 79.0% | 14.08 | **79.8%** | 14.37 | 60.5% (9.26) | 88.7% (18.51) |
| Rebounds | 79.0% | 5.78 | **79.8%** | 5.91 | 61.0% (3.86) | 88.9% (7.73) |
| Assists | 80.4% | 4.17 | **80.0%** | 4.13 | 63.4% (2.71) | 87.8% (5.42) |
| Threes | 78.2% | 2.73 | **79.8%** | 2.86 | 62.5% (1.80) | 87.1% (3.59) |

Clause (a) — coverage within ±3 points of 80%: split lands 78.2–80.4%,
rolling 79.8–80.0% on every stat. Clause (b) — not winning by width: the
narrowest MAE band covering as much is ±2× on every stat, and both conformal
methods are 20–25% narrower than it. A reader who wants an ~80% interval
today has to take ±2× MAE and actually gets ~88% at width 18.5 (points);
conformal gives 79–80% at width ~14.1–14.4.

Rolling is the stronger arm: coverage closest to nominal on every stat at
every level, the most stable over time, and the best transport to the live
day (below) — at a small width cost against split on three stats (assists it
is both closer and narrower). If anything ships, the evidence points at
rolling. **Nothing ships on this document**: the PR stays open; the call is
Kevin's.

Two caveats travel with the verdict, both measured below: coverage is an
average — star lines are covered far less than scrub lines by every
symmetric method — and quantiles calibrated on this holdout under-cover on
the live 2026-27 day they were transported to. An interval feature readers
rely on would need rolling calibration fed by the live tracking store's own
residuals, which is the rolling design pointed at live data.

## Setup

- Track: the star-factor races' — frozen history, production frame (12
  features), production `train_player_stat_model` per stat, date-aligned
  last-20% holdout. The anchor check passed: full-holdout MAEs reproduce the
  star-factor base arm exactly (points 4.606, rebounds 1.926, assists 1.355,
  threes 0.893; naive 4.696 / 1.956 / 1.374 / 0.896).
- Holdout: 36,551 rows, 2024-02-05 → 2025-04-13, 226 game days. Calibration
  segment = first 56 days (8,896 rows/stat, → 2024-04-06); evaluation = the
  remaining 170 days, 27,655 rows/stat. Bands use the manifest MAEs exactly
  as published (4.6286 / 1.9315 / 1.3548 / 0.8986).

## All levels (supporting evidence)

Coverage at 50% / 90% nominal tells the same story as 80%:

| Stat | Method | Cov @50 | Cov @80 | Cov @90 |
|---|---|---|---|---|
| Points | split | 48.6% | 79.0% | 89.4% |
| | rolling | 49.8% | 79.8% | 89.8% |
| | ±1× band (no nominal) | — | 60.5% | — |
| Rebounds | split | 49.0% | 79.0% | 89.2% |
| | rolling | 50.0% | 79.8% | 89.9% |
| | ±1× band | — | 61.0% | — |
| Assists | split | 49.5% | 80.4% | 90.0% |
| | rolling | 49.8% | 80.0% | 90.0% |
| | ±1× band | — | 63.4% | — |
| Threes | split | 48.6% | 78.2% | 89.3% |
| | rolling | 49.9% | 79.8% | 89.9% |
| | ±1× band | — | 62.5% | — |

Rolling is within 0.2 points of nominal at 50% and 90% on every stat; split
drifts up to 1.8 points low (threes @80). Pooled across stats @80: split
79.2%, rolling 79.9%, ±1× band 61.8%, ±2× band 88.1% — reported second, per
the preregistration; the per-stat verdicts above are the result.

## Diagnostic 1 — conditional coverage: the average hides the stars

Coverage @80 by projected-value tercile (per stat, split / rolling / ±1×
band):

| Stat | Low tercile | Mid tercile | High tercile |
|---|---|---|---|
| Points (≤6.7 / ≤11.9 / above) | 92.3 / 92.5 / 77.2% | 79.2 / 80.3 / 57.4% | **65.7 / 66.7 / 47.0%** |
| Rebounds | 91.7 / 91.8 / 77.0% | 80.0 / 81.4 / 59.3% | **65.2 / 66.2 / 46.7%** |
| Assists | 93.6 / 93.5 / 87.9% | 85.9 / 85.4 / 59.1% | **61.6 / 61.0 / 43.0%** |
| Threes | 94.2 / 95.0 / 92.6% | 79.8 / 81.6 / 53.3% | **60.7 / 62.8 / 41.4%** |

Every symmetric, per-stat interval over-covers low lines (~92–95%) and
under-covers high lines (~61–67%): star performances are intrinsically more
volatile, and one half-width per stat cannot be both. Conformal is better
than the band in every tercile, but it does not fix the shape. A
projection-scaled variant (conformal on |residual| / projection, or
tercile-conditional calibration) is the obvious next trial if this family
continues — it is NOT part of this race and would need its own
preregistration.

## Diagnostic 2 — coverage over time: rolling's edge is stability

Coverage @80 by evaluation third (early third runs 2024-04-07 → 2024-12-12
and contains the Oct–Nov 2024 early season):

| Stat | Early split / rolling | Middle split / rolling | Late split / rolling |
|---|---|---|---|
| Points | 79.2 / 80.0% | 79.7 / 80.1% | 78.3 / 79.4% |
| Rebounds | 79.7 / 79.9% | 79.5 / 79.9% | 77.7 / 79.6% |
| Assists | 80.7 / 80.0% | 80.9 / 80.1% | 79.6 / 79.8% |
| Threes | 78.4 / 79.5% | 78.5 / 80.2% | 77.8 / 79.7% |

The drift in this holdout is mild — the frozen Feb–Apr 2024 calibration
mostly transfers across the season boundary, so there is no early-season
collapse for rolling to rescue. What rolling does is refuse to sag: it never
leaves 79.4–80.2% in any third of any stat, while split drifts up to 2.3
points low by the late third (rebounds 77.7%). Pooled monthly, rolling ≥
split in all eight evaluation months, the gap widening late (Apr 2025:
78.4% vs 77.2% split — monthly figures in the JSON). The adaptation claim
survives in its modest form: not a rescue, a ratchet.

## Live-record cross-checks (bounded, as preregistered)

The full live per-row record is in the VPS tracking store, not this repo;
the two checks below are everything the committed snapshot supports.

**1. Status quo on the whole live counted record** (snapshot aggregates,
13,004 pre-tip lines, 2026-10-03 → 2026-11-02): ±1× MAE covers 52.3%
(points), 55.2% (rebounds), 55.5% (assists), 57.1% (threes); ±2× covers
81.9% / 84.4% / 84.3% / 86.0%. Worse than the eval frame's 60.5–63.4% at 1× —
the live model's errors run above the manifest's holdout MAEs (live points
MAE 5.55 vs band 4.63), so the fixed bands under-cover live even more than
they under-cover on the frame. The status quo's problem is not an artefact
of the evaluation frame.

**2. Transported quantiles on the latest live day** (2026-11-02, 1,332
counted rows; this race's quantiles applied as-is — different model vintage,
one day, directional only): at 80% nominal, split covers 67.9 / 71.5 / 71.2
/ 67.3% and rolling 71.2 / 76.3 / 72.4 / 70.3% across the four stats; even
±2× MAE only reaches 80.2–83.5% that day and ±1× just 45.9–53.2%. Rolling
transports best on every stat, but nothing calibrated on 2024-25 residuals
holds nominal on a Nov 2026 day whose errors run ~20% larger. Read plainly:
**the interval half-widths must be calibrated on the current season's own
residuals** — a shipped version calibrates rolling windows from the tracking
store as the season accumulates, exactly the mechanism this race validates
on the frame. A frozen pre-season calibration (split) is the weakest of the
three designs under real drift, on both the frame's late third and the live
day.

## What this does and does not establish

- Established on the frame: conformal intervals deliver their nominal
  coverage at 50/80/90 where MAE bands deliver ~61% (±1×) or ~88% (±2×) with
  no target at all; rolling calibration is the most faithful and most
  stable construction, 20–25% narrower than the narrowest band that covers
  as much.
- Established as limits: symmetric per-stat intervals mis-cover by
  projection size (stars ~2-in-3 at "80%"); calibration does not transport
  across seasons/model vintages at nominal.
- Not measured: any reader-facing design, any asymmetric or scaled variant,
  any change to the point models. The models themselves are untouched, as
  are all routes, payloads, manifests and UI.
