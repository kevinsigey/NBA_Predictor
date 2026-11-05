# Conformal intervals for player props, trial 2 — race results

Preregistered in `docs/conformal-scaled-preregistration-2026-11.md`
(committed and pushed before the race ran; the spec was not edited
afterwards). Raw numbers: `docs/conformal-scaled-race-2026-11.json`.
Reproduce with `PYTHONPATH=src python scripts/conformal_scaled_race.py`.

## Verdict

**Both challengers fix the shape under the frozen rule — and the frozen
selection rule ships the SCALED method.** Per the preregistration, when
both fix the shape the narrower pooled width wins: scaled 6.49 vs
Mondrian 6.70 at 80%. Stage 2 therefore builds scaled (normalized)
rolling conformal into production.

The full shape table at 80% (coverage by evaluation-segment projection
tercile; the frozen band is 74–86%):

| Stat | Champion low / mid / high | Scaled low / mid / high | Mondrian low / mid / high |
|---|---|---|---|
| Points | 92.5 / 80.3 / **66.7%** | 84.3 / 78.1 / 76.8% | 80.4 / 79.3 / 79.6% |
| Rebounds | 91.8 / 81.4 / **66.2%** | 82.4 / 78.9 / 77.9% | 80.1 / 79.6 / 79.8% |
| Assists | 93.5 / 85.4 / **61.0%** | 85.4 / 78.9 / 75.2% | 80.6 / 79.3 / 80.0% |
| Threes | 95.0 / 81.6 / **62.8%** | 86.0 / 81.9 / **71.6%** | 80.2 / 80.7 / 78.9% |

Clause by clause, exactly as frozen:

- **Scaled:** clause 1 (all terciles in band) holds on 3 of 4 stats —
  threes fails it (high tercile 71.6%, below the 74% band floor).
  Clause 2 (no worst tercile under 70%): worst is that same 71.6% ✓.
  Clause 3 (marginal in 77–83% on all stats): 79.7–79.8% ✓.
  Clause 4 (width ≤ champion +10% per stat): scaled is *narrower* than
  the champion at 80% on all four stats ✓. **Fixes the shape.**
- **Mondrian:** clause 1 holds on all 4 stats — its terciles span only
  78.9–80.7% across the whole table, the flattest profile measured.
  Clause 2: worst tercile 78.9% ✓. Clause 3: 79.8–79.9% ✓. Clause 4:
  within the cap on all stats (rebounds 5.94 vs champion 5.91, the only
  stat where it is wider at all) ✓. **Fixes the shape.**

**Stated plainly, because the table shows it:** Mondrian's shape is the
better of the two — no tercile of any stat leaves 78.9–80.7%, where
scaled still leaves threes' stars at 71.6%. Scaled ships because the
selection rule was frozen before these numbers existed — both fix the
shape, narrower pooled width wins — and a rule rewritten after seeing
the table would not be a rule. The cost of the choice is on the record
here: the shipped method's weakest cell is threes' high tercile at
71.6%, and the production aggregate block (Stage 2) keeps measuring
live coverage in public, so if the live record ever argues for
revisiting the choice, it will argue with numbers.

Marginal coverage and width at 80%:

| Stat | Champion cov / width | Scaled cov / width | Mondrian cov / width |
|---|---|---|---|
| Points | 79.8% / 14.37 | 79.7% / 13.83 | 79.8% / 14.23 |
| Rebounds | 79.8% / 5.91 | 79.7% / 5.81 | 79.9% / 5.94 |
| Assists | 80.0% / 4.13 | 79.8% / 3.89 | 79.9% / 3.99 |
| Threes | 79.8% / 2.86 | 79.8% / 2.43 | 79.9% / 2.63 |

Pooled @80: champion 79.9% (width 6.82), scaled 79.8% (6.49), Mondrian
79.9% (6.70) — reported second, per the preregistration; the per-stat
clauses above are the result.

## Setup

- Track, split, calibration anchor identical to trial 1 (frozen history,
  production frame, production factories, date-aligned last-20%
  holdout: 36,551 rows, 226 game days, 2024-02-05 → 2025-04-13;
  calibration = first 56 days, 8,896 rows/stat; evaluation = 27,655
  rows/stat over 170 days).
- Both anchor checks passed: the fitted models reproduce the
  star-factor base arm (holdout MAEs 4.606 / 1.926 / 1.355 / 0.893),
  and the champion arm reproduces trial 1's published rolling numbers
  exactly (79.8% and widths 14.37 / 5.91 / 4.13 / 2.86).

## Supporting levels

At 50% and 90% the challengers hold nominal as well as the champion
(coverage within ~0.2 points of nominal nearly everywhere). One honest
asymmetry: at 50% the scaled intervals are *wider* than the champion's
(points 8.01 vs 7.30) while at 80% and 90% they are narrower — the
σ curve moves width out of the middle of the distribution into the
tails, which is precisely the shape correction working as intended.
Mondrian shows the same direction, smaller (50%: 7.78 points).

## Diagnostic — coverage over time: no stability was spent

Coverage @80 by evaluation third, champion / scaled:

| Stat | Early | Middle | Late |
|---|---|---|---|
| Points | 80.0 / 80.1% | 80.1 / 79.8% | 79.4 / 79.3% |
| Rebounds | 79.9 / 80.1% | 79.9 / 79.8% | 79.6 / 79.3% |
| Assists | 80.0 / 79.9% | 80.1 / 79.9% | 79.8 / 79.7% |
| Threes | 79.5 / 79.6% | 80.2 / 80.0% | 79.7 / 79.9% |

Scaled never sags more than 0.3 points below the champion's same-third
coverage on any stat — far inside the preregistration's 3-point
reporting threshold. Mondrian's thirds are equally stable (all within
79.5–80.3%). Trial 1's ratchet survives the shape fix intact.

## Diagnostic — the machinery, on the record

- **Mondrian fallback:** 0 invocations in 510 bin-days per stat; the
  smallest window bin held 431 residuals (the frozen floor is 100). On
  this track the fallback is a tripwire that never came close to
  firing; production keeps it anyway, because live early-season windows
  are thinner than this holdout's.
- **Scaled σ curves** (calibration-segment fits; bin edges are
  projection deciles, scales are mean |residual| after isotonic
  smoothing): points σ rises 2.47 → 6.47 from the lowest projection
  decile to the highest; rebounds 1.07 → 3.09; assists 0.66 → 2.37;
  threes 0.07 → 1.43 (lowest bin floored at 0.087). Volatility roughly
  doubles-to-triples across the projection range on every stat — the
  quantitative form of the defect: one half-width per stat was never
  going to fit both ends of that curve.

## What this does and does not establish

- Established on the frame: projection-aware conformal flattens the
  coverage shape the symmetric champion cannot — both constructions
  bring nearly every tercile into or beside the 74–86% band at nominal
  marginal coverage, at no width cost at 80% (both are narrower than
  the champion pooled). The selection between them was made by the
  frozen rule, and its cost (threes' stars at 71.6% under scaled vs
  78.9% under Mondrian) is printed above, not buried.
- Not established: transport to the live season. This holdout has now
  selected the champion (trial 1) and the shape fix (trial 2); as both
  preregistrations state, the shipped claim rests on the production
  design — rolling calibration from the live store's own current-season
  residuals — and on the live coverage figure Stage 2 publishes, which
  continues this measurement in public on data no race selected on.
- The point models are untouched, as are all routes, payloads,
  manifests and UI — by this stage. Stage 2 is the ship.
