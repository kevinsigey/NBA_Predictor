# NBA-4: rest, streak and altitude per team (flag default OFF)

Raw results: `docs/nba-4-rest-altitude.json`. Tool: `tools/nba4_rest_altitude_eval.py`.

## Setup

- Games: `data/cache/training/games_all.json` (gitignored), 9,976 rows after feature NaN drop, 2018-10 to 2026-06.
- Walk-forward: expanding windows, 4 test windows, first 40% of dates train. Pooled out-of-fold n = 6,090.
- Paired bootstrap: 95% interval of the mean per-game difference (candidate minus baseline), 2,000 resamples, seed 20261103.
- Ship rule (strict): both primary intervals (win log-loss, margin absolute error) exclude zero in the improving direction, AND no reported metric's point estimate is worse than baseline.

## Results

| Metric | Baseline | rest_altitude | Delta [95% CI] | +interactions | Delta [95% CI] |
|---|---|---|---|---|---|
| win log-loss (primary) | 0.62430 | 0.62296 | -0.00134 [-0.00300, +0.00038] | 0.62303 | -0.00128 [-0.00306, +0.00057] |
| win Brier | 0.21740 | 0.21672 | -0.00068 | 0.21676 | -0.00064 |
| win AUC | 0.7008 | 0.7032 | +0.00244 | 0.7027 | +0.00189 |
| margin abs error (primary) | 11.0398 | 11.0073 | -0.03247 [-0.05749, -0.00784] | 11.0067 | -0.03314 [-0.05911, -0.00690] |
| margin RMSE | 14.1245 | 14.0919 | -0.03253 | 14.0925 | -0.03197 |

Brier and RMSE are point estimates only (no interval was computed for them).

## Verdict

- rest_altitude: **does not ship.** The margin improvement is real (interval excludes zero), but the win log-loss interval includes zero, so the primary win criterion fails.
- +interactions: **does not ship.** Same result. The visitor interactions add nothing measurable over rest_altitude alone, so they are not justified on this data.
- Both arms stay behind `REST_ALTITUDE_ENABLED = False` (`src/nba_predictor/features/build.py`). Nothing is wired into serving and no model artefact is touched.

## What changed in code

- `features/rest_travel.py`: `rest_features` (rest capped at 7 days, `is_first_game` flag instead of the 99 sentinel), `HIGH_ALTITUDE_HOME = {DEN, UTA}`, `is_high_altitude_home`.
- `features/build.py`: opt-in `rest_altitude` and `rest_altitude_interactions` on `build_feature_frame` and `build_training_frame`. Team-based rest, team-based streak, capped rest, altitude with Utah, and the visitor back-to-back interactions. The default path is unchanged: `FEATURE_COLUMNS` and the frame hash match origin/main exactly.
- `tests/test_rest_altitude.py`: 15 tests covering the sentinel, role-split rest, team streak against an independent oracle, Utah altitude, and the interactions.

## Defects found

1. **99 sentinel.** `compute_rest_days` returns 99 for a team's first game in the data, and season gaps run to about 180 days. Both reach the default `home_rest_days` / `away_rest_days` columns uncapped. Fixed in the flagged path only, because the committed models were fitted on the raw values.
2. **Role-split rest (demonstrated).** The default measures a home game's rest from the last home game and an away game's from the last away game. In the test fixture DEN plays at home on Jan 3 and away on Jan 4, a back-to-back, but the default reads about 24 days of rest for the Jan 4 game. The flagged path reads 0. Fixed in the flagged path only.
3. **Role-split streak (demonstrated).** The default streak uses only a team's home results for home games and only its away results for away games. The flagged path uses all of its results. The test checks every row against an oracle built from raw games.
4. **Utah altitude is 0 in `team_reference`.** `is_high_altitude` uses `altitude_ft > 1000`, so Utah (about 4,226 ft) is not flagged. The flagged path uses the DEN/UTA set instead. The default still reads Utah as 0. Recorded here and not corrected in the default path.
5. **Home/away mix-up check.** Verified no mix-up: per-team venue history, the travel path, and both sides' rest and fatigue read the team's own prior games. No code change was needed.

## Not run

- Calibration gap and the total model were not evaluated for this arm. The ship rule did not need them, since neither arm passed the primary criterion.
- No retrain, no walk-forward across other season splits, no per-season breakdown.
- Logistic regression emits ConvergenceWarnings on these features (unscaled inputs). Same in the baseline arm, so it does not bias the comparison, but the fit was not tuned.
- CodeRabbit was not triggered on this PR.
