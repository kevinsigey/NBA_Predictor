# NBA: in-season home-margin drift correction, preregistration (2026-11-20)

**Why.** In the paper ledger's score rows (219 settled games this season, earliest pre-tip observation), the model
gives the home team 56.3% on average. The Kalshi mid gives 53.4%, and home teams won 49.3%. Model log loss is
0.053 worse than the mid [-0.075, -0.032]. The model learns its home edge from earlier seasons and has no way to
notice that this season's edge is smaller. No number for the candidate below has been computed.

## Candidate (one, fixed)

**R1, running in-season margin bias.** Within each season, for a game on date D:

    delta = mean(predicted_margin - actual_margin) over that season's completed games dated < D
            (0 if fewer than 100 such games)
    margin' = predicted_margin - delta
    p' = Phi(margin' / sigma)

Here predicted_margin is home minus away. sigma is the same margin sigma the base prediction uses, so the base is
p = Phi(predicted_margin / sigma). Only the margin and the win probability move. The total is untouched.

## Data: out-of-sample predictions, season folds

For each of the last 4 completed seasons in the training history (`data/history/nba_history.json.gz`), train the
SERVED margin model, with the same features, candidate and params as `pipeline/retrain.py`, on all earlier seasons
only. Predict every game of the held-out season. sigma is fitted on training residuals only, as retrain does.
Games are taken in date order. The 2026-27 season is reported separately, from served predictions (the
`/picks/history` rows: last prediction created before tip), and is descriptive only.

## Ship rule (all must hold)

1. **Primary:** pooled over the 4 held-out seasons, the paired per-game delta = LL(p) - LL(p'), with LL clipped at
   0.001. Game-level bootstrap, 4000 draws, seed 20261017. low95 > 0.
2. **Nothing else worse:**
   - margin MAE(base) - MAE(margin') must not have upper95 < 0;
   - no single held-out season has mean LL delta with upper95 < 0.

**If it passes:** a follow-up PR applies R1 to served margins and win probabilities behind a flag, default off,
storing the raw values beside them. Switching it on needs Kevin's go.
**If it fails:** docs only.

## Reported, not gating

- delta at season end for each held-out season.
- Per-season LL and MAE deltas.
- The 2026-27 served-prediction descriptive: the same R1 applied to served rows, LL delta with CI.
