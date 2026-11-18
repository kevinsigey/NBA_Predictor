import { stat } from "../predictor-ui";
import type { Prediction } from "../api/client";

// Lead with whoever the model favours. Home wins a 50/50 tie, matching the
// `>= 0.5` rule used to judge the pick after the game.
export function favourite(p: Prediction, home: string, away: string): { team: string; prob: number } {
  return p.home_win_probability >= 0.5 ? { team: home, prob: p.home_win_probability } : { team: away, prob: 1 - p.home_win_probability };
}

/**
 * The projected margin, written so it never contradicts the pick. The win and
 * margin numbers come from separate models: when they point at different
 * teams, or the margin rounds to nothing, say "Toss-up" rather than print
 * "BOS to win" beside "MIA by 0.6".
 */
export function marginLine(p: Prediction, home: string, away: string): string {
  const pick = favourite(p, home, away);
  const team = p.predicted_margin >= 0 ? home : away;
  const value = Math.abs(p.predicted_margin);
  return team === pick.team && value >= 0.5 ? `${team} by ${stat(value)}` : "Toss-up";
}

/** Whether the pick (home at >= 50%) matched the result. */
export const pickWon = (p: Prediction, homePts: number, awayPts: number) => p.home_win_probability >= 0.5 === homePts > awayPts;

/** The market's favourite and its probability, or null when no pre-tip moneyline exists (never defaulted). */
export function marketFavourite(p: Prediction, home: string, away: string): { team: string; prob: number } | null {
  const m = p.market_home_win_probability;
  if (m == null) return null;
  return m >= 0.5 ? { team: home, prob: m } : { team: away, prob: 1 - m };
}
