// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * Returns the colour tier for a rank within a field of size n.
 * Thirds: good (top third), mid (middle third), bad (bottom third).
 * Rank 1 is always good, rank n is always bad.
 */
export function rankTier(rank: number, n: number): "good" | "mid" | "bad" {
  if (n <= 1) return "good"; // degenerate, but rank 1 is best
  if (rank === n) return "bad"; // rank n is always bad
  const third = Math.ceil(n / 3);
  if (rank <= third) return "good";
  if (rank <= 2 * third) return "mid";
  return "bad";
}