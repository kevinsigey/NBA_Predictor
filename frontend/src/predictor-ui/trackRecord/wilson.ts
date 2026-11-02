// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The plausible range behind a hit rate.
 *
 * Ported unchanged from NBA's `trackRecordViews.ts`, where it was the one thing
 * that stopped "11 of 24" reading as a settled fact. The arithmetic is written
 * once here so five pages cannot drift apart on it.
 */

/**
 * Wilson score interval. Returns `[low, high]` as shares in [0, 1], or `null`
 * when nothing is graded — never `[0, 0]`, which would draw a bar pinned at 0%
 * and read as a model that has never been right.
 */
export function wilsonInterval(correct: number, n: number, z = 1.96): [number, number] | null {
  if (n <= 0) return null;
  const p = correct / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}
