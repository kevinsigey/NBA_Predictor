/**
 * The post-game review: which projections the model got right, which it got
 * wrong, and by how much.
 *
 * A projection is judged against a BAND, not against zero: the props model's
 * own holdout error for that stat (points 4.6, rebounds 1.9, assists 1.3,
 * threes 0.9 today). A player who misses by two points has not been beaten
 * by the model -- the model was two points wrong, which is what its error
 * estimate says it can be. The band moves with each retrain, because it is
 * read from the model rather than written down here.
 *
 * Three rules that are the whole of it:
 *
 * * **A gap exactly equal to the band is on the number.** The band is the
 *   model's own error, so a miss of exactly that size is the model being
 *   wrong as much as it said it would be. Counting it as a beating would
 *   make the review's verdict depend on a rounding.
 * * **A player who did not play is not judged.** A projection for someone
 *   who never got on the floor has nothing to be right or wrong about.
 * * **A projection rebuilt after tip-off is never judged.** It was made with
 *   the result already known, so beating it would be beating a copy of the
 *   answer.
 */

export interface ReviewProjection {
  player_id: string;
  player_name: string;
  stat: string;
  value: number;
  /** Built after tip-off: shown elsewhere, never judged here. */
  rebuilt?: boolean;
}

export interface ReviewActual {
  player_id: string;
  stat: string;
  value: number;
}

export interface ReviewRow {
  player_id: string;
  player_name: string;
  stat: string;
  projected: number;
  actual: number;
  /** Actual minus projection: positive means the player beat the projection. */
  gap: number;
  /** The miss in units of the stat's own error: On the number (<= 1/2 the
   *  band), Close (<= the band), Off (<= twice it), Way off (beyond). The
   *  same bands the backend grades with, so the page and the payload agree. */
  grade: string;
}

/** The grade for one miss, in units of the stat's own band (its holdout MAE). */
export function gradeForGap(gap: number, band: number): string {
  const ratio = Math.abs(gap) / band;
  if (ratio <= 0.5 + BAND_EPSILON) return "On the number";
  if (ratio <= 1 + BAND_EPSILON) return "Close";
  if (ratio <= 2 + BAND_EPSILON) return "Off";
  return "Way off";
}

export interface ReviewResult {
  /** Beat the projection: the actual was more than the band above it. */
  beat: ReviewRow[];
  /** Fell short: the actual was more than the band below it. */
  short: ReviewRow[];
  /** Player lines inside the band: "on the number". */
  onNumber: number;
  /** Player lines judged at all, so the count can be read "14 of 22". */
  total: number;
}

/** How many rows a column shows. The biggest gaps first, so the cap keeps
 *  the ones a reader would want to see. */
export const REVIEW_ROWS = 5;

/** The tolerance on the band comparison.
 *
 *  15.4 - 20 is -4.6000000000000005 in binary floating point, and a band of
 *  4.6 would call that a miss bigger than the band. A gap that is the band's
 *  size is on the number, so the comparison carries a tolerance far finer
 *  than the band itself is measured (the band is two decimal places).
 */
const BAND_EPSILON = 1e-9;

/**
 * The review, as three lists.
 *
 * `bands` is the props model's error per stat. A stat with no band is not
 * judged: there is no measured error to judge it against, and a band of zero
 * would make every miss a beating.
 */
export function reviewPlayers(
  projections: ReviewProjection[],
  actuals: ReviewActual[],
  bands: Record<string, number | null | undefined> | null | undefined,
  limit: number = REVIEW_ROWS,
): ReviewResult {
  const result: ReviewResult = { beat: [], short: [], onNumber: 0, total: 0 };
  if (!bands) return result;

  const actualByKey = new Map<string, number>();
  for (const actual of actuals ?? []) {
    actualByKey.set(`${actual.player_id}|${actual.stat}`, actual.value);
  }

  const beat: ReviewRow[] = [];
  const short: ReviewRow[] = [];
  let onNumber = 0;
  let total = 0;

  for (const projection of projections ?? []) {
    if (projection.rebuilt) continue;
    const actual = actualByKey.get(`${projection.player_id}|${projection.stat}`);
    if (actual === undefined) continue;          // did not play
    const band = bands[projection.stat];
    if (typeof band !== "number" || !Number.isFinite(band)) continue;

    total += 1;
    const gap = actual - projection.value;
    const row: ReviewRow = {
      player_id: projection.player_id,
      player_name: projection.player_name,
      stat: projection.stat,
      projected: projection.value,
      actual,
      gap,
      grade: gradeForGap(gap, band),
    };
    if (gap > band + BAND_EPSILON) beat.push(row);
    else if (gap < -band - BAND_EPSILON) short.push(row);
    else onNumber += 1;
  }

  const byGap = (a: ReviewRow, b: ReviewRow) => Math.abs(b.gap) - Math.abs(a.gap);
  result.beat = beat.sort(byGap).slice(0, limit);
  result.short = short.sort(byGap).slice(0, limit);
  result.onNumber = onNumber;
  result.total = total;
  return result;
}
