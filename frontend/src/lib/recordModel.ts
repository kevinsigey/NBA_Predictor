import type { CalibrationSeries, MarketTally, RecentPick, RecordModel, WeekPoint } from "../predictor-ui";
import { kickoff } from "../predictor-ui";
import type { TrackRecord, TrackRecordPick } from "../api/client";

/**
 * NBA's adapter: its `/hub/track-record` payload onto the shared `RecordModel`.
 * The kit renders the model, so no part of this wire shape reaches a kit
 * component.
 *
 * NBA is the one sport whose headline is the PRE-TIP-OFF subset, so the model is
 * built with `headline: "preKickoff"` and the big number comes from `pre_tip`.
 * `counted` still carries the whole record, per the model's contract, even
 * though this page's headline does not lead with it.
 */

// Plain names for each tracked market, in the order they read best. These are
// NBA's own words; the shared kit takes the label from the adapter, so this page
// keeps the wording it already ships.
const MARKETS: Record<string, string> = {
  game_outcome: "Winner pick",
  h2h: "Moneyline",
  spread: "Spread",
  total: "Total points",
  player_props: "Player props",
};
const ORDER = Object.keys(MARKETS);
const rank = (market: string) => (ORDER.includes(market) ? ORDER.indexOf(market) : ORDER.length);

/** A market's reader-facing name. A raw key never reaches the page. */
export const labelFor = (market: string): string => MARKETS[market] ?? market;

/** The API's confidence buckets arrive under the aggregate's own field names. */
type RawBucket = { bucket: string; total_predictions: number; correct_predictions: number };
type RowWithBuckets = TrackRecord & { confidence_buckets?: RawBucket[] };

const weekPoint = (w: { week_start: string; n: number; correct: number; tracked: boolean }): WeekPoint => ({
  weekStart: w.week_start,
  n: w.n,
  correct: w.correct,
  tracked: w.tracked,
});

const toPick = (market: string, p: TrackRecordPick): RecentPick => ({
  id: `${market}-${p.game_id}-${p.created_at}`,
  label: labelFor(market),
  pick: p.pick,
  actual: p.actual,
  hit: p.hit,
  when: p.gameday ? kickoff(p.gameday) : "—",
  madeBeforeKickoff: p.made_before_tip,
});

export function toRecordModel(rows: TrackRecord[]): RecordModel {
  // Settled markets only: a market the backend cannot judge shows no rate.
  const settled = [...rows].filter((r) => r.settled).sort((a, b) => rank(a.market) - rank(b.market));
  const lead = settled.find((r) => r.market === "game_outcome") ?? settled[0];

  const markets: MarketTally[] = settled.map((r) => ({
    key: r.market,
    label: labelFor(r.market),
    counted: { correct: r.correct_predictions, n: r.total_predictions },
    preKickoff: r.pre_tip ? { correct: r.pre_tip.correct_predictions, n: r.pre_tip.total_predictions } : undefined,
  }));

  const weeks = (lead?.pre_tip?.weekly ?? lead?.weekly ?? []).map(weekPoint);

  const calibration: CalibrationSeries[] = (settled as RowWithBuckets[])
    .filter((r) => r.confidence_buckets?.length)
    .map((r) => ({
      key: r.market,
      label: labelFor(r.market),
      buckets: r.confidence_buckets!.map((b) => ({
        bucket: b.bucket,
        n: b.total_predictions,
        correct: b.correct_predictions,
      })),
    }));

  // The latest counted pre-tip picks, newest first. The backend ships each
  // market's bounded `recent_picks` (the full per-pick disclosure moved to
  // /hub/track-record/picks); the adapter merges and sorts because the kit
  // only slices, and a formatted date string would sort alphabetically. The
  // counted/pre-tip filter stays here too, so the block's population is the
  // headline's even if a row's list ever widens.
  const recent = rows
    .flatMap((r) => (r.recent_picks ?? []).map((p) => ({ market: r.market, p })))
    .filter(({ p }) => p.counted && p.made_before_tip)
    .sort((a, b) => (b.p.gameday ?? b.p.created_at).localeCompare(a.p.gameday ?? a.p.created_at))
    .map(({ market, p }) => toPick(market, p));

  // No `all`: this page's "Every recorded pick" table also shows whether a pick
  // was counted or rebuilt, which `RecentPick` does not carry. It pages the rows
  // from /hub/track-record/picks when its fold opens, not from this payload.
  return { markets, lead: lead?.market ?? "", headline: "preKickoff", weeks, calibration, recent };
}
