// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The view model every sport maps its own API onto.
 *
 * The kit renders this and nothing else: no site shape leaks into a component,
 * and a backend contract never changes to fit a page. Each site writes one
 * adapter (`toRecordModel(...)`) and the surfaces below are identical across
 * all five sports.
 *
 * The one rule the model exists to carry (PRODUCT.md, revised 2026-10-01):
 * **every recorded pick counts, once per game and market.** So `counted` is the
 * whole record and `preKickoff` is the subset made before the start, shown
 * beside it. NBA is the deliberate exception and says so with
 * `headline: "preKickoff"` — its big number is the pre-tip subset.
 */

/** A tally: how many right out of how many graded. */
export type Tally = { correct: number; n: number };

/** One market's record. `counted` is required; `preKickoff` is the disclosure beside it. */
export type MarketTally = {
  key: string;
  label: string;
  counted: Tally;
  /** The subset made before the start. Absent when the feed has none yet. */
  preKickoff?: Tally;
};

/** Which tally is the big headline number. */
export type HeadlineMode = "preKickoff" | "counted";

/** One week of the strip. `tracked: false` is history before tracking began. */
export type WeekPoint = { weekStart: string; n: number; correct: number; tracked: boolean };

/** One stated-confidence bucket for the calibration chart. */
export type CalibrationBucket = { bucket: string; n: number; correct: number };

/** One market's calibration curve. */
export type CalibrationSeries = { key: string; label: string; buckets: CalibrationBucket[] };

/**
 * One pick, in reader's words: names, never ids.
 *
 * `madeBeforeKickoff: false` is the disclosure the whole product is built
 * around — a rebuilt pick is labelled, never hidden, and never presented as a
 * pre-game prediction.
 */
export type RecentPick = {
  /** Stable per pick; the React key. Not shown. */
  id: string;
  /** The market, already through `vocabulary.marketLabel`. */
  label: string;
  pick: string;
  actual: string;
  hit: boolean | null;
  /** When it was made, in words (the adapter formats the timestamp). */
  when: string;
  madeBeforeKickoff: boolean;
};

/**
 * The whole record page, as data.
 *
 * `error` is per-part and isolated: a broken comparison block must not take the
 * record above it down. It is carried here so an adapter reports it in one
 * place; the kit renders failures through the shared `States` components.
 *
 * NOTE — deviation from the plan's sketch, flagged in the PR: `calibration` is a
 * list of series, not a flat bucket list. NBA's chart has a per-market toggle
 * over several curves, and Phase 2 requires NBA to look exactly as it does
 * today, so the flat shape could not carry it.
 */
export interface RecordModel {
  markets: MarketTally[];
  /** Market key for the headline. */
  lead: string;
  /** Which tally is the big number. Defaults to `counted` at the adapter. */
  headline: HeadlineMode;
  weeks?: WeekPoint[];
  calibration?: CalibrationSeries[];
  recent?: RecentPick[];
  /** Every recorded pick, for the collapsed table. Never the first thing shown. */
  all?: RecentPick[];
  error?: Partial<Record<"market" | "props" | "calibration", boolean>>;
}
