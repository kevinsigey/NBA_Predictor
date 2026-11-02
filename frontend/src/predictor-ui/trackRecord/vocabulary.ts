// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * One vocabulary for five sports.
 *
 * The record pages were rebuilt in four places and named the same facts four
 * ways. These constants are the single source of the words: a market's label,
 * the verdict on a pick, and the moment a pick has to beat. Components import
 * from here rather than typing the string, and `vocabulary.test.ts` fails the
 * build if a kit file hard-codes a superseded word.
 */
import type { Sport } from "../components/AppFrame";
import type { Moment } from "../components/StatusBadge";

/** The verdict on a graded pick. Right/Wrong, never Called it/Missed. */
export const RIGHT = "Right";
export const WRONG = "Wrong";
/** A pick that landed on the line: not right, not wrong. Shown, never scored. */
export const PUSH = "Push";

/**
 * Market keys to the words a reader knows. Keys are the wire values (`h2h`,
 * `player_props`); the values are what appears on the page. A key not listed
 * falls back to itself, which is a sport's job to fix in its adapter rather
 * than a raw code the kit would happily print.
 */
export const MARKET_LABELS: Record<string, string> = {
  game_outcome: "Winner pick",
  moneyline: "Moneyline",
  h2h: "Moneyline",
  spread: "Spread",
  total: "Total",
  totals: "Total",
  props: "Props",
  player_props: "Props",
};

/** The label for a market key, falling back to the key itself. */
export const marketLabel = (key: string): string => MARKET_LABELS[key] ?? key;

/**
 * Reading order: the lead market first (the caller moves it), then Moneyline,
 * Spread, Total, Props, then anything else in the order it arrived. Ordered by
 * key so two pages cannot disagree about which market reads first. The keys are
 * the wire values, written unquoted so they read as a mapping rather than as a
 * string this kit would print.
 */
const ORDER: Record<string, number> = {
  game_outcome: 0,
  moneyline: 1,
  h2h: 2,
  spread: 3,
  total: 4,
  totals: 5,
  props: 6,
  player_props: 7,
};

/** A market's position in the fixed reading order. Unknown keys sort last. */
export const marketRank = (key: string): number => ORDER[key] ?? Object.keys(ORDER).length;

/** The word each sport uses for the moment a pick must beat. */
export const SPORT_MOMENT: Record<Sport, Moment> = {
  nba: "tip-off",
  f1: "the session",
  pl: "kickoff",
  nfl: "kickoff",
  cfb: "kickoff",
  hub: "kickoff",
};

/** The moment for a sport. */
export const momentFor = (sport: Sport): Moment => SPORT_MOMENT[sport];
