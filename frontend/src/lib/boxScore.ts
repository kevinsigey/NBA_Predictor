import type { PlayerProp, PlayerHubRow } from "../api/client";

/**
 * The per-game player feed is one row per player PER STAT, so a box score is a
 * pivot: 88 rows across 22 players becomes 22 rows with four stat columns.
 * Measured against the live API, not assumed.
 */
export const STAT_KEYS = ["points", "rebounds", "assists", "threes"] as const;
export type StatKey = (typeof STAT_KEYS)[number];

export const STAT_LABELS: Record<StatKey, string> = {
  points: "Pts",
  rebounds: "Reb",
  assists: "Ast",
  threes: "3PM",
};

/**
 * Position order for reading order down a side of the table.
 *
 * The hub feed is coarse for almost everyone -- 308 "G", 240 "F", 93 "C" out of
 * 648 players, with a handful of granular PG/SG/SF/PF rows -- so both granular
 * and coarse labels have to sort, and anything unrecognised goes last rather
 * than first. A blank position is real (one player has one), not an error.
 */
const POSITION_ORDER = ["PG", "SG", "G", "SF", "PF", "F", "C"];

export interface BoxScoreRow {
  player_id: string;
  player_name: string;
  /** From the hub feed. Empty when the player is not in it. */
  team: string;
  position: string;
  predicted: Partial<Record<StatKey, number>>;
  /** Absent until the game is played; never defaulted to 0. */
  actual: Partial<Record<StatKey, number>>;
  /** The model's 80% conformal range per stat, where the API served one.
   *  Absent per stat when no calibration tier could serve the day. */
  intervals: Partial<Record<StatKey, { low: number; high: number }>>;
  rebuilt: boolean;
}

function isStatKey(stat: string): stat is StatKey {
  return (STAT_KEYS as readonly string[]).includes(stat);
}

/**
 * Group one-row-per-stat props into one row per player.
 *
 * `actual` is only set where the API actually reported a number. Treating a
 * missing actual as 0 would make a pre-game table claim every player went for
 * zero points, so an absent value stays absent and renders as a dash.
 */
export function pivotProps(props: PlayerProp[]): BoxScoreRow[] {
  const rows = new Map<string, BoxScoreRow>();

  for (const prop of props) {
    if (!isStatKey(prop.stat)) continue;
    let row = rows.get(prop.player_id);
    if (!row) {
      row = {
        player_id: prop.player_id,
        player_name: prop.player_name,
        team: prop.team ?? "",
        position: "",
        predicted: {},
        actual: {},
        intervals: {},
        rebuilt: false,
      };
      rows.set(prop.player_id, row);
    }
    // The API already returns the latest prop made before tip-off for each
    // (player, stat) pair, so a repeat is a duplicate, not an update.
    if (row.predicted[prop.stat] === undefined) {
      row.predicted[prop.stat] = prop.predicted_value;
    }
    if (
      row.intervals[prop.stat] === undefined &&
      prop.interval_low !== null && prop.interval_low !== undefined &&
      prop.interval_high !== null && prop.interval_high !== undefined
    ) {
      row.intervals[prop.stat] = { low: prop.interval_low, high: prop.interval_high };
    }
    if (prop.actual_value !== null && prop.actual_value !== undefined) {
      row.actual[prop.stat] = prop.actual_value;
    }
    if (prop.rebuilt) row.rebuilt = true;
  }

  return [...rows.values()];
}

/** Attach each player's position (and team, unless the projection already carries his current one) from the season hub feed. */
export function attachHubRows(rows: BoxScoreRow[], hub: PlayerHubRow[]): BoxScoreRow[] {
  const byId = new Map(hub.map((p) => [p.player_id, p]));
  return rows.map((row) => {
    const match = byId.get(row.player_id);
    return match ? { ...row, team: row.team || match.team, position: match.position } : row;
  });
}

export interface BoxScoreSides {
  away: BoxScoreRow[];
  home: BoxScoreRow[];
  /** Players whose team the hub feed does not know. */
  unattributed: BoxScoreRow[];
}

/**
 * Split by the player's actual team.
 *
 * The per-game feed carries no team, so the split is a join against the hub
 * feed. Anyone the join cannot place is returned rather than dropped, and the
 * caller says so on screen -- a player silently moved to the wrong side of the
 * table would be a fabricated box score.
 *
 * Known limit, and it is a freshness limit rather than a guess: the hub feed's
 * `team` is the team in a player's most recent box score inside its window, not
 * their team as of this game's date. For a game earlier in that window where a
 * player was traded, they file under their later team. Fixing it properly means
 * sourcing team and position from the per-game box scores the API already
 * fetches, so the join disappears; until then, do not describe the split as
 * exact for historical games.
 */
export function splitByTeam(rows: BoxScoreRow[], homeTeam: string, awayTeam: string): BoxScoreSides {
  const away: BoxScoreRow[] = [];
  const home: BoxScoreRow[] = [];
  const unattributed: BoxScoreRow[] = [];

  for (const row of rows) {
    if (row.team === awayTeam) away.push(row);
    else if (row.team === homeTeam) home.push(row);
    else unattributed.push(row);
  }

  return { away: orderPlayers(away), home: orderPlayers(home), unattributed };
}

function positionRank(position: string): number {
  const i = POSITION_ORDER.indexOf(position);
  return i === -1 ? POSITION_ORDER.length : i;
}

/** Guards, then the biggest projected role, then the name, so it is stable. */
export function orderPlayers(rows: BoxScoreRow[]): BoxScoreRow[] {
  return [...rows].sort(
    (a, b) =>
      positionRank(a.position) - positionRank(b.position) ||
      (b.predicted.points ?? 0) - (a.predicted.points ?? 0) ||
      a.player_name.localeCompare(b.player_name),
  );
}

export function sumStat(rows: BoxScoreRow[], stat: StatKey, source: "predicted" | "actual"): number {
  return rows.reduce((total, row) => total + (row[source][stat] ?? 0), 0);
}

export function hasActuals(rows: BoxScoreRow[]): boolean {
  return rows.some((row) => STAT_KEYS.some((stat) => row.actual[stat] !== undefined));
}
