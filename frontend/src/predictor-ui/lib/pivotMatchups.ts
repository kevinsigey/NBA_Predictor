// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { MatchupRow } from "../components/MatchupBrief";

const isText = (v: unknown): v is string => typeof v === "string" && v.trim() !== "";
const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

const drawable = (m: MatchupRow) =>
  !!m &&
  isText(m.id) &&
  isText(m.attacker) &&
  isText(m.defender) &&
  isText(m.stat) &&
  isText(m.foil) &&
  isInt(m.attacker_rank) &&
  isInt(m.defender_rank) &&
  isInt(m.n_teams) &&
  m.n_teams >= 2 &&
  m.attacker_rank >= 1 &&
  m.attacker_rank <= m.n_teams &&
  m.defender_rank >= 1 &&
  m.defender_rank <= m.n_teams;

function stripSuffix(id: string): string {
  return id.replace(/:home$|:away$/, "");
}

function capitalizeFirst(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Pivots matchup rows into a table structure with home on the left, away on the right.
 * Returns null when nothing drawable.
 *
 * For each duel type (base key):
 * - If both :home and :away rows exist: produces TWO rows (offence then defence)
 *   - Offence: homeRank = :home.attacker_rank, awayRank = :away.attacker_rank, label = :home.stat
 *   - Defence: homeRank = :away.defender_rank, awayRank = :home.defender_rank, label = :home.foil
 * - If only one side exists: produces two rows with null for missing side
 * - Unsuffixed rows (like `form`): produces ONE row with both ranks from that row
 * - Row order: first appearance of each base key, offence before defence, unsuffixed last
 */
export function pivotMatchups(rows: MatchupRow[]): {
  home: string;
  away: string;
  rows: { key: string; label: string; homeRank: number | null; awayRank: number | null; n: number }[];
} | null {
  const valid = (Array.isArray(rows) ? rows : []).filter(drawable);
  if (valid.length === 0) return null;

  // Determine home/away from :home rows (attacker is home team in :home rows)
  const homeRow = valid.find(r => r.id.endsWith(":home"));
  const awayRow = valid.find(r => r.id.endsWith(":away"));

  let home: string;
  let away: string;

  if (homeRow) {
    home = homeRow.attacker;
    away = homeRow.defender;
  } else if (awayRow) {
    // Fallback: if only :away rows, attacker is away team
    away = awayRow.attacker;
    home = awayRow.defender;
  } else {
    // No suffix rows - use first row
    home = valid[0].attacker;
    away = valid[0].defender;
  }

  // Group by base key, track what we need for each
  interface KeyData {
    homeRow: MatchupRow | null;
    awayRow: MatchupRow | null;
    unsuffixedRows: MatchupRow[];
    firstIndex: number;
    n: number;
  }

  const byKey = new Map<string, KeyData>();

  for (let i = 0; i < valid.length; i++) {
    const row = valid[i];
    const key = stripSuffix(row.id);
    const isHome = row.id.endsWith(":home");
    const isAway = row.id.endsWith(":away");
    const existing = byKey.get(key) ?? {
      homeRow: null,
      awayRow: null,
      unsuffixedRows: [],
      firstIndex: i,
      n: row.n_teams,
    };

    if (isHome) {
      existing.homeRow = row;
    } else if (isAway) {
      existing.awayRow = row;
    } else {
      existing.unsuffixedRows.push(row);
    }

    byKey.set(key, existing);
  }

  // Sort by first appearance for consistent output matching live data feed order
  const sortedKeys = [...byKey.entries()].sort((a, b) => a[1].firstIndex - b[1].firstIndex);

  const outRows: { key: string; label: string; homeRank: number | null; awayRank: number | null; n: number }[] = [];
  const unsuffixedOut: { key: string; label: string; homeRank: number | null; awayRank: number | null; n: number }[] = [];

  for (const [key, data] of sortedKeys) {
    const { homeRow, awayRow, unsuffixedRows, n } = data;

    if (homeRow && awayRow) {
      // Both sides present: produce offence then defence
      // Offence: home's stat rank vs away's stat rank
      outRows.push({
        key: `${key}_offence`,
        label: capitalizeFirst(homeRow.stat),
        homeRank: homeRow.attacker_rank,
        awayRank: awayRow.attacker_rank,
        n,
      });
      // Defence: home's foil rank vs away's foil rank
      outRows.push({
        key: `${key}_defence`,
        label: capitalizeFirst(homeRow.foil),
        homeRank: awayRow.defender_rank,
        awayRank: homeRow.defender_rank,
        n,
      });
    } else if (homeRow && !awayRow) {
      // Only :home row: offence (home has stat, away has foil)
      outRows.push({
        key: `${key}_offence`,
        label: capitalizeFirst(homeRow.stat),
        homeRank: homeRow.attacker_rank,
        awayRank: null,
        n,
      });
      outRows.push({
        key: `${key}_defence`,
        label: capitalizeFirst(homeRow.foil),
        homeRank: null,
        awayRank: homeRow.defender_rank,
        n,
      });
    } else if (!homeRow && awayRow) {
      // Only :away row: offence (away has stat, home has foil)
      // A lone :away row carries ONLY away offence (attacker_rank) and home defence (defender_rank); the other
      // two cells have no source and stay empty rather than borrowing a rank for a different metric.
      outRows.push({
        key: `${key}_offence`,
        label: capitalizeFirst(awayRow.stat),
        homeRank: null,
        awayRank: awayRow.attacker_rank,
        n,
      });
      outRows.push({
        key: `${key}_defence`,
        label: capitalizeFirst(awayRow.foil),
        homeRank: awayRow.defender_rank,
        awayRank: null,
        n,
      });
    }

    // Unsuffixed rows (like form): collect to add at the end
    for (const r of unsuffixedRows) {
      unsuffixedOut.push({
        key: r.id,
        label: capitalizeFirst(r.stat),
        homeRank: r.attacker === home ? r.attacker_rank : (r.defender === home ? r.defender_rank : null),
        awayRank: r.attacker === away ? r.attacker_rank : (r.defender === away ? r.defender_rank : null),
        n: r.n_teams,
      });
    }
  }

  // Append unsuffixed rows at the end
  outRows.push(...unsuffixedOut);

  return {
    home,
    away,
    rows: outRows,
  };
}