// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * A presentational box score. It renders a structure the caller has already
 * built; it never fetches, never reorders, and does not know what a touchdown
 * is. If it decided ordering, the caller could not put a team's players in
 * depth-chart order, which is the one ordering that matters.
 */

export type BoxScoreRow = {
  key: string; // stable React key
  name: string;
  position: string; // display group label
  team: string; // abbreviation
  order: number; // caller-assigned; lower sorts first
  isStarter: boolean | null; // null = this sport has no real starter data
  values: (number | null)[]; // aligned to `columns`; null renders an em-dash
  emphasisIndex?: number; // which of `values` is the one number that shouts
};

export type BoxScoreColumn = {
  key: string;
  label: string;
  align?: "left" | "right";
  /** Screen-reader text for a row's cell, since the grid itself is terse. */
  describe?: (value: number | null, row: BoxScoreRow) => string;
};

export type BoxScoreGroup = {
  position: string;
  /** Already grouped, already ordered starters-then-bench. Rendered as given. */
  rows: BoxScoreRow[];
  /** Per-team subtotal for this position, or null when not wanted. */
  subtotals?: BoxScoreTotal[] | null;
};

export type BoxScoreTotal = {
  label: string; // e.g. "MIA total"
  values: (number | null)[]; // aligned to the same `columns`
  strong?: boolean; // the projected score: renders at display scale
};

export interface BoxScoreProps {
  columns: BoxScoreColumn[];
  groups: BoxScoreGroup[];
  /** Heading for the block, e.g. "Predicted box score". */
  title?: string;
  /** Which states exist across the rows, which decides the note. */
  starterLabel?: string;
  benchLabel?: string;
}

import { StickyStatTable, type StickyRow } from "./StickyStatTable";

/**
 * The three starter states, decided across every row rather than per row.
 *
 * A single null row in an otherwise-known game must not flip the whole table
 * to "projected", or the header would lie about the eleven rows that do have a
 * real starter. The header reports what the DATA knows, and a per-row tag
 * reports the row that does not.
 */
function starterState(groups: BoxScoreGroup[]): "known" | "projected" | "none" {
  let seenNull = false;
  let seenStarter = false;
  for (const group of groups) {
    for (const row of group.rows) {
      if (row.isStarter === null) seenNull = true;
      else if (row.isStarter) seenStarter = true;
    }
  }
  if (seenStarter) return "known";
  if (seenNull) return "projected";
  return "none";
}

function toRow(row: BoxScoreRow, columns: BoxScoreColumn[]): StickyRow {
  return {
    key: row.key,
    header: (
      <>
        <span>{row.name}</span>
        {row.team && <span className="ml-1.5 text-sm text-pr-text-dim">{row.team}</span>}
        {row.isStarter === null && <span className="sr-only"> — projected order, no depth-chart data</span>}
      </>
    ),
    cells: columns.map((_, i) => row.values[i] ?? null),
    // The describe hook exists because the grid is terse: "17.9" alone is not
    // a sentence. Without it a screen reader hears a bare number.
    cellLabels: columns.map((column, i) => column.describe?.(row.values[i] ?? null, row)),
    emphasisIndex: row.emphasisIndex,
    testId: "box-score-row",
    attrs: { "data-starter": row.isStarter === null ? "projected" : row.isStarter ? "starter" : "bench" },
  };
}

export function BoxScore({ columns, groups, title, starterLabel = "Starters", benchLabel = "Bench" }: BoxScoreProps) {
  const state = starterState(groups);
  const rowTotal = groups.reduce((n, g) => n + g.rows.length, 0);

  if (rowTotal === 0) return null;

  return (
    <section className="mt-5" data-testid="box-score">
      {title && (
        <h3 className="mb-1 text-sm text-pr-text-dim">
          {title}
          {state === "projected" && (
            <span className="ml-2 text-sm" data-testid="box-score-projected-note">
              Projected order — no depth-chart feed for this sport
            </span>
          )}
        </h3>
      )}

      <StickyStatTable
        tableTestId="box-score-table"
        caption={`${title ?? "Predicted box score"}${state === "projected" ? ", in projected order: this sport has no depth-chart feed" : ""}`}
        columns={columns}
        sections={groups.map((group) => ({
          label: group.position,
          rows: [
            ...group.rows.map((row) => toRow(row, columns)),
            ...(group.subtotals ?? []).map((total) => ({
              key: total.label,
              header: total.label,
              cells: columns.map((_, i) => total.values[i] ?? null),
              total: true,
              testId: "box-score-subtotal",
            })),
          ],
        }))}
      />

      {state === "known" && (
        <p className="mt-1 text-sm text-pr-text-dim">
          {starterLabel} listed before {benchLabel.toLowerCase()}.
        </p>
      )}
    </section>
  );
}
