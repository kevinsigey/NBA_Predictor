// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState, type ReactNode } from "react";

/**
 * The phone-readable stat table: players down the side, stats across the top,
 * the stats slide sideways inside their own scroller and the first column
 * (the full player name) never moves. Presentational: it renders rows the
 * caller has built and decides nothing about order or content.
 */

export type StickyColumn = { key: string; label: string };

export type StickyRow = {
  key: string;
  /** First, pinned cell: the full name, plus whatever marker the data has. */
  header: ReactNode;
  /** Aligned to `columns`. null / undefined / NaN / "" render an em dash. */
  cells: ReactNode[];
  /** Screen-reader text per cell, aligned to `cells`. */
  cellLabels?: (string | undefined)[];
  /** Index of the one cell that shouts. */
  emphasisIndex?: number;
  /** A total / team row: bolder, ruled off, and (in `footer`) last. */
  total?: boolean;
  testId?: string;
  attrs?: Record<string, string>;
  /** Raw values to sort by, aligned to [header, ...columns]. A row without them keeps its order. */
  sortValues?: (number | string | null)[];
};

export type StickySection = { label?: string; rows: StickyRow[] };

export interface StickyStatTableProps {
  columns: StickyColumn[];
  sections: StickySection[];
  /** Rows after every section (team totals), rendered in <tfoot>. */
  footer?: StickyRow[];
  /** Visually hidden table caption. */
  caption: string;
  firstColumnLabel?: string;
  ariaLabel?: string;
  /** One-line message shown instead of an empty table. */
  empty?: string;
  testId?: string;
  tableTestId?: string;
}

type Sort = { index: number; dir: "asc" | "desc" } | null;

/** Missing values sort last whichever way the column is sorted. */
function compareValues(a: number | string | null | undefined, b: number | string | null | undefined, dir: "asc" | "desc"): number {
  const missing = (v: unknown) => v === null || v === undefined || v === "" || (typeof v === "number" && !Number.isFinite(v));
  if (missing(a) || missing(b)) return Number(missing(a)) - Number(missing(b));
  const order = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
  return dir === "asc" ? order : -order;
}

const EM_DASH = "—";
/** Pinned-column width: full names wrap to two lines inside it rather than clip. */
const NAME_WIDTH = "10rem";

function show(cell: ReactNode): ReactNode {
  if (cell === null || cell === undefined || cell === "") return EM_DASH;
  if (typeof cell === "number") {
    if (!Number.isFinite(cell)) return EM_DASH;
    return Number.isInteger(cell) ? String(cell) : cell.toFixed(1);
  }
  return cell;
}

// Opaque on purpose: a translucent pinned cell lets scrolled stats show through.
const bgFor = (i: number) => (i % 2 === 1 ? "bg-pr-panel-2" : "bg-pr-panel");

export function StickyStatTable({
  columns,
  sections,
  footer = [],
  caption,
  firstColumnLabel = "Player",
  ariaLabel = "Box score, scroll sideways for more stats",
  empty,
  testId,
  tableTestId,
}: StickyStatTableProps) {
  const [pinned, setPinned] = useState(false);
  const [sort, setSort] = useState<Sort>(null);
  const rowCount = sections.reduce((n, s) => n + s.rows.length, 0) + footer.length;

  if (rowCount === 0) {
    return empty ? (
      <p className="py-3 text-sm text-pr-text-dim" data-testid={testId}>
        {empty}
      </p>
    ) : null;
  }

  const pin = {
    boxShadow: pinned ? "4px 0 6px -2px rgba(0,0,0,0.5)" : undefined,
    borderRight: "1px solid var(--color-pr-rule)",
  };
  const nameStyle = { width: NAME_WIDTH, minWidth: NAME_WIDTH, maxWidth: NAME_WIDTH, ...pin };
  const nameCls = "sticky left-0 z-10 whitespace-normal break-words px-3 py-1.5 text-left align-middle text-[0.8125rem] font-normal leading-tight text-pr-text";

  // Stats start high-to-low, the name column A-Z; a third click puts the caller's order back.
  const cycle = (index: number) =>
    setSort((cur) => {
      const first = index === 0 ? "asc" : "desc";
      if (!cur || cur.index !== index) return { index, dir: first };
      return cur.dir === first ? { index, dir: first === "asc" ? "desc" : "asc" } : null;
    });
  const sortable = sections.some((sec) => sec.rows.some((r) => r.sortValues));
  const ordered = (rows: StickyRow[]) =>
    sort ? [...rows].sort((a, b) => compareValues(a.sortValues?.[sort.index], b.sortValues?.[sort.index], sort.dir)) : rows;
  const headLabel = (index: number, label: string) => {
    if (!sortable) return label;
    const active = sort?.index === index;
    return (
      <button type="button" onClick={() => cycle(index)} className="inline-flex items-center gap-1 uppercase tracking-[0.04em] focus-visible:outline focus-visible:outline-2 focus-visible:outline-pr-accent">
        {label}
        <span aria-hidden="true">{active ? (sort!.dir === "asc" ? "▲" : "▼") : ""}</span>
      </button>
    );
  };
  const ariaSort = (index: number) => (sortable ? (sort?.index === index ? (sort.dir === "asc" ? "ascending" : "descending") : "none") : undefined);

  const renderRow = (row: StickyRow, i: number) => {
    const bg = row.total ? "bg-pr-panel-2" : bgFor(i);
    return (
      <tr
        key={row.key}
        data-testid={row.testId}
        {...row.attrs}
        className={`border-t border-pr-rule ${row.total ? "font-semibold" : ""} ${bg}`}
      >
        <th scope="row" data-pinned={String(pinned)} className={`${nameCls} ${bg} ${row.total ? "font-semibold" : ""}`} style={nameStyle}>
          {row.header}
        </th>
        {columns.map((column, c) => {
          const label = row.cellLabels?.[c];
          return (
            <td
              key={column.key}
              {...(label ? { "aria-label": label } : {})}
              className={`${bg} whitespace-nowrap px-3 py-1.5 text-right align-middle text-[0.8125rem] tabular-nums ${row.emphasisIndex === c || row.total ? "font-semibold" : ""}`}
            >
              {show(row.cells[c])}
            </td>
          );
        })}
      </tr>
    );
  };

  return (
    <div data-testid={testId}>
      <div
        role="region"
        tabIndex={0}
        aria-label={ariaLabel}
        onScroll={(e) => setPinned(e.currentTarget.scrollLeft > 0)}
        className="max-h-[min(75vh,44rem)] overflow-auto rounded-pr border border-pr-rule [-webkit-overflow-scrolling:touch] focus-visible:outline focus-visible:outline-2 focus-visible:outline-pr-accent"
      >
        <table className="w-max min-w-full border-collapse" data-testid={tableTestId}>
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr>
              <th
                scope="col"
                aria-sort={ariaSort(0)}
                data-pinned={String(pinned)}
                className="sticky left-0 top-0 z-30 bg-pr-panel px-3 py-1.5 text-left text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-pr-text-dim"
                style={nameStyle}
              >
                {headLabel(0, firstColumnLabel)}
              </th>
              {columns.map((column, c) => (
                <th
                  key={column.key}
                  scope="col"
                  aria-sort={ariaSort(c + 1)}
                  className="sticky top-0 z-20 whitespace-nowrap bg-pr-panel px-3 py-1.5 text-right text-[0.6875rem] font-semibold uppercase tracking-[0.04em] text-pr-text-dim"
                >
                  {headLabel(c + 1, column.label)}
                </th>
              ))}
            </tr>
          </thead>
          {sections.map((section, s) => (
            <tbody key={section.label ?? s}>
              {section.label && (
                <tr className="border-t border-pr-rule bg-pr-panel-2">
                  <th
                    scope="rowgroup"
                    data-pinned={String(pinned)}
                    className="sticky left-0 z-10 bg-pr-panel-2 px-3 py-1 text-left font-pr-display text-xs font-bold uppercase tracking-wider text-pr-text"
                    style={nameStyle}
                  >
                    {section.label}
                  </th>
                  <td colSpan={columns.length} className="bg-pr-panel-2" />
                </tr>
              )}
              {ordered(section.rows).map(renderRow)}
            </tbody>
          ))}
          {footer.length > 0 && <tfoot>{footer.map(renderRow)}</tfoot>}
        </table>
      </div>
    </div>
  );
}
