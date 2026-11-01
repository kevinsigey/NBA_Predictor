// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { kickoff } from "../fmt";
import type { WeekPoint } from "./types";

/**
 * One cell per week: colour is the hit rate, height is how many picks stood
 * behind it. Ported from NBA. Only weeks from the first tracked one on: the
 * empty history before tracking began is not a record.
 */
export function WeekStrip({ weeks }: { weeks: WeekPoint[] }) {
  const first = weeks.findIndex((w) => w.tracked && w.n > 0);
  const shown = first === -1 ? [] : weeks.slice(first);
  const maxN = Math.max(1, ...shown.map((w) => w.n));
  if (shown.length === 0) {
    return <p className="text-sm text-pr-text-dim">No graded week yet.</p>;
  }
  return (
    <div>
      <ol className="flex h-24 items-end gap-1" aria-label="Hit rate by week">
        {shown.map((w) => {
          const graded = w.tracked && w.n > 0;
          const rate = graded ? w.correct / w.n : null;
          const h = graded ? 18 + (w.n / maxN) * 82 : 12;
          const tone = !graded ? "bg-pr-panel-2" : rate! >= 0.5 ? "bg-pr-win" : "bg-pr-loss";
          return (
            <li
              key={w.weekStart}
              data-testid="week-cell"
              title={graded ? `${kickoff(w.weekStart)}: ${w.correct} of ${w.n} (${Math.round(rate! * 100)}%)` : `${kickoff(w.weekStart)}: nothing graded`}
              className={`min-w-1.5 max-w-9 flex-1 rounded-t-pr ${tone} ${graded ? "" : "opacity-60"}`}
              style={{ height: `${h}%` }}
            >
              <span className="sr-only">
                {graded ? `${kickoff(w.weekStart)}: ${w.correct} of ${w.n}` : `${kickoff(w.weekStart)}: nothing graded`}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-pr-text-faint">
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-pr bg-pr-win" />50% or better</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-pr bg-pr-loss" />under 50%</span>
        <span>Taller = more picks that week</span>
      </p>
    </div>
  );
}
