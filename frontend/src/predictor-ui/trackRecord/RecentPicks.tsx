// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { Moment } from "../components/StatusBadge";
import { PUSH, RIGHT, WRONG } from "./vocabulary";
import type { RecentPick } from "./types";

const verdict = (hit: boolean | null) => {
  if (hit === null) return <span className="text-pr-text-faint">{PUSH}</span>;
  return hit ? (
    <span className="font-semibold text-pr-win" aria-label={RIGHT}>{RIGHT}</span>
  ) : (
    <span className="font-semibold text-pr-loss" aria-label={WRONG}>{WRONG}</span>
  );
};

/**
 * The latest picks, newest first: what was called, what happened.
 *
 * The adapter supplies the rows already filtered to the set the headline counts
 * and already sorted — sorting a formatted date string here would sort "Mon 5
 * Oct" and "Sun 4 Oct" alphabetically. A pick made after the start is labelled
 * with when it was made, so a rebuilt pick is disclosed rather than hidden.
 */
export function RecentPicks({
  picks,
  limit = 12,
  moment = "kickoff",
}: {
  picks: RecentPick[];
  limit?: number;
  /** The sport's word for the start, shown on any pick not made before it. */
  moment?: Moment;
}) {
  const rows = picks.slice(0, limit);
  if (rows.length === 0) return <p className="text-sm text-pr-text-dim">No counted pick has been graded yet.</p>;
  return (
    <ul className="divide-y divide-pr-rule rounded-pr border border-pr-rule bg-pr-panel text-sm">
      {rows.map((p) => (
        <li
          key={p.id}
          data-testid="recent-pick"
          className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-x-3 px-4 py-2"
        >
          <span className="tabular-nums text-pr-text-faint">{p.when}</span>
          <span className="min-w-0 truncate">
            <span className="font-semibold text-pr-text">{p.pick}</span>
            <span className="text-pr-text-faint">
              {" · "}
              {p.label}
              {" · was "}
              {p.actual}
            </span>
            {!p.madeBeforeKickoff && <span className="text-pr-text-faint"> · rebuilt after {moment}</span>}
          </span>
          {verdict(p.hit)}
        </li>
      ))}
    </ul>
  );
}
