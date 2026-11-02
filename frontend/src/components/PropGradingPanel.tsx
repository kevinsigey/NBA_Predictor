import { useEffect, useState } from "react";
import { api, type PropGrading, type PropGradingRow } from "../api/client";
import { STAT_LABELS, type StatKey } from "../lib/boxScore";
import { teamName } from "../lib/teams";
import { EmptyState, ErrorState, Skeleton } from "../predictor-ui";
import PlayerAvatar from "./PlayerAvatar";

/**
 * Prop grading: how close each counted player projection landed, for the
 * most recent completed game day.
 *
 * Three reads of the same graded rows, all computed by the backend from the
 * recorded predictions and outcomes (the earliest recorded projection per
 * player and stat -- the pick the track record counts):
 *
 * * **Most confident** -- the model's biggest moves from a player's own
 *   rolling average, each shown with the grade it earned after the fact.
 * * **Biggest surprises** -- the largest misses in both directions.
 * * **Every graded line**, game by game.
 *
 * The delta is actual minus projected, and the chip follows the PLAYER:
 * ▲ when he beat the projection, ▼ when he fell short. The grade judges the
 * model instead, in units of its own holdout error for the stat. A row that
 * cannot prove it was made before tip-off is graded and labelled, never
 * dropped and never passed off as pre-tip; the day's two counts sit side by
 * side in the header so the headline cannot quietly become the subset.
 */
export default function PropGradingPanel() {
  const [grading, setGrading] = useState<PropGrading | null>(null);
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    setGrading(null);
    api.getPropGrading().then(setGrading).catch(() => setError(true));
  }, [reloadKey]);

  if (error)
    return (
      <ErrorState
        message="We couldn't load the prop grading. Check your connection and try again."
        onRetry={() => setReloadKey((k) => k + 1)}
      />
    );
  if (grading === null) return <Skeleton label="Loading prop grading…" />;
  if (grading.latest_day === null)
    return <EmptyState message="No graded player projections yet. Lines are graded here once their games are final." />;

  const day = grading.latest_day;
  return (
    <div className="flex flex-col gap-8">
      <header>
        <h2 className="font-pr-display text-xl font-semibold uppercase tracking-wide text-pr-text">
          Prop grading {"\u00b7"} {day.date}
        </h2>
        <p className="mt-1 text-sm text-pr-text-dim">
          <span className="tabular-nums">{day.n_graded}</span> player lines graded {"\u00b7"}{" "}
          <span className="tabular-nums">{day.n_pre_tip}</span> made before tip-off
          {grading.served_stale_seconds ? (
            <>
              {" "}
              {"\u00b7"} showing the last good grading from{" "}
              <span className="tabular-nums">{Math.round(grading.served_stale_seconds)}s</span> ago because a refresh
              failed
            </>
          ) : null}
        </p>
        <p className="mt-1 max-w-3xl text-sm text-pr-text-faint">{grading.counting}</p>
      </header>

      <section aria-labelledby="prop-grading-confident">
        <h3
          id="prop-grading-confident"
          className="mb-1 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text"
        >
          Most confident
        </h3>
        <p className="mb-3 max-w-3xl text-sm text-pr-text-faint">
          {grading.confidence_basis} Ranked over{" "}
          <span className="tabular-nums">{grading.n_with_baseline}</span> of{" "}
          <span className="tabular-nums">{day.n_graded}</span> graded lines.
        </p>
        <GradedList rows={grading.most_confident} showBaseline />
      </section>

      <section aria-labelledby="prop-grading-surprises">
        <h3
          id="prop-grading-surprises"
          className="mb-1 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text"
        >
          Biggest surprises
        </h3>
        <p className="mb-3 max-w-3xl text-sm text-pr-text-faint">{grading.grade_basis}</p>
        <GradedList rows={grading.biggest_surprises} />
      </section>

      <section aria-labelledby="prop-grading-games">
        <h3
          id="prop-grading-games"
          className="mb-3 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text"
        >
          Every graded line {"\u00b7"} {day.date}
        </h3>
        <div className="flex flex-col gap-6">
          {day.games.map((game) => (
            <div key={game.game_id}>
              <h4 className="mb-2 text-sm font-semibold text-pr-text-dim">
                {teamName(game.away_team ?? "")} @ {teamName(game.home_team ?? "")}
              </h4>
              <GradedList rows={game.rows} />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function GradedList({ rows, showBaseline = false }: { rows: PropGradingRow[]; showBaseline?: boolean }) {
  if (rows.length === 0) return <p className="text-sm text-pr-text-faint">None.</p>;
  return (
    <ul className="grid gap-2 md:grid-cols-2">
      {rows.map((row) => (
        <li
          key={`${row.game_id}-${row.player_id}-${row.stat}`}
          className="flex items-center gap-3 rounded-md border border-pr-rule bg-pr-panel px-3 py-2"
        >
          <PlayerAvatar playerId={row.player_id} name={row.player_name ?? row.player_id} size={36} />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-semibold text-pr-text">
              {row.player_name ?? row.player_id}
              {!row.made_before_tip && (
                <span className="ml-2 text-xs font-normal uppercase tracking-wide text-pr-lean">
                  Made after tip-off
                </span>
              )}
            </span>
            <span className="block text-sm text-pr-text-dim">
              Proj {row.predicted.toFixed(1)} {"\u2192"} {row.actual} actual {"\u00b7"} {statLabel(row.stat)}
              {showBaseline && row.baseline !== null ? <> {"\u00b7"} his average {row.baseline.toFixed(1)}</> : null}
            </span>
          </span>
          <span className="flex shrink-0 flex-col items-end gap-1">
            <DeltaChip delta={row.delta} />
            {row.grade !== null && (
              <span className="rounded-sm border border-pr-rule px-1.5 py-0.5 text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
                {row.grade}
              </span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** The delta chip: direction is the player against the number. ▲ and the win
 *  colour when he beat the projection, ▼ and the loss colour when he fell
 *  short. An exact hit is neither, and says so. */
function DeltaChip({ delta }: { delta: number }) {
  if (delta === 0)
    return <span className="text-sm font-semibold tabular-nums text-pr-text-dim">0.0 vs proj</span>;
  const beat = delta > 0;
  return (
    <span className={`text-sm font-semibold tabular-nums ${beat ? "text-pr-win" : "text-pr-loss"}`}>
      {beat ? "\u25b2" : "\u25bc"} {beat ? "+" : "\u2212"}
      {Math.abs(delta).toFixed(1)} vs proj
    </span>
  );
}

function statLabel(stat: string): string {
  return STAT_LABELS[stat as StatKey] ?? stat;
}
