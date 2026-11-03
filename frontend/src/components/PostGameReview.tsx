import type { PropGameGrading, PropGradingRow } from "../api/client";
import { STAT_LABELS, type StatKey } from "../lib/boxScore";
import PlayerAvatar from "./PlayerAvatar";

/**
 * The post-game review: how the counted player projections did in THIS game.
 *
 * The rows are the backend's grading of the counted picks -- the earliest
 * recorded projection per player and stat, the same picks the track record
 * counts -- served for this game alone (`/hub/prop-grading/games/{id}`).
 * Every counted row is here: a line is judged against the props model's own
 * holdout error for its stat (the grade chip), the delta chip follows the
 * PLAYER (▲ when he beat the projection), and a row that cannot prove it
 * was made before tip-off is labelled, never dropped and never passed off
 * as pre-tip. A game with no counted props gets no section at all -- the
 * caller renders nothing for it, and so does this component.
 *
 * The game's biggest surprises lead: the rows the model missed by more
 * than its own error, biggest miss first. The full list follows, grouped
 * by player, so the review alone answers how the props did in this match.
 */

/** How many of the game's surprises lead the review. */
const SURPRISES_SHOWN = 4;

export function PostGameReview({ grading }: { grading: PropGameGrading | null }) {
  if (!grading || grading.rows.length === 0) return null;

  const banded = grading.rows.filter((r) => r.grade !== null);
  const within = banded.filter((r) => r.grade === "On the number" || r.grade === "Close").length;
  const surprises = grading.biggest_surprises
    .filter((r) => r.grade === "Off" || r.grade === "Way off")
    .slice(0, SURPRISES_SHOWN);

  // The backend serves the game's rows ordered by player, then by stat in
  // box-score order; group them so a player reads as one block of lines.
  const players: { player_id: string; player_name: string | null; rows: PropGradingRow[] }[] = [];
  for (const row of grading.rows) {
    const last = players[players.length - 1];
    if (last && last.player_id === row.player_id) last.rows.push(row);
    else players.push({ player_id: row.player_id, player_name: row.player_name, rows: [row] });
  }

  return (
    <section aria-labelledby="player-review-heading" data-testid="post-game-review">
      <h3
        id="player-review-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Player review
      </h3>
      <p className="text-sm text-pr-text-dim">
        <span className="tabular-nums">{grading.n_graded}</span> player lines graded {"\u00b7"}{" "}
        <span className="tabular-nums">{grading.n_pre_tip}</span> made before tip-off
      </p>

      {banded.length > 0 && (
        <div data-testid="review-on-the-number" className="mt-3">
          <p className="text-sm text-pr-text">
            <span className="font-pr-display text-lg font-semibold tabular-nums">{within}</span> of{" "}
            <span className="tabular-nums">{banded.length}</span> player lines within range
          </p>
          <p className="mt-1 text-sm text-pr-text-faint">
            A line is within range when the miss is no bigger than the model's own error for that stat.
          </p>
        </div>
      )}

      {surprises.length > 0 && (
        <div className="mt-4" data-testid="review-surprises">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
            Biggest surprises in this game
          </h4>
          <ul className="grid gap-2 sm:grid-cols-2">
            {surprises.map((row) => (
              <GradedRowCard key={`${row.player_id}-${row.stat}`} row={row} />
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
          Every graded line
        </h4>
        <div className="flex flex-col gap-3">
          {players.map((player) => (
            <div key={player.player_id} className="flex items-start gap-2" data-testid="review-player">
              <PlayerAvatar playerId={player.player_id} name={player.player_name ?? player.player_id} size={32} />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-pr-text">
                  {player.player_name ?? player.player_id}
                </p>
                <ul className="mt-0.5 flex flex-col gap-0.5">
                  {player.rows.map((row) => (
                    <li
                      key={row.stat}
                      className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm text-pr-text-dim"
                    >
                      <span>
                        Proj {row.predicted.toFixed(1)} {"\u2192"} {row.actual} actual {"\u00b7"} {statLabel(row.stat)}
                        {!row.made_before_tip && (
                          <span className="ml-2 text-xs font-normal uppercase tracking-wide text-pr-lean">
                            Made after tip-off
                          </span>
                        )}
                      </span>
                      <span className="flex shrink-0 items-center gap-2">
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
              </div>
            </div>
          ))}
        </div>
      </div>

      <p className="mt-3 max-w-3xl text-sm text-pr-text-faint">{grading.grade_basis}</p>
    </section>
  );
}

/** One surprise as a card: the player, the line, and both judgements of it. */
function GradedRowCard({ row }: { row: PropGradingRow }) {
  return (
    <li className="flex items-center gap-3 rounded-md border border-pr-rule bg-pr-panel px-3 py-2">
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

/** The stat's own short name, as the box score names it. */
function statLabel(stat: string): string {
  return STAT_LABELS[stat as StatKey] ?? stat;
}

export default PostGameReview;
