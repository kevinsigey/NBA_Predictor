import { useMemo } from "react";
import type { PlayerProp } from "../api/client";
import { STAT_LABELS, type StatKey } from "../lib/boxScore";
import { reviewPlayers, type ReviewRow } from "../lib/review";
import PlayerAvatar from "./PlayerAvatar";

/**
 * The post-game review: what the model got right about the players.
 *
 * Three columns, in the reader's words: **Beat the projection**, **Fell
 * short**, and **On the number** -- the count of lines inside the model's own
 * error. A projection is judged against that error (the props model's holdout
 * MAE per stat), never against zero: a player who misses by two points has
 * not been beaten by the model, the model was two points wrong.
 *
 * **Nothing without bands.** The band is read from the props manifest, and a
 * manifest that is missing means there is no measured error to judge against.
 * A review with a made-up band would be a review of a made-up claim, so the
 * section is not drawn at all.
 */
export function PostGameReview({
  props,
  bands,
}: {
  props: PlayerProp[];
  bands: Record<string, number> | null | undefined;
}) {
  const review = useMemo(() => {
    const projections = props.map((p) => ({
      player_id: p.player_id,
      player_name: p.player_name,
      stat: p.stat,
      value: p.predicted_value,
      rebuilt: p.rebuilt,
    }));
    const actuals = props
      .filter((p) => p.actual_value !== null && p.actual_value !== undefined)
      .map((p) => ({ player_id: p.player_id, stat: p.stat, value: p.actual_value as number }));
    return reviewPlayers(projections, actuals, bands);
  }, [props, bands]);

  const hasBands = !!bands && Object.keys(bands).length > 0;
  if (!hasBands || review.total === 0) return null;

  return (
    <section aria-labelledby="player-review-heading" data-testid="post-game-review">
      <h3
        id="player-review-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Player review
      </h3>
      <div className="grid gap-4 sm:grid-cols-3">
        <ReviewColumn title="Beat the projection" rows={review.beat} tone="win" />
        <ReviewColumn title="Fell short" rows={review.short} tone="loss" />
        <div data-testid="review-on-the-number">
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
            On the number
          </h4>
          <p className="text-sm text-pr-text">
            <span className="font-pr-display text-lg font-semibold tabular-nums">{review.onNumber}</span> of{" "}
            <span className="tabular-nums">{review.total}</span> player lines within range
          </p>
          <p className="mt-1 text-sm text-pr-text-faint">
            A line is inside the range when the miss is no bigger than the model's own error for that stat.
          </p>
        </div>
      </div>
    </section>
  );
}

/** One column of the review: the players who beat (or missed) the model. */
function ReviewColumn({
  title,
  rows,
  tone,
}: {
  title: string;
  rows: ReviewRow[];
  tone: "win" | "loss";
}) {
  return (
    <div data-testid={tone === "win" ? "review-beat" : "review-short"}>
      <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{title}</h4>
      {rows.length === 0 ? (
        <p className="text-sm text-pr-text-faint">Nobody.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row) => (
            <li key={`${row.player_id}-${row.stat}`} className="flex items-center gap-2">
              <PlayerAvatar playerId={row.player_id} name={row.player_name} size={32} />
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold text-pr-text">{row.player_name}</span>
                <span className="block text-sm text-pr-text-dim">
                  Proj {row.projected.toFixed(1)} {"\u2192"} {row.actual} actual {"\u00b7"} {statLabel(row.stat)}{" "}
                  <span className={tone === "win" ? "text-pr-win" : "text-pr-loss"}>
                    {row.gap > 0 ? "\u25b2" : "\u25bc"} {row.gap > 0 ? "+" : "\u2212"}
                    {Math.abs(row.gap).toFixed(1)} vs proj
                  </span>{" "}
                  {"\u00b7"} {row.grade}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The stat's own short name, as the box score names it. */
function statLabel(stat: string): string {
  return STAT_LABELS[stat as StatKey] ?? stat;
}

export default PostGameReview;
