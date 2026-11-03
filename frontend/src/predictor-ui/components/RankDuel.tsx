// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * One offence-versus-defence duel as two bars.
 *
 * A bar is how good a unit ranks: full is first of the league, a stub is last.
 * The rank is **always printed beside its bar**, so the bar is a picture of the
 * words rather than the only thing carrying them — the same rule `SignalRows`
 * enforces on its rows, and the reason this component draws ranks rather than
 * percentages: "#3 of 32" is a claim a reader can check against the league table,
 * and a bar is a picture of that claim.
 *
 * Nothing here computes a duel. `Duel` lives in each sport repo and its rows
 * arrive in `context.matchups`; this draws the two ranks it is handed.
 */
export type RankDuelProps = {
  attacker: string;
  attackerStat: string;
  attackerRank: number;
  defender: string;
  defenderStat: string;
  defenderRank: number;
  nTeams: number;
};

/** A rank outside the league is a wrong number, and a bar drawn from it is a lie. */
export class RankOutOfRangeError extends Error {
  constructor(rank: number, n: number) {
    super(`rank ${rank} is not within 1..${n}`);
    this.name = "RankOutOfRangeError";
  }
}

/**
 * Share of the track to fill: 1st of n is 100%, last of n is 1/n.
 *
 * **Throws rather than drawing an impossible bar.** `PicksList`'s
 * `OutPlayerInRankingError` and `SignalRows`' `HeadlineFigureMismatchError` are
 * two-for-two in this package for the same shape of mistake — a caller passed
 * something the component must not draw — and a third component answering
 * differently would mean a caller had to learn which rule applied where. A rank
 * of 40 in a league of 32 is a bad fact, and the failure belongs in the adapter
 * that produced it, which has tests.
 *
 * `Number.isInteger` on both, because `3.5` and `n: 1` are both real inputs that
 * would otherwise slip past `<` and `>` comparisons: `1/1` is a full bar, which
 * reads as "first of one", and a half-place rank is not a rank at all.
 */
export function rankFill(rank: number, n: number): number {
  if (!Number.isInteger(rank) || !Number.isInteger(n) || n < 2 || rank < 1 || rank > n) {
    throw new RankOutOfRangeError(rank, n);
  }
  return (n - rank + 1) / n;
}

function Bar({ team, stat, rank, n, tone }: {
  team: string; stat: string; rank: number; n: number; tone: string;
}) {
  // Rounded to one decimal rather than left at full float precision: the bar's
  // width is a drawing, and a track that is 93.75000000000001% wide is the same
  // drawing as one 93.8% wide. `tabular-nums` because the rank beside it is a
  // figure and the pair has to line up with the row above it.
  const pct = Math.round(rankFill(rank, n) * 1000) / 10;
  return (
    <div className="min-w-0 flex-1">
      <div className="flex items-baseline justify-between gap-2 font-pr-body text-xs text-pr-text-dim">
        <span className="truncate">{team} · {stat}</span>
        <span className="shrink-0 tabular-nums font-pr-display text-sm font-semibold text-pr-text">
          #{rank}<span className="font-normal text-pr-text-faint"> of {n}</span>
        </span>
      </div>
      <div className="mt-1 h-1.5 rounded-pr bg-pr-panel-2" role="presentation">
        <div data-testid="rank-fill" className={`h-1.5 rounded-pr ${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function RankDuel(p: RankDuelProps) {
  return (
    <div data-testid="rank-duel" className="flex flex-col gap-2 sm:flex-row sm:gap-4">
      <Bar team={p.attacker} stat={p.attackerStat} rank={p.attackerRank} n={p.nTeams} tone="bg-pr-accent" />
      <Bar team={p.defender} stat={p.defenderStat} rank={p.defenderRank} n={p.nTeams} tone="bg-pr-fill-mute" />
    </div>
  );
}