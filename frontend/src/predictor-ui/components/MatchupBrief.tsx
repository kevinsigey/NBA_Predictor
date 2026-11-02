// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The Matchup section: how the two sides' units meet, as data, with no AI.
 *
 * Rendered above the AI panel and visible before anything is asked for. It draws
 * what the fixture's `context` carries: code-computed rank duels (`matchups`)
 * pivoted by `pivotMatchups` into a home-left / away-right table (`MatchupTable`) with rank
 * boxes coloured by thirds of the field. With none the component renders NOTHING, not a heading over
 * an empty list and not a placeholder.
 *
 * **Form rows (`form_rows`) are deliberately not drawn.** F1 emits one per driver
 * and constructor per metric — recent form, qualifying pace, track history — which
 * on a full grid is eighteen rows that pushed the fixture this section exists to
 * frame off the screen. They stay in `context` because the AI read is built from
 * them (`prompts.py`), and the read is the right place for them: dropping the draw
 * here does not drop them from the read.
 *
 * Duels stay neutral. `toward_pick` rides along on the row but is not drawn: no
 * "advantage", no "edge", until a lift-gate run proves a duel type predicts.
 * NFL player props are a different block and are not touched here.
 */
import { pivotMatchups } from "../lib/pivotMatchups";
import { MatchupTable } from "./MatchupTable";

/** One code-computed duel, in the shape `signals/matchups.to_context` emits. */
export type MatchupRow = {
  id: string;
  attacker: string;
  defender: string;
  stat: string;
  foil: string;
  attacker_rank: number;
  defender_rank: number;
  n_teams: number;
  toward_pick: boolean | null;
};

/** One labelled form fact (F1: driver/constructor form, quali pace, track history).
 *  Never rendered — see the note at the top. Typed because the payload carries it. */
export type FormRow = {
  id: string;
  subject: string;
  label: string;
  value: string;
  rank: number | null;
  n: number | null;
};

/** What `/explain/{sport}/{id}/context` returns; every key optional. `form_rows`
 *  rides along as read input and as the client's "this fixture already has matchup
 *  context, don't re-fetch it" signal; no panel reads it. */
export type MatchupContext = { matchups?: MatchupRow[]; form_rows?: FormRow[]; player_context?: unknown[] };

/**
 * A loader for the context route, next to the site's own explain request.
 * `baseUrl` is the site's explain base (no trailing slash needed); the id is
 * encoded per path segment so a slash-bearing id still reaches the route.
 * Rejects on a non-2xx, a timeout or bad JSON; callers treat any rejection as
 * "no section".
 */
export function createContextLoader(baseUrl: string, timeoutMs = 10_000) {
  const base = baseUrl.replace(/\/+$/, "");
  return async (id: string): Promise<MatchupContext> => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(`${base}/${id.split("/").map(encodeURIComponent).join("/")}/context`, { signal: ctl.signal });
      if (!res.ok) throw new Error(`context ${res.status}`);
      return (await res.json()) as MatchupContext;
    } finally {
      clearTimeout(timer);
    }
  };
}

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/** A duel the component can draw: both ranks and the league size are integers.
 *  Anything else is dropped rather than thrown on: this is context shown on every
 *  fixture, and one malformed row must not take the page down. */
const isText = (v: unknown): v is string => typeof v === "string" && v.trim() !== "";

const drawable = (m: MatchupRow) =>
  !!m && isText(m.id) && isText(m.attacker) && isText(m.defender) && isText(m.stat) && isText(m.foil) &&
  isInt(m.attacker_rank) && isInt(m.defender_rank) && isInt(m.n_teams) && m.n_teams >= 2 &&
  m.attacker_rank >= 1 && m.attacker_rank <= m.n_teams && m.defender_rank >= 1 && m.defender_rank <= m.n_teams;

export function MatchupBrief({ matchups = [] }: { matchups?: MatchupRow[] }) {
  // `bundle.context` is untyped site data: anything that is not a list is nothing.
  const duels = (Array.isArray(matchups) ? matchups : []).filter(drawable);
  const pivot = pivotMatchups(duels);
  if (!pivot) return null;
  return (
    <section aria-labelledby="matchup-heading" data-testid="matchup-section">
      <h3
        id="matchup-heading"
        className="mb-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        Matchup
      </h3>
      <MatchupTable pivot={pivot} />
    </section>
  );
}
