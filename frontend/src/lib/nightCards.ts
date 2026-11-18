import { contrast, kickoff, parseKickoff, pct, stat, type Segment, type Side, type Status } from "../predictor-ui";
import type { Game } from "../api/client";
import { signedLine } from "./lines";
import { favourite, marginLine, marketFavourite, pickWon } from "./pick";
import { teamColor, teamName } from "./teams";

export type CardModel = {
  left: Side;
  right: Side;
  centre: string;
  status?: Status;
  pick?: { label: string; prob: number };
  market?: { label: string; prob: number } | null;
  when: string;
  meta?: string;
  bar?: Segment[];
};

// A game past tip-off with no final score is Live only this long; after that
// the score feed is behind (or the game was postponed): "Awaiting result".
const LIVE_WINDOW_MS = 3.5 * 3600_000;
const DAY_MS = 86_400_000;

const isFinal = (g: Game) => g.completed && g.home_pts != null && g.away_pts != null;
const tipMs = (g: Game) => (g.tip_off ? parseKickoff(g.tip_off).getTime() : null);

function localTime(iso: string, timeZone?: string): string {
  return parseKickoff(iso).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone });
}

// Two clubs can share a primary colour (MIN and NOP are both navy). Then the
// bar falls back to the family's neutral fills so the halves stay apart.
function barColours(away: string, home: string): [string | undefined, string | undefined] {
  const a = teamColor(away);
  const h = teamColor(home);
  if (!a || !h || contrast(a, h) < 1.5) return [undefined, undefined];
  return [a, h];
}

/**
 * One NBA game as a family MatchCard: away left, home right (US order).
 *
 * The honesty rule: the API sends the latest pick made before tip-off, or,
 * when only a backtest exists, that pick flagged rebuilt. A final is judged
 * only on a pre-tip pick; a rebuilt one is labelled and never judged.
 */
export function toCardModel(game: Game, isNext: boolean, timeZone?: string, now: number = Date.now()): CardModel {
  const final = isFinal(game);
  const tip = tipMs(game);
  // The day is the game's own (Eastern) date, so it always matches the day
  // heading it sits under; only the tip time moves to the viewer's zone.
  const day = kickoff(game.game_date);
  const model: CardModel = {
    left: { code: game.away_team, name: teamName(game.away_team), color: teamColor(game.away_team) },
    right: { code: game.home_team, name: teamName(game.home_team), color: teamColor(game.home_team) },
    centre: final ? `${game.away_pts}–${game.home_pts}` : game.tip_off ? localTime(game.tip_off, timeZone) : "at",
    when: final ? `${day} · Final` : day,
  };

  const p = game.prediction;
  if (p) {
    const fav = favourite(p, game.home_team, game.away_team);
    const [awayColour, homeColour] = barColours(game.away_team, game.home_team);
    model.pick = { label: fav.team, prob: fav.prob };
    const mk = marketFavourite(p, game.home_team, game.away_team);
    model.market = mk && { label: mk.team, prob: mk.prob };
    model.bar = [
      { label: game.away_team, prob: 1 - p.home_win_probability, color: awayColour },
      { label: game.home_team, prob: p.home_win_probability, color: homeColour },
    ];
    const parts: string[] = [marginLine(p, game.home_team, game.away_team)];
    // The chance the home side covers the listed spread, and the chance of the over: only when the books quote a line.
    if (p.cover_prob_spread != null && p.market_point != null) {
      parts.push(`${game.home_team} ${signedLine(p.market_point)} · ${pct(p.cover_prob_spread)}`);
    }
    // With a listed line the total reads as the chance of the over; without one it is the model's own total, as before.
    if (p.cover_prob_total != null && p.market_total_point != null) {
      parts.push(`Over ${stat(p.market_total_point)} · ${pct(p.cover_prob_total)}`);
    } else {
      parts.push(`Total ${stat(p.predicted_total)}`);
    }
    model.meta = parts.join(" · ");
  }

  // Who is out, in the card's own words: surnames, upcoming games only.
  if (!final && game.key_out?.length) {
    const out = `Out: ${game.key_out.map((k) => k.player_name.trim().split(/\s+/).slice(-1)[0]).join(", ")}`;
    model.meta = model.meta ? `${model.meta} · ${out}` : out;
  }

  if (p && game.rebuilt) {
    // Built after tip-off: labelled from the moment it exists, never judged.
    model.status = "rebuilt";
    if (!final && tip !== null && now - tip >= LIVE_WINDOW_MS) model.when = `${day} · Awaiting result`;
  } else if (final) {
    if (!p) model.status = "nopick";
    else model.status = pickWon(p, game.home_pts!, game.away_pts!) ? "called" : "missed";
  } else if (tip !== null && tip <= now) {
    if (now - tip < LIVE_WINDOW_MS) model.status = "live";
    else model.when = `${day} · Awaiting result`;
  } else if (tip === null && now > Date.parse(`${game.game_date}T12:00:00Z`) + DAY_MS) {
    // No tip time, and the game date is well past: never guess Live.
    model.when = `${day} · Awaiting result`;
  } else if (isNext) {
    model.status = "next";
  }
  return model;
}

/** "Next up": every game at the earliest tip still to come, in the current week only. */
export function nextUpIds(games: Game[], isCurrentWeek: boolean, now: number = Date.now()): Set<string> {
  if (!isCurrentWeek) return new Set();
  const future = games.filter((g) => !isFinal(g) && (tipMs(g) ?? -Infinity) > now);
  if (future.length === 0) return new Set();
  const first = Math.min(...future.map((g) => tipMs(g)!));
  return new Set(future.filter((g) => tipMs(g) === first).map((g) => g.game_id));
}

/** The week's record: finals with a pick made before tip-off; rebuilt ones counted apart. */
export function weekTally(games: Game[]): { hits: number; settled: number; rebuilt: number } {
  const judged = games.filter((g) => isFinal(g) && g.prediction);
  const counted = judged.filter((g) => !g.rebuilt);
  return {
    hits: counted.filter((g) => pickWon(g.prediction!, g.home_pts!, g.away_pts!)).length,
    settled: counted.length,
    rebuilt: judged.length - counted.length,
  };
}

const utc = (iso: string) => new Date(`${iso}T12:00:00Z`);
// en-US parts (three-letter "Sep"), laid out day-first like fmt.kickoff.
const part = (d: Date, o: Intl.DateTimeFormatOptions) => d.toLocaleDateString("en-US", { timeZone: "UTC", ...o });

/** "20–26 Oct", or "27 Oct – 2 Nov" across a month end. */
export function weekLabel(weekStart: string): string {
  const start = utc(weekStart);
  const end = new Date(start.getTime() + 6 * DAY_MS);
  const [sd, sm] = [part(start, { day: "numeric" }), part(start, { month: "short" })];
  const [ed, em] = [part(end, { day: "numeric" }), part(end, { month: "short" })];
  return sm === em ? `${sd}–${ed} ${em}` : `${sd} ${sm} – ${ed} ${em}`;
}

/** "Tuesday 21 Oct". Game dates are US Eastern calendar dates. */
export function dayHeading(gameDate: string): string {
  const d = utc(gameDate);
  return `${part(d, { weekday: "long" })} ${part(d, { day: "numeric" })} ${part(d, { month: "short" })}`;
}

/** The zone(s) the week's tip times are shown in: "CDT", or "CDT/CST" across a clock change. */
export function tipZones(games: Game[], timeZone?: string): string {
  const zones = games
    .filter((g) => g.tip_off)
    .map((g) => kickoff(g.tip_off!, timeZone).split(" ").pop() ?? "");
  return [...new Set(zones)].filter(Boolean).join("/");
}
