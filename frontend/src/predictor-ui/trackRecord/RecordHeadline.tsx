// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { Moment } from "../components/StatusBadge";
import { marketRank } from "./vocabulary";
import { wilsonInterval } from "./wilson";
import type { HeadlineMode, MarketTally, Tally } from "./types";

/**
 * The headline record: one line per market, the lead one big, and the range the
 * sample allows under each.
 *
 * Ported from NBA's `RecordHeadline`/`TallyLine` so NBA's page is unchanged when
 * it moves onto the kit. The one addition is `mode`: `preKickoff` (NBA's choice)
 * uses the pre-tip subset as the big number; `counted` (PRODUCT.md, the default
 * for the other four) uses every recorded pick and prints the pre-kickoff figure
 * beside it.
 */

/** Below this many graded picks the page says so, instead of letting a rate look settled. */
export const EARLY_SAMPLE = 30;

const whole = (x: number) => `${Math.round(x * 100)}%`;

type Line = { key: string; label: string; correct: number; n: number; beside: Tally | null };

/** One counted-record line: "5 of 10", and a bar showing the range that sample allows. */
function TallyLine({ line, lead, moment }: { line: Line; lead?: boolean; moment: Moment }) {
  const range = wilsonInterval(line.correct, line.n);
  const rate = line.n > 0 ? line.correct / line.n : null;
  return (
    <div
      data-testid={`tally-${line.key}`}
      className="grid grid-cols-[minmax(7rem,9rem)_1fr] items-center gap-x-4 gap-y-1 sm:grid-cols-[minmax(7rem,9rem)_8rem_1fr]"
    >
      <span className={lead ? "font-pr-display text-sm font-semibold text-pr-text" : "text-sm text-pr-text-dim"}>{line.label}</span>
      <span className={`tabular-nums ${lead ? "font-pr-display text-3xl font-semibold" : "text-lg font-semibold"} text-pr-text`}>
        {line.correct}
        <span className="text-pr-text-faint"> of </span>
        {line.n}
        {rate != null && <span className="ml-2 text-sm font-normal text-pr-text-dim">{whole(rate)}</span>}
      </span>
      <div className="col-span-2 sm:col-span-1">
        {range ? (
          <>
            <div
              role="img"
              aria-label={`Plausible range ${whole(range[0])} to ${whole(range[1])}, 50 percent marked`}
              className="relative h-2 rounded-pr bg-pr-panel-2"
            >
              <span
                className="absolute inset-y-0 rounded-pr bg-pr-accent/70"
                style={{ left: `${range[0] * 100}%`, width: `${Math.max(1, (range[1] - range[0]) * 100)}%` }}
              />
              {rate != null && (
                <span className="absolute -inset-y-0.5 w-0.5 bg-pr-text" style={{ left: `calc(${rate * 100}% - 1px)` }} />
              )}
              <span className="absolute -inset-y-1 left-1/2 w-px bg-pr-text-faint" />
            </div>
            <p className="mt-1 text-xs tabular-nums text-pr-text-faint">
              Could plausibly be {whole(range[0])}–{whole(range[1])}
            </p>
            {line.beside && (
              <p className="mt-1 text-xs text-pr-text-faint">
                {line.beside.correct} of {line.beside.n} made before {moment}
              </p>
            )}
          </>
        ) : (
          <span className="text-xs text-pr-text-faint">Nothing graded yet</span>
        )}
      </div>
    </div>
  );
}

/** The record, led by the market the caller names. */
export function RecordHeadline({
  markets,
  mode = "counted",
  lead,
  moment = "kickoff",
}: {
  markets: MarketTally[];
  mode?: HeadlineMode;
  lead?: string;
  /** The sport's word for the start, for the pre-kickoff figure beside the headline. */
  moment?: Moment;
}) {
  const lines: Line[] = [];
  for (const market of markets) {
    const big = mode === "preKickoff" ? market.preKickoff : market.counted;
    if (!big) continue;
    lines.push({
      key: market.key,
      label: market.label,
      correct: big.correct,
      n: big.n,
      beside: mode === "counted" ? market.preKickoff ?? null : null,
    });
  }
  // The fixed reading order, then the lead market to the front.
  lines.sort((a, b) => marketRank(a.key) - marketRank(b.key));
  const leadKey = lead ?? lines[0]?.key;
  if (leadKey) {
    const at = lines.findIndex((l) => l.key === leadKey);
    if (at > 0) lines.unshift(...lines.splice(at, 1));
  }

  if (lines.length === 0) return null;
  const [first, ...rest] = lines;
  return (
    <div className="flex flex-col gap-4 rounded-pr border border-pr-rule bg-pr-panel p-5">
      <TallyLine line={first} lead moment={moment} />
      {first.n < EARLY_SAMPLE && (
        <p data-testid="early-sample" className="text-sm text-pr-text-dim">
          Early sample: {first.n} counted picks is not enough to call this model good or bad. The bar shows the range the
          true rate could sit in; the thin line is 50%.
        </p>
      )}
      {rest.length > 0 && (
        <div className="flex flex-col gap-3 border-t border-pr-rule pt-4">
          {rest.map((line) => (
            <TallyLine key={line.key} line={line} moment={moment} />
          ))}
        </div>
      )}
    </div>
  );
}
