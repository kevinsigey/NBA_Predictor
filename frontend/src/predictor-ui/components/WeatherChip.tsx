// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * Kickoff-hour conditions as one small chip: a picture, then the words.
 *
 * The words are always there, so the picture never has to carry the meaning
 * alone and a screen reader gets the same sentence as everyone else. That is the
 * product's standing rule (PRODUCT.md, "no meaning carried by colour alone") and
 * it is why this is a `<span>` of text with an icon in front of it rather than an
 * icon with a tooltip.
 *
 * `conditions` is the object the sport API serves: `kind`, `temp_f`, `wind_mph`,
 * `precip_pct`. The WMO-code-to-kind mapping lives in the API, once; this
 * component draws a `kind` it knows and renders NOTHING for one it does not,
 * because a sky drawn as a guessed one is a false statement rather than a rough
 * edge. No conditions, no chip — the absence of a chip says nothing, which is the
 * honest state for a game two weeks out or a stadium we have no forecast for.
 *
 * The thresholds are what a football fan needs, not what the data carries: wind
 * under `WINDY_MPH` is noise on a field, and a 20% chance of rain is not worth
 * the words. Both are named constants so a reader can see the line rather than
 * find it in a comparison.
 */
export type WeatherKind = "clear" | "partly" | "cloudy" | "fog" | "rain" | "snow" | "storm" | "dome";

export type Conditions = {
  kind: WeatherKind;
  temp_f?: number | null;
  wind_mph?: number | null;
  precip_pct?: number | null;
};

const WORDS: Record<WeatherKind, string> = {
  clear: "Clear", partly: "Partly cloudy", cloudy: "Cloudy", fog: "Fog",
  rain: "Rain", snow: "Snow", storm: "Storms", dome: "Indoors",
};

/** Wind worth naming on a football field. Below this it is noise. */
export const WINDY_MPH = 15;

/** A chance of precipitation worth naming. Below this the sky decides, not the number. */
export const WET_ENOUGH_PCT = 30;

/** The kinds where a chance of rain or snow is what the number is about. */
const WET: ReadonlySet<WeatherKind> = new Set<WeatherKind>(["rain", "snow", "storm"]);

/** One stroke style for every icon, so the set reads as one hand's work. */
const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

function Icon({ kind }: { kind: WeatherKind }) {
  const cloud = <path d="M7 18h9.5a3.5 3.5 0 0 0 .4-6.98A5 5 0 0 0 7.3 9.6 4.2 4.2 0 0 0 7 18Z" />;
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false" {...stroke}>
      {kind === "clear" && (<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>)}
      {kind === "partly" && (<><circle cx="8" cy="8" r="3" /><path d="M8 2v1.5M2 8h1.5M3.8 3.8l1 1M12.2 3.8l-1 1" /><g transform="translate(2 2)">{cloud}</g></>)}
      {kind === "cloudy" && cloud}
      {kind === "fog" && (<><path d="M5 9h14M3 13h18M6 17h12" /></>)}
      {kind === "rain" && (<>{cloud}<path d="M9 20l-1 2M13 20l-1 2M17 20l-1 2" /></>)}
      {kind === "snow" && (<>{cloud}<path d="M9 20.5h.01M13 21.5h.01M17 20.5h.01" strokeWidth={2.6} /></>)}
      {kind === "storm" && (<>{cloud}<path d="M12.5 15l-2.5 4h3l-2 3" /></>)}
      {kind === "dome" && (<><path d="M3 18a9 9 0 0 1 18 0" /><path d="M2 18h20M12 9V6" /></>)}
    </svg>
  );
}

/**
 * The sentence, as words — exported so a caller (and a site that wants the same
 * figures somewhere else) reads one definition of which figures are worth saying.
 */
export function weatherSentence(c: Conditions): string {
  const parts = [WORDS[c.kind]];
  if (c.kind !== "dome") {
    if (typeof c.temp_f === "number") parts.push(`${Math.round(c.temp_f)}°F`);
    if (typeof c.wind_mph === "number" && c.wind_mph >= WINDY_MPH) {
      parts.push(`${Math.round(c.wind_mph)} mph wind`);
    }
    if (typeof c.precip_pct === "number" && c.precip_pct >= WET_ENOUGH_PCT && WET.has(c.kind)) {
      parts.push(`${Math.round(c.precip_pct)}% chance`);
    }
  }
  return parts.join(" · ");
}

/**
 * `hasOwnProperty`, not `kind in WORDS` and not a `Set`.
 *
 * Both alternatives are wrong in the same direction: an object handed straight
 * from parsed JSON has `Object.prototype` on its chain, so `"toString" in WORDS`
 * is TRUE and the chip would render the function's source as the sky. The test
 * pins it, because "renders nothing for a kind it does not know" is exactly the
 * promise that a prototype key breaks.
 */
export function WeatherChip({ conditions }: { conditions?: Conditions | null }) {
  if (!conditions || !Object.prototype.hasOwnProperty.call(WORDS, conditions.kind)) return null;
  return (
    <span
      data-testid="weather-chip"
      data-kind={conditions.kind}
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-pr border border-pr-rule bg-pr-panel-2 px-1.5 py-0.5 font-pr-body text-xs text-pr-text-dim"
    >
      <Icon kind={conditions.kind} />
      <span>{weatherSentence(conditions)}</span>
    </span>
  );
}