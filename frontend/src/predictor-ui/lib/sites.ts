// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { SiteLink, Sport } from "../components/AppFrame";

/**
 * The family's six entries, in the order the switcher has always shown them:
 * the hub first, then the five sports. Labels live here once; a relabel is a
 * one-line change in this package instead of four site PRs.
 */
export const SPORT_SITES: { sport: Sport; label: string }[] = [
  { sport: "hub", label: "Home" },
  { sport: "pl", label: "Premier League" },
  { sport: "nfl", label: "NFL" },
  { sport: "cfb", label: "College football" },
  { sport: "nba", label: "NBA" },
  { sport: "f1", label: "F1" },
];

/**
 * The family switcher's links for one site. The current site is a relative
 * link (`/`, or `?sport=` for the shared NFL/CFB app); every other sport is its
 * own subdomain. `host` is passed in so a real domain is a call-site change,
 * not a component edit.
 */
export function siteLinks(current: Sport, host: string): SiteLink[] {
  return SPORT_SITES.map(({ sport, label }) => ({
    sport,
    label,
    href:
      sport === current
        ? sport === "nfl"
          ? "?sport=nfl"
          : sport === "cfb"
            ? "?sport=cfb"
            : "/"
        : sport === "hub"
          ? `https://${host}`
          : sport === "nfl" || sport === "cfb"
            ? `https://sports.${host}/?sport=${sport}`
            : `https://${sport}.${host}`,
  }));
}
