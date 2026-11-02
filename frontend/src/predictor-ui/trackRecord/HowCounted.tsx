// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";
import type { Moment } from "../components/StatusBadge";
import type { HeadlineMode } from "./types";

/**
 * "How this is counted", collapsed, stated once for the whole family.
 *
 * The two sentences the old copy got wrong are gone. The counting rule is a
 * fact (the earliest recorded pick per game and market, never double-counted);
 * the meaning of the headline is a property of the mode, and it is said in the
 * sport's own word for the start. Extra disclosure a sport has — pushes,
 * markets with no result rule yet, a comparison method — is passed as children.
 */
const CORE =
  "One counted pick per game and market: the earliest recorded. A re-run of the model never replaces it and is never counted twice.";

function modeLine(mode: HeadlineMode, moment: Moment): string {
  return mode === "preKickoff"
    ? `The record above shows every pick made before ${moment}. Picks rebuilt after ${moment} are listed separately, below.`
    : `The record above counts every recorded pick. The figure beside it is the subset made before ${moment}, and every pick below says when it was made.`;
}

export function HowCounted({
  moment = "kickoff",
  headline = "counted",
  children,
}: {
  moment?: Moment;
  headline?: HeadlineMode;
  children?: ReactNode;
}) {
  return (
    <details className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm">
      <summary className="cursor-pointer font-semibold text-pr-text">How this is counted</summary>
      <div className="mt-3 flex max-w-prose flex-col gap-2 text-pr-text-dim">
        <p>{CORE}</p>
        <p>{modeLine(headline, moment)}</p>
        {children}
      </div>
    </details>
  );
}
