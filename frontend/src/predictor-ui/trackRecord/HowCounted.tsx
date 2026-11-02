// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
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
  asOf,
  children,
}: {
  moment?: Moment;
  headline?: HeadlineMode;
  /** ISO timestamp of the snapshot this record was read from, shown as a
   * "Record as of …" line. Absent or unparseable renders nothing — the
   * stamp is a courtesy, never a reason to fail the block. */
  asOf?: string | null;
  children?: ReactNode;
}) {
  const asOfDate = asOf ? new Date(asOf) : null;
  const asOfText =
    asOfDate && !Number.isNaN(asOfDate.getTime())
      ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(asOfDate)
      : null;
  return (
    <details className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm">
      <summary className="cursor-pointer font-semibold text-pr-text">How this is counted</summary>
      <div className="mt-3 flex max-w-prose flex-col gap-2 text-pr-text-dim">
        <p>{CORE}</p>
        <p>{modeLine(headline, moment)}</p>
        {asOfText && <p>Record as of {asOfText}.</p>}
        {children}
      </div>
    </details>
  );
}
