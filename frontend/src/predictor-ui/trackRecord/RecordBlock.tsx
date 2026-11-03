// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";

/**
 * A titled block. The heading asks the question a reader arrived with ("Week by
 * week", "Does 70% mean 70%?"); the optional caption is one line of context. No
 * blurb under it: the data under the title is the explanation. Ported from NBA's
 * `Block`, given the caption the plan asks for and a region role so a screen
 * reader can jump between sections.
 */
export function RecordBlock({
  id,
  title,
  caption,
  children,
}: {
  id: string;
  title: string;
  caption?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="flex flex-col gap-3">
      <h3 id={`${id}-h`} className="font-pr-display text-base font-semibold text-pr-text">
        {title}
      </h3>
      {caption && <p className="max-w-prose text-sm text-pr-text-dim">{caption}</p>}
      {children}
    </section>
  );
}
