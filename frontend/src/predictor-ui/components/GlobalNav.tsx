// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useCallback, useEffect, useRef, useState } from "react";
import type { Sport } from "./AppFrame";

export type NavEntry = { id: string; label: string; href: string; sport?: Sport };

type Props = {
  entries: NavEntry[];
  activeId: string | null;
  ariaLabel?: string;
};

/**
 * The shared top bar: one horizontally-scrolling row of cross-site anchors.
 * The fade is rendered only when the row actually overflows, and the active
 * entry is scrolled into view on change. No page-level horizontal scroll.
 */
export function GlobalNav({ entries, activeId, ariaLabel = "Sports" }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [overflowing, setOverflowing] = useState(false);

  const checkOverflow = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setOverflowing(el.scrollWidth > el.clientWidth + 1);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    checkOverflow();
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(checkOverflow);
      ro.observe(el);
    }
    el.addEventListener("scroll", checkOverflow);
    window.addEventListener("resize", checkOverflow);
    return () => {
      ro?.disconnect();
      el.removeEventListener("scroll", checkOverflow);
      window.removeEventListener("resize", checkOverflow);
    };
  }, [checkOverflow, entries]);

  useEffect(() => {
    if (!activeId) return;
    const el = scrollRef.current?.querySelector(`[data-nav-id="${activeId}"]`) as HTMLElement | null;
    el?.scrollIntoView?.({ inline: "nearest", block: "nearest", behavior: "smooth" });
  }, [activeId]);

  return (
    <nav aria-label={ariaLabel} className="relative min-w-0 max-w-full max-sm:flex-1">
      <div
        ref={scrollRef}
        data-testid="global-nav-scroller"
        className="flex gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {entries.map((entry) => {
          const current = entry.id === activeId;
          return (
            <a
              key={entry.id}
              href={entry.href}
              data-nav-id={entry.id}
              aria-current={current ? "page" : undefined}
              className={`inline-flex items-center whitespace-nowrap rounded-pr px-2.5 py-1 font-pr-display text-sm font-semibold uppercase tracking-wide transition-colors max-sm:min-h-10 ${
                current ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-dim hover:text-pr-text"
              }`}
            >
              {entry.label}
            </a>
          );
        })}
      </div>
      {overflowing && (
        <span
          aria-hidden="true"
          data-testid="global-nav-fade"
          className="pointer-events-none absolute inset-y-0 right-0 w-6 bg-gradient-to-l from-pr-stage to-transparent"
        />
      )}
    </nav>
  );
}