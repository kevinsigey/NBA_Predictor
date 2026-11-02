// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useId, useRef, useEffect, useState, useCallback, type KeyboardEvent, type ReactNode } from "react";

/** One tab: a stable id and the words on it. */
export type HubTab = { id: string; label: string };

/**
 * The shared tab bar, for all five sports.
 *
 * Replaces four ad-hoc bars (NBA's `aria-current` buttons, Sports' `aria-pressed`
 * buttons, PL's strings, F1's `display:none` panels) with one tablist: roving
 * tabindex, arrow-key movement, `aria-selected` announced, and a `role="tabpanel"`
 * wired back to its tab. The bar contains its own overflow (`overflow-x-auto`),
 * with a fade at the right edge as the cue, so it never clips and never widens
 * the page at 375px.
 */
export function HubTabs({
  tabs,
  active,
  onChange,
  panels,
  label = "Sections",
}: {
  tabs: HubTab[];
  active: string;
  onChange: (id: string) => void;
  /** The panel body per tab id. Only the active one is rendered. */
  panels?: Record<string, ReactNode>;
  /** The tablist's accessible name, e.g. "Sections" or "Pages". */
  label?: string;
}) {
  const uid = useId();
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [overflowing, setOverflowing] = useState(false);
  const tabId = (t: HubTab) => `${uid}-tab-${t.id}`;
  const panelId = (t: HubTab) => `${uid}-panel-${t.id}`;

  const moveTo = (index: number) => {
    if (tabs.length === 0) return;
    const next = tabs[(index + tabs.length) % tabs.length];
    onChange(next.id);
    refs.current[next.id]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      moveTo(index + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      moveTo(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      moveTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      moveTo(tabs.length - 1);
    }
  };

  const currentTab = tabs.find((t) => t.id === active) ?? tabs[0];

  const checkOverflow = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    setOverflowing(el.scrollWidth > el.clientWidth + 1);
  }, []);

  useEffect(() => {
    const el = scrollerRef.current;
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
  }, [checkOverflow, tabs]);

  useEffect(() => {
    if (!currentTab) return;
    const el = refs.current[currentTab.id];
    el?.scrollIntoView?.({ inline: "nearest", block: "nearest", behavior: "smooth" });
  }, [currentTab]);

  return (
    <div className="min-w-0 max-w-full">
      <div className="relative min-w-0 max-w-full">
        <div
          ref={scrollerRef}
          data-testid="hub-tabs-scroller"
          role="tablist"
          aria-label={label}
          className="flex gap-4 overflow-x-auto"
        >
          {tabs.map((tab, index) => {
            const current = tab.id === currentTab?.id;
            return (
              <button
                key={tab.id}
                ref={(el) => {
                  refs.current[tab.id] = el;
                }}
                id={tabId(tab)}
                role="tab"
                type="button"
                aria-selected={current}
                aria-controls={panelId(tab)}
                tabIndex={current ? 0 : -1}
                onClick={() => onChange(tab.id)}
                onKeyDown={(event) => onKeyDown(event, index)}
                className={`inline-flex items-center whitespace-nowrap border-b-2 pb-3 text-sm font-semibold transition-colors max-sm:min-h-10 ${
                  current ? "border-pr-accent text-pr-text" : "border-transparent text-pr-text-dim hover:text-pr-text"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
        {overflowing && (
          <span
            aria-hidden="true"
            data-testid="hub-tabs-fade"
            className="pointer-events-none absolute inset-y-0 right-0 w-6 bg-gradient-to-l from-pr-stage to-transparent"
          />
        )}
      </div>
      {currentTab && panels && (
        <div role="tabpanel" id={panelId(currentTab)} aria-labelledby={tabId(currentTab)} tabIndex={0} className="pt-4">
          {panels[currentTab.id]}
        </div>
      )}
    </div>
  );
}
