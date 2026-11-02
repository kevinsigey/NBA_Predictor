// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { HubTabs, type HubTab } from "../trackRecord/HubTabs";
import type { ReactNode } from "react";

export type { HubTab as SportTab };

type Props = {
  tabs: HubTab[];
  active: string;
  onChange: (id: string) => void;
  panels?: Record<string, ReactNode>;
  label?: string;
  className?: string;
};

/**
 * Per-sport tab row. Built on `HubTabs` (tablist, roving tabindex, arrow keys);
 * adds overflow containment with a fade that appears only when the row
 * overflows, and scrolls the active tab into view.
 */
export function SportTabs({ tabs, active, onChange, panels, label = "Pages", className }: Props) {
  return (
    <div className={`min-w-0 max-w-full${className ? ` ${className}` : ""}`}>
      <HubTabs tabs={tabs} active={active} onChange={onChange} panels={panels} label={label} />
    </div>
  );
}