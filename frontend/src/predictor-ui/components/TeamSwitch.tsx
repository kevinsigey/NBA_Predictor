// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState, type ReactNode } from "react";

export type TeamOption = { key: string; label: string };

/** A two-option segmented control: one team at a time, never both side by side. */
export function TeamToggle({
  teams,
  value,
  onChange,
  label = "Team",
}: {
  teams: TeamOption[];
  value: string;
  onChange: (key: string) => void;
  label?: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-pr border border-pr-rule">
      {teams.map((team) => {
        const on = team.key === value;
        return (
          <button
            key={team.key}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(team.key)}
            className={`min-h-11 min-w-16 px-4 font-pr-display text-sm font-semibold uppercase tracking-wide ${on ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-dim hover:text-pr-text"}`}
          >
            {team.label}
          </button>
        );
      })}
    </div>
  );
}

/** The toggle plus the selected team's content. First team (home) is the default. */
export function TeamSwitch({
  teams,
  label,
  defaultKey,
  children,
}: {
  teams: TeamOption[];
  label?: string;
  defaultKey?: string;
  children: (teamKey: string) => ReactNode;
}) {
  const [key, setKey] = useState(defaultKey ?? teams[0]?.key ?? "");
  return (
    <div>
      <div className="mb-2">
        <TeamToggle teams={teams} value={key} onChange={setKey} label={label} />
      </div>
      {children(key)}
    </div>
  );
}
