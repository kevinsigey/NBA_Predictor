import { useMemo, type ReactNode } from "react";
import type { PlayerProp, PlayerHubRow } from "../api/client";
import {
  STAT_KEYS,
  STAT_LABELS,
  attachHubRows,
  hasActuals,
  pivotProps,
  splitByTeam,
  sumStat,
  type BoxScoreRow,
  type StatKey,
} from "../lib/boxScore";
import { gradeForGap } from "../lib/review";
import { teamName } from "../lib/teams";
import { StickyStatTable, TeamSwitch, stat, type StickyRow } from "../predictor-ui";

interface PlayerBoxScoreProps {
  playerProps: PlayerProp[];
  hubPlayers: PlayerHubRow[];
  homeTeam: string;
  awayTeam: string;
  /** The props model's own error per stat. A delta outside it is coloured; a
   *  delta inside it is the model being as wrong as it said it would be. */
  bands?: Record<string, number> | null;
}

const COLUMNS = STAT_KEYS.map((key) => ({ key, label: STAT_LABELS[key] }));

function statCell(
  row: BoxScoreRow,
  key: StatKey,
  bands?: Record<string, number> | null,
): ReactNode {
  const predicted = row.predicted[key];
  const actual = row.actual[key];
  // A prop rebuilt after tip-off is shown, never judged -- so no delta, which
  // would be a score against a projection made with the result already known.
  const showDelta = actual !== undefined && predicted !== undefined && !row.rebuilt;
  if (actual === undefined) {
    if (predicted === undefined) return null;
    // Before the game, the projection carries its 80% range: the model's
    // own measured spread for a line this size, calibrated on the counted
    // record before this game day. No range was served, no range is drawn.
    const range = row.intervals[key];
    if (!range) return stat(predicted);
    return (
      <>
        {stat(predicted)}
        <span
          className="block text-[0.6875rem] text-pr-text-dim"
          title="The model's 80% range for this line, calibrated on the counted record before this game"
        >
          {stat(range.low)}{"\u2013"}{stat(range.high)}
        </span>
      </>
    );
  }
  const gap = actual - predicted!;
  const band = bands?.[key];
  // The colour says whether the model was wrong by more than it says it can
  // be. Inside the band the delta is plain: the model was as wrong as it
  // promised, which is not a colour.
  const tone =
    band === undefined
      ? ""
      : gap > band
        ? "text-pr-win"
        : gap < -band
          ? "text-pr-loss"
          : "";
  return (
    <>
      {predicted === undefined ? "—" : stat(predicted)}
      <span className="block text-[0.6875rem] text-pr-text-dim" title={showDelta ? "Actual, and how far off the prediction was" : "Actual"}>
        {actual}
        {showDelta && (
          <>
            {" "}(<span className={tone}>{actual - predicted > 0 ? "+" : ""}
            {(actual - predicted).toFixed(1)}</span>){band !== undefined ? ` ${"\u00b7"} ${gradeForGap(gap, band)}` : ""}
          </>
        )}
      </span>
    </>
  );
}

function playerRow(row: BoxScoreRow, bands?: Record<string, number> | null): StickyRow {
  return {
    key: row.player_id,
    testId: "box-score-row",
    sortValues: [row.player_name, row.predicted.points ?? null, row.predicted.rebounds ?? null, row.predicted.assists ?? null, row.predicted.threes ?? null],
    header: (
      <>
        {row.player_name}
        {row.rebuilt && (
          <span className="block text-[0.5625rem] uppercase tracking-wider text-pr-text-dim" title="Built after tip-off. Shown, never judged.">
            Rebuilt
          </span>
        )}
      </>
    ),
    cells: STAT_KEYS.map((key) => statCell(row, key, bands)),
  };
}

function teamFooter(rows: BoxScoreRow[]): StickyRow[] {
  // **No projected total.** The sum of every rostered player's projection is
  // not the team's projection: it adds up the players who never get minutes,
  // and a team was shown a "Total 139.7" against an actual 121. The team's
  // own projected total is on the page already, in the pick block. The ACTUAL
  // total is a fact, so it is shown once the game is over.
  if (!hasActuals(rows)) return [];
  return [
    {
      key: "actual",
      testId: "box-score-actual-totals",
      total: true,
      header: "Actual",
      sortValues: ["Actual", ...STAT_KEYS.map((key) => sumStat(rows, key, "actual"))],
      cells: STAT_KEYS.map((key) => sumStat(rows, key, "actual")),
    },
  ];
}

/**
 * The predicted box score, one team at a time.
 *
 * A phone cannot hold two rosters side by side without clipping names, so the
 * reader picks a team (home first, the default) and gets the shared
 * StickyStatTable: full player names pinned on the left, the four stats sliding
 * sideways beside them, totals last. Players are placed by the same
 * splitByTeam as before; a player who cannot be placed is named in a note, not
 * guessed onto a side.
 */
export default function PlayerBoxScore({ playerProps, hubPlayers, homeTeam, awayTeam, bands }: PlayerBoxScoreProps) {
  const { away, home, unattributed } = useMemo(
    () => splitByTeam(attachHubRows(pivotProps(playerProps), hubPlayers), homeTeam, awayTeam),
    [playerProps, hubPlayers, homeTeam, awayTeam],
  );

  // No player could be placed on either side. Rendering nothing at all would be
  // a silent deletion: the caller asked for a box score and would get silence
  // with no hint that anything failed. Say what happened instead.
  if (home.length + away.length === 0) {
    return (
      <div className="mt-5 text-sm text-pr-text-dim" data-testid="player-box-score">
        {hubPlayers.length === 0
          ? "Player breakdown unavailable — the team feed did not load, so nobody could be matched to a side. Guessing a side would be worse than saying nothing."
          : "No player could be matched to either team."}
      </div>
    );
  }

  const sides = [
    { key: "home", label: homeTeam || "Home", name: teamName(homeTeam) || "Home", rows: home },
    { key: "away", label: awayTeam || "Away", name: teamName(awayTeam) || "Away", rows: away },
  ];
  const byKey = Object.fromEntries(sides.map((side) => [side.key, side]));

  return (
    <div className="mt-5" data-testid="player-box-score">
      <TeamSwitch
        teams={sides.map(({ key, label }) => ({ key, label }))}
        label="Box score team"
        // Home is the default, unless only the away side has anyone in it.
        defaultKey={home.length === 0 ? "away" : "home"}
      >
        {(key) => {
          const side = byKey[key];
          return (
            <StickyStatTable
              testId="box-score-scroll"
              caption={`${side.name} projected box score`}
              columns={COLUMNS}
              sections={[{ rows: side.rows.map((row) => playerRow(row, bands)) }]}
              footer={side.rows.length > 0 ? teamFooter(side.rows) : []}
              empty={`No box score for ${side.name} yet`}
            />
          );
        }}
      </TeamSwitch>
      {unattributed.length > 0 && (
        <p className="mt-2 text-sm text-pr-text-dim">
          {unattributed.length} player{unattributed.length === 1 ? "" : "s"} could not be matched to{" "}
          {teamName(awayTeam)} or {teamName(homeTeam)} and {unattributed.length === 1 ? "is" : "are"} left out
          rather than guessed onto a side.
        </p>
      )}
    </div>
  );
}
