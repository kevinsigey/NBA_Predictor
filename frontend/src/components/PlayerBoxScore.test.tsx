import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import PlayerBoxScore from "./PlayerBoxScore";
import {
  STAT_KEYS,
  attachHubRows,
  hasActuals,
  orderPlayers,
  pivotProps,
  splitByTeam,
  sumStat,
} from "../lib/boxScore";
import type { PlayerHubRow, PlayerProp } from "../api/client";

const STATS = ["points", "rebounds", "assists", "threes"] as const;

function propsFor(playerId: string, name: string, values: number[], actual: number[] | null = null): PlayerProp[] {
  return STATS.map((stat, i) => ({
    player_id: playerId,
    player_name: name,
    stat,
    predicted_value: values[i],
    actual_value: actual ? actual[i] : null,
    rebuilt: false,
  }));
}

function hubRow(playerId: string, name: string, team: string, position: string): PlayerHubRow {
  return {
    player_id: playerId, player_name: name, team, position,
    rating: 0, live_form_rating: 0, points_per_game: 0, rebounds_per_game: 0,
    assists_per_game: 0, fg_pct: 0, three_pt_pct: 0, ft_pct: 0, usage_rate: 0, minutes_per_game: 0,
  };
}

/** Two teams of three, the shape the live feed actually has. */
const AWAY = [
  { id: "a1", name: "A One", vals: [20, 5, 6, 2] },
  { id: "a2", name: "A Two", vals: [10, 8, 3, 1] },
  { id: "a3", name: "A Three", vals: [5, 2, 1, 0] },
] as const;
const HOME = [
  { id: "h1", name: "H One", vals: [30, 10, 12, 4] },
  { id: "h2", name: "H Two", vals: [15, 3, 2, 1] },
  { id: "h3", name: "H Three", vals: [8, 1, 1, 0] },
] as const;

const allProps: PlayerProp[] = [
  ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals])),
  ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals])),
];
const allHub: PlayerHubRow[] = [
  ...AWAY.map((p, i) => hubRow(p.id, p.name, "MIA", i === 2 ? "C" : "G")),
  ...HOME.map((p, i) => hubRow(p.id, p.name, "BOS", i === 2 ? "C" : "F")),
];

describe("boxScore pivot", () => {
  it("collapses one row per player per stat into one row per player", () => {
    expect(allProps).toHaveLength(24);
    const rows = pivotProps(allProps);
    expect(rows).toHaveLength(6);
    expect(rows.find((r) => r.player_id === "a1")?.predicted).toEqual({
      points: 20, rebounds: 5, assists: 6, threes: 2,
    });
  });

  it("leaves an unplayed stat absent rather than defaulting it to 0", () => {
    const rows = pivotProps(propsFor("x", "X", [10, 0, 0, 0]));
    expect(rows[0].actual).toEqual({});
    expect(hasActuals(rows)).toBe(false);
  });

  it("ignores a stat key it does not have a column for", () => {
    const rows = pivotProps([
      ...propsFor("x", "X", [10, 5, 5, 1]),
      { player_id: "x", player_name: "X", stat: "blocks", predicted_value: 3, actual_value: null, rebuilt: false },
    ]);
    // The four known stats survive; "blocks" is dropped rather than shown as an
    // unlabelled column.
    expect(rows[0].predicted).toEqual({ points: 10, rebounds: 5, assists: 5, threes: 1 });
  });

  it("produces no row at all for a player who only has unknown stats", () => {
    const rows = pivotProps([
      { player_id: "x", player_name: "X", stat: "blocks", predicted_value: 3, actual_value: null, rebuilt: false },
    ]);
    expect(rows).toEqual([]);
  });

  it("keeps only the first prop for a repeated player and stat", () => {
    const rows = pivotProps([
      ...propsFor("x", "X", [10, 1, 1, 1]),
      { player_id: "x", player_name: "X", stat: "points", predicted_value: 99, actual_value: null, rebuilt: false },
    ]);
    expect(rows[0].predicted.points).toBe(10);
  });
});

describe("boxScore team split", () => {
  it("puts each player on the side their team says, not by array order", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { away, home, unattributed } = splitByTeam(rows, "BOS", "MIA");
    expect(away.map((r) => r.player_id)).toEqual(["a1", "a2", "a3"]);
    expect(home.map((r) => r.player_id)).toEqual(["h1", "h2", "h3"]);
    expect(unattributed).toHaveLength(0);
  });

  it("places an upcoming-game player by the team on his projection when the hub feed has not seen him play", () => {
    const props = propsFor("new1", "New Guy", [10, 4, 3, 1]).map((p) => ({ ...p, team: "BOS" }));
    const rows = attachHubRows(pivotProps(props), []);
    const { home, unattributed } = splitByTeam(rows, "BOS", "MIA");
    expect(home.map((r) => r.player_id)).toEqual(["new1"]);
    expect(unattributed).toHaveLength(0);
  });

  it("returns a player it cannot place instead of guessing a side", () => {
    const hub = [...allHub.filter((p) => p.player_id !== "a2")];
    const rows = attachHubRows(pivotProps(allProps), hub);
    const { away, home, unattributed } = splitByTeam(rows, "BOS", "MIA");
    expect(away.map((r) => r.player_id)).not.toContain("a2");
    expect(home.map((r) => r.player_id)).not.toContain("a2");
    expect(unattributed.map((r) => r.player_id)).toEqual(["a2"]);
  });

  it("orders guards, then wings, then centres, and the biggest role first within a position", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { home } = splitByTeam(rows, "BOS", "MIA");
    // h1/h2 are F, h3 is C -- so the forwards lead and the centre is last.
    expect(home.map((r) => r.position)).toEqual(["F", "F", "C"]);
    expect(home[0].player_id).toBe("h1"); // 30 points beats 15
  });

  it("sorts an unrecognised position last rather than first", () => {
    const hub = allHub.map((p) => (p.player_id === "h3" ? { ...p, position: "" } : p));
    const rows = attachHubRows(pivotProps(allProps), hub);
    expect(splitByTeam(rows, "BOS", "MIA").home.map((r) => r.player_id)).toEqual(["h1", "h2", "h3"]);
  });

  it("does not mutate its input", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const before = rows.map((r) => r.player_id);
    orderPlayers(rows);
    expect(rows.map((r) => r.player_id)).toEqual(before);
  });
});

describe("boxScore intervals", () => {
  it("carries a prop's 80% range through the pivot, per stat", () => {
    const props = propsFor("x", "X Man", [20, 5, 6, 2]).map((p) =>
      p.stat === "points" ? { ...p, interval_low: 12.5, interval_high: 27.5 } : p,
    );
    const [row] = pivotProps(props);
    expect(row.intervals.points).toEqual({ low: 12.5, high: 27.5 });
    expect(row.intervals.rebounds).toBeUndefined();
  });

  it("leaves the range off a prop the API served without one", () => {
    const [row] = pivotProps(propsFor("x", "X Man", [20, 5, 6, 2]));
    expect(row.intervals).toEqual({});
  });
});

describe("boxScore totals", () => {
  it("sums each side's own column", () => {
    const rows = attachHubRows(pivotProps(allProps), allHub);
    const { away, home } = splitByTeam(rows, "BOS", "MIA");
    expect(sumStat(away, "points", "predicted")).toBe(35);
    expect(sumStat(home, "points", "predicted")).toBe(53);
    expect(sumStat(away, "rebounds", "predicted")).toBe(15);
  });
});

describe("PlayerBoxScore", () => {
  function renderBox(overrides: { props?: PlayerProp[]; hub?: PlayerHubRow[]; bands?: Record<string, number> } = {}) {
    return render(
      <PlayerBoxScore
        playerProps={overrides.props ?? allProps}
        hubPlayers={overrides.hub ?? allHub}
        homeTeam="BOS"
        awayTeam="MIA"
        bands={overrides.bands}
      />,
    );
  }
  const pick = (name: string) => fireEvent.click(screen.getByRole("button", { name }));

  it("shows one team at a time: home by default, away after the switch", () => {
    renderBox();
    expect(screen.getAllByRole("table")).toHaveLength(1);
    expect(screen.getAllByTestId("box-score-row")).toHaveLength(3);
    expect(screen.getByRole("rowheader", { name: "H One" })).toBeTruthy();
    expect(screen.queryByRole("rowheader", { name: "A One" })).toBeNull();
    const group = screen.getByRole("group", { name: "Box score team" });
    const [home, away] = within(group).getAllByRole("button");
    expect([home.textContent, away.textContent]).toEqual(["BOS", "MIA"]);
    expect(home.getAttribute("aria-pressed")).toBe("true");

    pick("MIA");
    expect(screen.getByRole("rowheader", { name: "A One" })).toBeTruthy();
    expect(screen.queryByRole("rowheader", { name: "H One" })).toBeNull();
  });

  it("pins the full player name in a sticky first column and scrolls the stats", () => {
    const props = propsFor("g", "Giannis Antetokounmpo", [31, 11, 5, 1]);
    renderBox({ props, hub: [hubRow("g", "Giannis Antetokounmpo", "BOS", "F")] });
    const name = screen.getByRole("rowheader", { name: "Giannis Antetokounmpo" });
    expect(name.textContent).toBe("Giannis Antetokounmpo");
    expect(name.className).toMatch(/sticky/);
    const scroller = screen.getByRole("region", { name: /scroll sideways/i });
    expect(scroller.contains(name)).toBe(true);
    for (const label of ["Pts", "Reb", "Ast", "3PM"]) {
      expect(within(scroller).getByRole("columnheader", { name: label })).toBeTruthy();
    }
  });

  it("shows the predicted value, which is the whole point of the page", () => {
    renderBox();
    const row = screen.getAllByTestId("box-score-row")[0];
    expect(within(row).getByRole("rowheader", { name: "H One" })).toBeTruthy();
    expect(within(row).getByText("30.0")).toBeTruthy();
  });

  it("totals the selected team from its own players only", () => {
    // The ACTUAL total is the fact that is summed. The projected total is
    // gone: summing every rostered player's projection is not the team's
    // projection -- it adds up the players who never get minutes.
    const played: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    renderBox({ props: played });
    // BOS 28+28+28 = 84 points. The other team's players never leak in.
    expect(within(screen.getByTestId("box-score-actual-totals")).getByText("84")).toBeTruthy();
    pick("MIA");
    expect(within(screen.getByTestId("box-score-actual-totals")).getByText("66")).toBeTruthy();
  });

  it("has no projected total: the sum of every rostered player is not the team's projection", () => {
    renderBox();
    expect(screen.queryByTestId("box-score-totals")).toBeNull();
    expect(screen.queryByTestId("box-score-actual-totals")).toBeNull();
  });

  it("says so when a player cannot be matched to either team", () => {
    renderBox({ hub: allHub.filter((p) => p.player_id !== "a2") });
    expect(screen.getByText(/could not be matched/i)).toBeTruthy();
  });

  it("shows the actual under the prediction once the game is played", () => {
    const played: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    renderBox({ props: played });
    pick("MIA");
    const row = screen.getAllByTestId("box-score-row")[0];
    expect(within(row).getByText("20.0")).toBeTruthy();
    // The delta is its own element now, because it carries the colour: green
    // when the player beat the projection by more than the model's own error.
    expect(within(row).getByText("+2.0")).toBeTruthy();
    expect(screen.getByTestId("box-score-actual-totals")).toBeTruthy();
  });

  it("leaves the delta plain when the miss is inside the model's own error", () => {
    const played: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    // A points band of 4.6: a miss of 2 points is inside it.
    renderBox({ props: played, bands: { points: 4.6 } });
    pick("MIA");
    const row = screen.getAllByTestId("box-score-row")[0];
    const inside = within(row).getByText("+2.0");
    expect(inside.className).not.toMatch(/text-pr-(win|loss)/);
  });

  it("colours the delta when the player beat the projection by more than the error", () => {
    const played: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    const bigMiss: PlayerProp[] = played.map((p) =>
      p.stat === "points" ? { ...p, actual_value: (p.actual_value ?? 0) + 9 } : p,
    );
    renderBox({ props: bigMiss, bands: { points: 4.6 } });
    pick("MIA");
    const row = screen.getAllByTestId("box-score-row")[0];
    const outside = within(row).getByText(/\+11\.0/);
    expect(outside.className).toMatch(/text-pr-win/);
  });

  it("scores a rebuilt projection never: no delta, even when it is wrong", () => {
    const rebuilt: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3]).map((prop) => ({ ...prop, rebuilt: true }))),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [28, 9, 11, 5])),
    ];
    renderBox({ props: rebuilt });
    const norm = (el: HTMLElement) => within(el).getAllByRole("cell").map((c) => c.textContent?.replace(/\s+/g, ""));
    // Home is not rebuilt: its delta is the control proving the absence below is the rule.
    expect(norm(screen.getAllByTestId("box-score-row")[0])[0]).toBe("30.028(-2.0)");
    pick("MIA");
    expect(norm(screen.getAllByTestId("box-score-row")[0])).toEqual(["20.022", "5.04", "6.05", "2.03"]);
    expect(screen.getAllByText("Rebuilt").length).toBeGreaterThan(0);
  });

  it("a stat with no projection is an em dash", () => {
    const props = propsFor("h1", "H One", [30, 10, 12, 4]).filter((p) => p.stat !== "assists");
    renderBox({ props, hub: [hubRow("h1", "H One", "BOS", "F")] });
    expect(within(screen.getAllByTestId("box-score-row")[0]).getAllByRole("cell").map((c) => c.textContent)).toEqual(["30.0", "10.0", "—", "4.0"]);
  });

  it("a team with no players gets a one-line message, not an empty table", () => {
    renderBox({ hub: allHub.filter((p) => p.team !== "MIA") });
    pick("MIA");
    expect(screen.getByText("No box score for Heat yet")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("opens on the away team when only the away team has players", () => {
    renderBox({ hub: allHub.filter((p) => p.team !== "BOS") });
    expect(screen.getByRole("rowheader", { name: "A One" })).toBeTruthy();
  });

  it("renders no table when there are no props, and says so", () => {
    const { container } = renderBox({ props: [] });
    expect(container.querySelector("table")).toBeNull();
    expect(screen.getByTestId("player-box-score")).toBeInTheDocument();
  });

  it("covers exactly the four stats the live feed carries", () => {
    const withUnknown = [
      ...propsFor("x", "X", [10, 5, 5, 1]),
      { player_id: "x", player_name: "X", stat: "fantasy_points", predicted_value: 99, actual_value: null, rebuilt: false },
    ];
    const rows = pivotProps(withUnknown);
    expect(Object.keys(rows[0].predicted).sort()).toEqual(["assists", "points", "rebounds", "threes"]);
    expect(rows[0].predicted).not.toHaveProperty("fantasy_points");
    expect(STAT_KEYS).toHaveLength(4);
  });

  it("says so when the team feed is empty, rather than rendering nothing", () => {
    renderBox({ hub: [] });
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByText(/team feed did not load/i)).toBeInTheDocument();
  });

  it("keeps the actual-totals row well formed when only one side has actuals", () => {
    const halfPlayed: PlayerProp[] = [
      ...AWAY.flatMap((p) => propsFor(p.id, p.name, [...p.vals], [22, 4, 5, 3])),
      ...HOME.flatMap((p) => propsFor(p.id, p.name, [...p.vals])),
    ];
    renderBox({ props: halfPlayed });
    pick("MIA");
    expect(within(screen.getByTestId("box-score-actual-totals")).getAllByRole("cell")).toHaveLength(4);
    pick("BOS");
    // Home has no actuals at all, so no actual row is drawn for it.
    expect(screen.queryByTestId("box-score-actual-totals")).toBeNull();
  });

  it("shows the 80% range under a projection before the game, and not after it", () => {
    const withRange = propsFor("h1", "H One", [30, 10, 12, 4]).map((p) =>
      p.stat === "points" ? { ...p, interval_low: 21.5, interval_high: 38.5 } : p,
    );
    const props = [...withRange, ...HOME.slice(1).flatMap((p) => propsFor(p.id, p.name, [...p.vals]))];
    const first = renderBox({ props });
    const rowEl = screen.getAllByTestId("box-score-row")[0];
    expect(rowEl.textContent).toContain("30.0");
    expect(rowEl.textContent).toContain("21.5\u201338.5");
    first.unmount();

    // Once the game is played the cell is judged against the actual; the
    // pre-game range line goes away (the review carries the verdict).
    const played = withRange.map((p) => ({ ...p, actual_value: 33 }));
    renderBox({ props: [...played, ...HOME.slice(1).flatMap((p) => propsFor(p.id, p.name, [...p.vals]))] });
    const playedRow = screen.getAllByTestId("box-score-row")[0];
    expect(playedRow.textContent).not.toContain("21.5\u201338.5");
  });
});
