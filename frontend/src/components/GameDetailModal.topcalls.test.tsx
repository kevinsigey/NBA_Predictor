import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    // `GameDetailModal` fetches its signal rows separately, so a mock that omits
    // it leaves it `undefined` and the modal's effect throws on open -- 41 failures
    // across four files when this row landed.
    //
    // Default: a valid EMPTY list -- spec §2's "no data, no row". A bare `vi.fn()`
    // returns `undefined` and the modal calls `.then` on it.
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "", signals: [] })),
    // The post-game review's own feed: null is its absence (no counted props, or unreadable) -- the state these tests want unless they say otherwise.
    getGamePropGrading: vi.fn(async () => null),
    getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(),
    getGameOutPlayers: vi.fn(),
    getGameDoubtfulPlayers: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn().mockResolvedValue([]),
    getTrackRecord: vi.fn().mockResolvedValue([]),
    explainGame: vi.fn(),
  },
}));

afterEach(() => vi.restoreAllMocks());

beforeEach(() => {
  vi.mocked(api.getHubPlayers).mockResolvedValue([]);
  vi.mocked(api.getTrackRecord).mockResolvedValue([]);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([]);
  vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([]);
  vi.mocked(api.getGameDetail).mockResolvedValue({
    game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
    prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
    completed: false, home_pts: null, away_pts: null,
    markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
  } as never);
});

/** The modal renders on its own `gameId`; there is no trigger to click. */
const open = async () => {
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  return waitFor(() => expect(api.getGameDetail).toHaveBeenCalledWith("g1"));
};

it("fetches the out route as its own call, not a wrapper on the props feed", async () => {
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "1", player_name: "Scorer", stat: "points", predicted_value: 31, actual_value: null, mae: 4.2 },
  ]);
  await open();
  await waitFor(() => expect(api.getGameOutPlayers).toHaveBeenCalledWith("g1"));
});

it("renders the model's top calls inside the game modal", async () => {
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "1", player_name: "Scorer", stat: "points", predicted_value: 31, actual_value: null, mae: 4.2 },
  ]);
  await open();
  await waitFor(() => expect(screen.getByTestId("picks-list")).toBeInTheDocument());
  expect(screen.getByTestId("picks-title")).toHaveTextContent("Model's top calls");
  // The row is the player and the figure. The ± the API sent is not on it.
  expect(screen.getByText("31.0")).toBeInTheDocument();
  expect(document.body.textContent).not.toContain("±");
});

it("still renders the box score: this adds a block, it does not replace one", async () => {
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "1", player_name: "Scorer", stat: "points", predicted_value: 31, actual_value: null, mae: 4.2 },
  ]);
  await open();
  await waitFor(() => expect(screen.getByText("Projected box score")).toBeInTheDocument());
});

it("removes an out player from the modal's ranking and names him once below", async () => {
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "1", player_name: "Healthy", stat: "points", predicted_value: 31, actual_value: null, mae: 4.2 },
    // The gate normally withholds these; seeding them proves the page holds the
    // line even if the feed is wrong.
    { player_id: "2", player_name: "Absent", stat: "points", predicted_value: 35, actual_value: null, mae: 3.1 },
  ]);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([
    { player_id: "2", player_name: "Absent", team: "BOS", status: "Out", source: "ESPN injury report", dated: "2026-11-01 08:00" },
  ]);
  await open();
  await waitFor(() => expect(screen.getByTestId("availability-out")).toBeInTheDocument());
  for (const list of screen.getAllByTestId("picks-category")) {
    expect(within(list).queryByText(/Absent/)).not.toBeInTheDocument();
  }
  // Named ONCE: in the availability section, not a second time under the
  // ranking.
  expect(screen.queryByTestId("picks-out")).not.toBeInTheDocument();
});

it("shows no ranking at all when the out feed cannot be read", async () => {
  // An unreadable availability feed means "not a proven healthy player", which is
  // not the same claim as "ranked". The block stays out of the page rather than
  // showing a ranking nobody can vouch for.
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "1", player_name: "Scorer", stat: "points", predicted_value: 31, actual_value: null, mae: 4.2 },
  ]);
  vi.mocked(api.getGameOutPlayers).mockRejectedValue(new Error("503"));
  await open();
  await waitFor(() => expect(screen.getByTestId("top-calls-unavailable")).toBeInTheDocument());
  expect(screen.queryByTestId("picks-list")).not.toBeInTheDocument();
});
