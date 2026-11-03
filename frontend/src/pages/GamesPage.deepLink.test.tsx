import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import GamesPage from "./GamesPage";
import { api, type Game, type GameDetail } from "../api/client";

// The deep link arrives on the URL, so what is under test is the page's
// behaviour on load — not a click. Everything the page and the detail modal
// ask for is stubbed here; nothing in this file reaches the network.
vi.mock("../api/client", () => ({
  api: {
    // `GameDetailModal` fetches its signal rows separately; see
    // GameDetailModal.test.tsx. A valid EMPTY list -- spec §2's "no data, no row" --
    // because a bare `vi.fn()` returns `undefined` and the modal calls `.then`.
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "", signals: [] })),
    // The post-game review's own feed: null is its absence (no counted props, or unreadable).
    getGamePropGrading: vi.fn(async () => null),
    getSeasonFirstWeek: vi.fn(), getGamesWeek: vi.fn(), getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(), getHubPlayers: vi.fn().mockResolvedValue([]),
    // The gate's sibling route; see GameDetailModal.test.tsx.
    getGameOutPlayers: vi.fn().mockResolvedValue([]),
    getGameDoubtfulPlayers: vi.fn().mockResolvedValue([]),
    // The detail modal's block reads this for its record strip. Not the
    // subject here, and empty so the strip stays out of these assertions.
    getTrackRecord: vi.fn().mockResolvedValue([]),
  },
}));

const g1: Game = {
  game_id: "g1", game_date: "2026-02-16", tip_off: "2026-02-16T19:30:00Z",
  home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
};
const g2: Game = {
  game_id: "g2", game_date: "2026-02-18", tip_off: "2026-02-18T22:00:00Z",
  home_team: "LAL", away_team: "GSW",
  prediction: null, completed: false, home_pts: null, away_pts: null,
};

const detailOf = (game: Game): GameDetail => ({
  ...game, markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
});

const setUrl = (path: string) => window.history.replaceState({}, "", path);

describe("a game identifier on the URL", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setUrl("/");
    vi.mocked(api.getSeasonFirstWeek).mockResolvedValue({ first_week_start: "2026-02-10" });
    vi.mocked(api.getGamesWeek).mockResolvedValue([g1, g2]);
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);
    vi.mocked(api.getGameOutPlayers).mockResolvedValue([]);
    vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([]);
    vi.mocked(api.getGameDetail).mockImplementation(async (id: string) =>
      detailOf(id === "g1" ? g1 : g2));
  });
  afterEach(() => setUrl("/"));

  it("opens that game's detail", async () => {
    setUrl("/?game=g1");
    render(<GamesPage />);

    const dialog = await screen.findByRole("dialog");
    // The detail is the one the identifier names, not merely some detail:
    // g1 is Miami at Boston, g2 is the other game in the same week.
    await waitFor(() => expect(dialog).toHaveTextContent("Heat at Celtics"));
    expect(api.getGameDetail).toHaveBeenCalledWith("g1");
    // The normal list is still the page underneath it.
    expect(screen.getByTestId("game-card-g1")).toBeInTheDocument();
    expect(screen.getByTestId("game-card-g2")).toBeInTheDocument();
  });

  it("falls back to the normal list for an unknown identifier, with no error state", async () => {
    setUrl("/?game=not-a-game");
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByTestId("game-card-g1")).toBeInTheDocument());
    expect(screen.getByTestId("game-card-g2")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(api.getGameDetail).not.toHaveBeenCalled();
  });

  it("shows the normal list when there is no identifier at all", async () => {
    setUrl("/");
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByTestId("game-card-g1")).toBeInTheDocument());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
