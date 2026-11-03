import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import DataHubPage from "./DataHubPage";

vi.mock("../api/client", () => ({
  api: {
    getHubTeams: vi.fn().mockResolvedValue([]),
    getHubTeamOverview: vi.fn().mockResolvedValue({
      as_of: "2026-11-01", n_teams: 30, caption: "", metrics: [], ratings: {}, form: {}, last_games: {}, next_game: {},
    }),
    getHubPlayers: vi.fn().mockResolvedValue([]),
    getHubRankings: vi.fn().mockResolvedValue([]),
    getHubStandings: vi.fn().mockResolvedValue([]),
    getTrackRecord: vi.fn().mockResolvedValue([]),
    getTrackRecordPicks: vi.fn().mockResolvedValue({ total: 0, limit: 200, offset: 0, picks: [] }),
    getPropGrading: vi.fn().mockResolvedValue({
      bands: null, counting: "", confidence_basis: "", grade_basis: "",
      latest_day: null, most_confident: [], biggest_surprises: [], n_with_baseline: 0, aggregates: null,
    }),
    getSnapshotMeta: vi.fn().mockResolvedValue({ generated_at: null, source: "live" }),
    getVsMarket: vi.fn().mockResolvedValue({
      market: "h2h",
      n: 0,
      mean_model_probability: null,
      mean_market_probability: null,
      mean_edge_points: null,
      disagreement_n: 0,
      disagreement_hit_rate: null,
      disagreement_game_ids: [],
      weekly: [],
      scope: {
        population: "finished games with a pre-tip moneyline price",
        weekly_from: null,
        weekly_through: null,
        n_games_total: 0,
        n_games_in_weekly: 0,
        n_games_outside_weekly: 0,
      },
      method: {},
    }),
  },
}));

describe("DataHubPage", () => {
  it("shows the Team Hub panel by default", async () => {
    render(<MemoryRouter><DataHubPage /></MemoryRouter>);
    await waitFor(() => expect(screen.getByText(/no team data/i)).toBeInTheDocument());
  });

  it("shows the panel its route names, with no second tab row of its own", async () => {
    const { rerender } = render(<MemoryRouter><DataHubPage initialTab="player-hub" /></MemoryRouter>);
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
    expect(screen.queryByRole("navigation")).toBeNull();
    // Same instance, new route: the panel follows the prop instead of stale local state.
    rerender(<MemoryRouter><DataHubPage initialTab="track-record" /></MemoryRouter>);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});
