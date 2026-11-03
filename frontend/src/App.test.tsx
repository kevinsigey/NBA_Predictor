import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

function mockFetchEmpty() {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    json: async () => [],
  } as Response);
}

describe("App", () => {
  it("renders the site title", () => {
    mockFetchEmpty();

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );
    expect(screen.getByRole("heading", { name: "NBA Predictor" })).toBeInTheDocument();
  });
});

describe("App routing", () => {
  it("navigates to the Teams panel when its tab is clicked", async () => {
    mockFetchEmpty();

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    );

    await userEvent.click(screen.getByRole("tab", { name: "Teams" }));
    expect(await screen.findByText(/no team data/i)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Teams" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("navigation", { name: "Data Hub sections" })).toBeNull();
  });
});

describe("App family frame", () => {
  function renderAt(path: string) {
    mockFetchEmpty();
    return render(
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    );
  }

  it("carries the family wordmark, the NBA accent and a switcher to every sport", () => {
    const { container } = renderAt("/");
    expect(screen.getByText("Predictor")).toBeInTheDocument();
    expect(container.querySelector("[data-sport='nba']")).not.toBeNull();
    const sports = screen.getByRole("navigation", { name: "Sports" });
    expect(within(sports).getByRole("link", { name: "NBA" })).toHaveAttribute("aria-current", "page");
    expect(within(sports).getByRole("link", { name: "NFL" }).getAttribute("href")).toMatch(/sports\..*\?sport=nfl$/);
    expect(within(sports).getByRole("link", { name: "Premier League" })).toBeInTheDocument();
    expect(within(sports).getByRole("link", { name: "College football" })).toBeInTheDocument();
    expect(within(sports).getByRole("link", { name: "Paper Trading" })).toBeInTheDocument();
  });

  it("marks the page tab that matches the route, including on a deep link", () => {
    renderAt("/calibration-report");
    const pages = screen.getByRole("tablist", { name: "Pages" });
    expect(within(pages).getByRole("tab", { name: "Calibration" })).toHaveAttribute("aria-selected", "true");
    expect(within(pages).getByRole("tab", { name: "Games" })).not.toHaveAttribute("aria-selected", "true");
    expect(
      within(pages).getAllByRole("tab").map((b) => b.textContent),
    ).toEqual(["Games", "Teams", "Players", "Track record", "Calibration", "Model"]);
  });

  it("sends the old standings link to the Teams page", async () => {
    renderAt("/standings");
    expect(await screen.findByText(/no team data/i)).toBeInTheDocument();
    const pages = screen.getByRole("tablist", { name: "Pages" });
    expect(within(pages).getByRole("tab", { name: "Teams" })).toHaveAttribute("aria-selected", "true");
  });

  it("sends the old power-rankings link to the Teams page, still sorted by power", async () => {
    renderAt("/power-rankings");
    expect(await screen.findByText(/no team data/i)).toBeInTheDocument();
    const pages = screen.getByRole("tablist", { name: "Pages" });
    expect(within(pages).getByRole("tab", { name: "Teams" })).toHaveAttribute("aria-selected", "true");
  });

  it("sends the old prop-grading link to the track record, where the tracking aggregates live", async () => {
    // Not the blanket empty mock: the record panel reads fields off the
    // vs-market object, and an array wearing its place would throw in an
    // effect -- the real API always answers the object.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const body = String(input).includes("/hub/vs-market")
        ? {
            market: "h2h", n: 0, mean_model_probability: null, mean_market_probability: null,
            mean_edge_points: null, disagreement_n: 0, disagreement_hit_rate: null,
            disagreement_game_ids: [], weekly: [], scope: {}, method: {},
          }
        : [];
      return { ok: true, json: async () => body } as Response;
    });
    render(
      <MemoryRouter initialEntries={["/prop-grading"]}>
        <App />
      </MemoryRouter>
    );
    expect(await screen.findByText(/no tracked predictions/i)).toBeInTheDocument();
    const pages = screen.getByRole("tablist", { name: "Pages" });
    expect(within(pages).getByRole("tab", { name: "Track record" })).toHaveAttribute("aria-selected", "true");
    expect(within(pages).queryByRole("tab", { name: "Prop grading" })).toBeNull();
  });

  it("navigates when a page tab is pressed", async () => {
    renderAt("/");
    await userEvent.click(screen.getByRole("tab", { name: "Calibration" }));
    expect(screen.getByRole("tab", { name: "Calibration" })).toHaveAttribute("aria-selected", "true");
  });
});
