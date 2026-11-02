import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import PropGradingPanel from "./PropGradingPanel";
import { api, type PropGrading } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/client")>();
  return { ...original, api: { ...original.api, getPropGrading: vi.fn() } };
});

const row = (over: Partial<PropGrading["most_confident"][number]>): PropGrading["most_confident"][number] => ({
  game_id: "g2",
  game_date: "2026-11-01",
  home_team: "BOS",
  away_team: "MIA",
  player_id: "A",
  player_name: "Player A",
  stat: "points",
  predicted: 30,
  actual: 31,
  delta: 1,
  grade: "On the number",
  made_before_tip: true,
  baseline: 20,
  departure: 10,
  ...over,
});

const PAYLOAD: PropGrading = {
  bands: { points: 4.0 },
  counting: "Graded from the recorded predictions.",
  confidence_basis: "Ranked by departure from his rolling average.",
  grade_basis: "Grades measure the miss against the model's own error.",
  latest_day: {
    date: "2026-11-01",
    n_graded: 2,
    n_pre_tip: 1,
    games: [
      {
        game_id: "g2",
        game_date: "2026-11-01",
        home_team: "BOS",
        away_team: "MIA",
        rows: [
          row({}),
          row({
            player_id: "B",
            player_name: "Player B",
            predicted: 10,
            actual: 4,
            delta: -6,
            grade: "Off",
            made_before_tip: false,
            baseline: null,
            departure: null,
          }),
        ],
      },
    ],
  },
  most_confident: [row({})],
  biggest_surprises: [
    row({
      player_id: "B",
      player_name: "Player B",
      predicted: 10,
      actual: 4,
      delta: -6,
      grade: "Off",
      made_before_tip: false,
      baseline: null,
      departure: null,
    }),
  ],
  n_with_baseline: 1,
  served_stale_seconds: null,
};

describe("PropGradingPanel", () => {
  beforeEach(() => {
    vi.mocked(api.getPropGrading).mockResolvedValue(PAYLOAD);
  });

  it("leads with the day's two counts side by side", async () => {
    render(<PropGradingPanel />);
    const heading = await screen.findByText(/Prop grading/);
    const header = heading.closest("header")!;
    expect(header.textContent).toContain("2 player lines graded");
    expect(header.textContent).toContain("1 made before tip-off");
  });

  it("shows the delta chip direction as the player against the number", async () => {
    render(<PropGradingPanel />);
    expect((await screen.findAllByText("▲ +1.0 vs proj")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("▼ −6.0 vs proj").length).toBeGreaterThan(0);
  });

  it("prints the projection line in the Proj X to Y actual form, with the grade", async () => {
    render(<PropGradingPanel />);
    expect((await screen.findAllByText(/Proj 30\.0 → 31 actual/)).length).toBeGreaterThan(0);
    expect(screen.getAllByText("On the number").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Off").length).toBeGreaterThan(0);
  });

  it("labels a row that cannot prove pre-tip instead of dropping it", async () => {
    render(<PropGradingPanel />);
    expect((await screen.findAllByText("Made after tip-off")).length).toBeGreaterThan(0);
  });

  it("says how much of the day the most-confident ranking could cover", async () => {
    render(<PropGradingPanel />);
    const section = (await screen.findByText("Most confident")).closest("section")!;
    expect(within(section).getByText(/Ranked over/)).toBeTruthy();
    expect(section.textContent).toContain("1");
    expect(section.textContent).toContain("2");
  });

  it("draws an empty state when nothing has been graded yet", async () => {
    vi.mocked(api.getPropGrading).mockResolvedValue({ ...PAYLOAD, latest_day: null });
    render(<PropGradingPanel />);
    expect(await screen.findByText(/No graded player projections yet/)).toBeTruthy();
  });
});
