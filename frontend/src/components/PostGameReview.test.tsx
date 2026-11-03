import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import PostGameReview from "./PostGameReview";
import type { PropGameGrading, PropGradingRow } from "../api/client";

/**
 * The review renders the backend's grading of the game's counted picks:
 * every counted row, in the match's context. These tests pin the
 * completeness (a Close line is a row, not just a count), the labelling
 * (a post-tip row is shown and said to be post-tip, never dropped), and
 * the absence (no counted props, no section).
 */

const row = (over: Partial<PropGradingRow> & { player_id: string; stat: string }): PropGradingRow => ({
  game_id: "g2",
  game_date: "2026-01-05",
  home_team: "BOS",
  away_team: "MIA",
  player_name: `Player ${over.player_id}`,
  predicted: 20,
  actual: 20,
  delta: 0,
  grade: "On the number",
  made_before_tip: true,
  baseline: null,
  departure: null,
  ...over,
});

const grading = (rows: PropGradingRow[], over: Partial<PropGameGrading> = {}): PropGameGrading => ({
  game_id: "g2",
  game_date: "2026-01-05",
  home_team: "BOS",
  away_team: "MIA",
  n_graded: rows.length,
  n_pre_tip: rows.filter((r) => r.made_before_tip).length,
  rows,
  biggest_surprises: [...rows].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)),
  bands: { points: 4.6, rebounds: 1.9, assists: 1.3, threes: 0.9 },
  counting: "counting text",
  grade_basis: "grade text",
  ...over,
});

describe("PostGameReview", () => {
  it("shows every counted line, not only the extremes", () => {
    render(
      <PostGameReview
        grading={grading([
          row({ player_id: "1", player_name: "Jayson Tatum", stat: "points", predicted: 22.4, actual: 31, delta: 8.6, grade: "Off" }),
          row({ player_id: "2", player_name: "Jalen Brunson", stat: "points", predicted: 24.1, actual: 18, delta: -6.1, grade: "Off" }),
          row({ player_id: "3", player_name: "Mikal Bridges", stat: "points", predicted: 15.0, actual: 16.2, delta: 1.2, grade: "On the number" }),
        ])}
      />,
    );

    const review = screen.getByTestId("post-game-review");
    // The middle line -- neither a beating nor a miss -- is a row too.
    expect(within(review).getByText("Mikal Bridges")).toBeTruthy();
    expect(review.textContent).toContain("Proj 15.0 \u2192 16.2 actual");
    expect(review.textContent).toContain("3 player lines graded");
    expect(review.textContent).toContain("3 made before tip-off");
  });

  it("follows the player with the delta chip, in both directions", () => {
    render(
      <PostGameReview
        grading={grading([
          row({ player_id: "1", player_name: "Jayson Tatum", stat: "points", predicted: 22.4, actual: 31, delta: 8.6, grade: "Off" }),
          row({ player_id: "2", player_name: "Jalen Brunson", stat: "points", predicted: 24.1, actual: 18, delta: -6.1, grade: "Off" }),
        ])}
      />,
    );

    expect(screen.getAllByText("\u25b2 +8.6 vs proj").length).toBeGreaterThan(0);
    expect(screen.getAllByText("\u25bc \u22126.1 vs proj").length).toBeGreaterThan(0);
  });

  it("counts the lines that landed within the model's own error", () => {
    render(
      <PostGameReview
        grading={grading([
          row({ player_id: "1", stat: "points", predicted: 20, actual: 22, delta: 2, grade: "On the number" }),
          row({ player_id: "2", stat: "points", predicted: 20, actual: 30, delta: 10, grade: "Way off" }),
        ])}
      />,
    );

    const withinRange = screen.getByTestId("review-on-the-number");
    expect(withinRange.textContent).toContain("1 of 2");
    expect(withinRange.textContent).toContain("player lines within range");
  });

  it("leads with the game's biggest surprises, and only real ones", () => {
    render(
      <PostGameReview
        grading={grading([
          row({ player_id: "1", player_name: "Jayson Tatum", stat: "points", predicted: 22.4, actual: 31, delta: 8.6, grade: "Off" }),
          row({ player_id: "2", player_name: "Mikal Bridges", stat: "points", predicted: 15.0, actual: 16.2, delta: 1.2, grade: "On the number" }),
        ])}
      />,
    );

    const surprises = screen.getByTestId("review-surprises");
    expect(within(surprises).getByText("Jayson Tatum")).toBeTruthy();
    // A line inside the model's error is not a surprise, however it is listed.
    expect(within(surprises).queryByText("Mikal Bridges")).toBeNull();
  });

  it("labels a row made after tip-off and still grades it", () => {
    render(
      <PostGameReview
        grading={grading(
          [
            row({ player_id: "1", player_name: "Jayson Tatum", stat: "points", predicted: 22.4, actual: 31, delta: 8.6, grade: "Off" }),
            row({ player_id: "2", player_name: "Late Line", stat: "points", predicted: 10, actual: 19, delta: 9, grade: "Way off", made_before_tip: false }),
          ],
          { n_pre_tip: 1 },
        )}
      />,
    );

    const review = screen.getByTestId("post-game-review");
    expect(review.textContent).toContain("2 player lines graded");
    expect(review.textContent).toContain("1 made before tip-off");
    expect(screen.getAllByText(/made after tip-off/i).length).toBeGreaterThan(0);
    // The late row is present with its grade, not silently dropped.
    expect(within(review).getAllByText("Late Line").length).toBeGreaterThan(0);
    expect(review.textContent).toContain("Way off");
  });

  it("draws nothing when the game has no counted grading", () => {
    const { container, rerender } = render(<PostGameReview grading={null} />);
    expect(container.querySelector("[data-testid=post-game-review]")).toBeNull();

    rerender(<PostGameReview grading={grading([])} />);
    expect(container.querySelector("[data-testid=post-game-review]")).toBeNull();
  });

  it("shows no within-range count when no line carries a grade", () => {
    render(
      <PostGameReview
        grading={grading(
          [row({ player_id: "1", stat: "points", predicted: 20, actual: 31, delta: 11, grade: null })],
          { bands: null },
        )}
      />,
    );

    expect(screen.getByTestId("post-game-review")).toBeTruthy();
    expect(screen.queryByTestId("review-on-the-number")).toBeNull();
    expect(screen.queryByTestId("review-surprises")).toBeNull();
  });
});
