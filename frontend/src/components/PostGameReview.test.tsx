import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import PostGameReview from "./PostGameReview";
import type { PlayerProp } from "../api/client";

const BANDS = { points: 4.6, rebounds: 1.9, assists: 1.3, threes: 0.9 };

const prop = (over: Partial<PlayerProp> & { player_id: string; stat: string }): PlayerProp => ({
  player_name: `Player ${over.player_id}`,
  predicted_value: 20,
  actual_value: null,
  ...over,
} as PlayerProp);

describe("PostGameReview", () => {
  it("names the players who beat the projection and by how much", () => {
    render(
      <PostGameReview
        props={[prop({ player_id: "1", player_name: "Jayson Tatum", stat: "points", predicted_value: 22.4, actual_value: 31 })]}
        bands={BANDS}
      />,
    );

    const beat = screen.getByTestId("review-beat");
    expect(within(beat).getByText("Jayson Tatum")).toBeTruthy();
    expect(beat.textContent).toContain("Proj 22.4 \u2192 31 actual");
    expect(within(beat).getByText("\u25b2 +8.6 vs proj")).toBeTruthy();
    // 8.6 against a 4.6 band is inside twice the error: Off.
    expect(beat.textContent).toContain("Off");
  });

  it("names the players who fell short", () => {
    render(
      <PostGameReview
        props={[prop({ player_id: "2", player_name: "Jalen Brunson", stat: "points", predicted_value: 24.1, actual_value: 18 })]}
        bands={BANDS}
      />,
    );

    const short = screen.getByTestId("review-short");
    expect(within(short).getByText("Jalen Brunson")).toBeTruthy();
    expect(short.textContent).toContain("Proj 24.1 \u2192 18 actual");
    expect(within(short).getByText("\u25bc \u22126.1 vs proj")).toBeTruthy();
  });

  it("counts the lines that were on the number", () => {
    render(
      <PostGameReview
        props={[
          prop({ player_id: "1", stat: "points", predicted_value: 20, actual_value: 22 }),
          prop({ player_id: "2", stat: "points", predicted_value: 20, actual_value: 30 }),
        ]}
        bands={BANDS}
      />,
    );

    const onNumber = screen.getByTestId("review-on-the-number");
    expect(onNumber.textContent).toContain("1 of 2");
    expect(onNumber.textContent).toContain("player lines within range");
  });

  it("draws nothing without bands: there is no measured error to judge against", () => {
    const { container } = render(
      <PostGameReview
        props={[prop({ player_id: "1", stat: "points", predicted_value: 20, actual_value: 31 })]}
        bands={null}
      />,
    );

    expect(container.querySelector("[data-testid=post-game-review]")).toBeNull();
  });

  it("draws nothing when no player has an actual to judge", () => {
    const { container } = render(
      <PostGameReview
        props={[prop({ player_id: "1", stat: "points", predicted_value: 20, actual_value: null })]}
        bands={BANDS}
      />,
    );

    expect(container.querySelector("[data-testid=post-game-review]")).toBeNull();
  });

  it("says nobody when a column is empty", () => {
    render(
      <PostGameReview
        props={[prop({ player_id: "1", stat: "points", predicted_value: 20, actual_value: 31 })]}
        bands={BANDS}
      />,
    );

    expect(screen.getByTestId("review-short").textContent).toContain("Nobody.");
  });
});
