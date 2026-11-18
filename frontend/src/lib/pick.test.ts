import { describe, expect, it } from "vitest";
import { marketFavourite, favourite, marginLine, pickWon } from "./pick";

const p = (home_win_probability: number, predicted_margin: number) => ({ home_win_probability, predicted_margin, predicted_total: 220 });

describe("pick helpers", () => {
  it("leads with the favourite, home at exactly 50%", () => {
    expect(favourite(p(0.47, -4.8), "BOS", "MIA")).toEqual({ team: "MIA", prob: 0.53 });
    expect(favourite(p(0.5, 0), "BOS", "MIA").team).toBe("BOS");
  });
  it("writes the margin only when it agrees with the pick", () => {
    expect(marginLine(p(0.62, 3.46), "BOS", "MIA")).toBe("BOS by 3.5");
    expect(marginLine(p(0.55, -0.6), "BOS", "MIA")).toBe("Toss-up");
    expect(marginLine(p(0.55, 0.3), "BOS", "MIA")).toBe("Toss-up");
  });
  it("judges the pick against the result", () => {
    expect(pickWon(p(0.62, 3), 110, 100)).toBe(true);
    expect(pickWon(p(0.62, 3), 90, 100)).toBe(false);
  });
});

describe("marketFavourite", () => {
  const p = { home_win_probability: 0.6, predicted_margin: 3, predicted_total: 220 };
  it("is null without a market line, never a default", () => {
    expect(marketFavourite(p, "BOS", "MIA")).toBeNull();
    expect(marketFavourite({ ...p, market_home_win_probability: null }, "BOS", "MIA")).toBeNull();
  });
  it("names the market's side, which can differ from the model's", () => {
    expect(marketFavourite({ ...p, market_home_win_probability: 0.45 }, "BOS", "MIA")).toEqual({ team: "MIA", prob: 0.55 });
  });
});
