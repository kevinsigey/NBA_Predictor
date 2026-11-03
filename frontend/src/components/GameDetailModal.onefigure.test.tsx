/**
 * The reviewer's Phase-1 live follow-ups, measured rather than eyeballed.
 *
 * Two classes of leftover were found on the live NBA modal:
 *
 *  1. **A bare heading with nothing under it.** `FixtureFlow`'s pre-game row is
 *     the fixture's own name and nothing else, so before tip-off it rendered an
 *     `MIA vs TOR` heading with no sentence beneath it. Measured across the
 *     three states, not assumed: the pre-game flow carries ONE row and it is
 *     that heading; the finished flow carries two real sentences (the result and
 *     the pick's rightness) that must survive. So the fix is not a blanket
 *     deletion — it is a heading that is conditional on rows existing beneath
 *     it, which is asserted here rather than checked by looking.
 *
 *  2. **The legacy header strip repeated the block's figures.** It read
 *     `53% MIA to win | MIA by 4.8 | 230.6` ABOVE the instant block, so the
 *     moneyline probability was on the page three times. The block owns the
 *     figures, so the header's copies go — but only the ones the block actually
 *     draws, and the field-by-field audit below is what decides which.
 *
 * The audit, measured on a rendered modal (see each test's comment):
 *   - pick probability   — the block draws it in the `moneyline` tile AND on the
 *                          bar, so the header's copy is a THIRD. Removed.
 *   - projected margin   — the block draws NO spread tile (the game carries no
 *                          market line), so the header's is the ONLY copy. Kept.
 *   - projected total    — same: no total tile, because no total line. Kept.
 * So the strip keeps its two unique figures and loses only the duplicated one.
 */
import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import GameDetailModal from "./GameDetailModal";
import { api, type TrackRecord } from "../api/client";

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
    // The gate's sibling route. Empty here: these tests count figures on the
    // page, and the ranking block's absence is TopCalls' own test to cover.
    getGameOutPlayers: vi.fn().mockResolvedValue([]),
    getGameDoubtfulPlayers: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn(),
    getTrackRecord: vi.fn(),
    explainGame: vi.fn(),
  },
}));

/** A pre-tip game where the model's pick is the AWAY side, so the audit's
 *  figures cannot collide with the home team's code or name by accident.
 *
 *  `home_win_probability: 0.47` makes TOR the pick at 53%, and a
 *  `predicted_margin` of -4.8 makes TOR the margin side too — so both figures
 *  agree with each other and the header is not falling into the "Toss-up"
 *  branch that would print neither team. */
const preTip = {
  game_id: "g1",
  game_date: "2026-11-01",
  home_team: "MIA",
  away_team: "TOR",
  prediction: { home_win_probability: 0.47, predicted_margin: -4.8, predicted_total: 230.6 },
  completed: false,
  home_pts: null,
  away_pts: null,
  markets: [],
  head_to_head: [],
  home_recent_form: [],
  away_recent_form: [],
};

/** The same game once it is over. The flow gains two real sentences here, and
 *  this is the fixture that holds them: the fix must not delete them. */
const finished = { ...preTip, completed: true, home_pts: 113, away_pts: 105 };

const winnerPick: TrackRecord = {
  market: "game_outcome",
  total_predictions: 40,
  correct_predictions: 22,
  hit_rate: 0.55,
  n_rebuilt: 3,
};

beforeEach(() => {
  vi.mocked(api.getHubPlayers).mockResolvedValue([]);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([]);
  vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([]);
  vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick]);
});
afterEach(() => {
  vi.restoreAllMocks();
});

async function renderModal(detail: unknown) {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail as never);
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  await screen.findByTestId("instant-block");
}

/** How many times a string appears in some text. Counted over the WHOLE
 *  document, not scoped to a component: a duplicate is a duplicate wherever it
 *  sits, and scoping the count is how the original triplication passed. */
function occurrences(needle: string, root: ParentNode = document.body): number {
  return (root.textContent ?? "").split(needle).length - 1;
}

describe("the pre-game heading is conditional on rows existing beneath it", () => {
  it("renders no heading, and no bare flow, before tip-off", async () => {
    await renderModal(preTip);

    const flow = screen.getByTestId("fixture-flow");

    // The defect: an `MIA vs TOR` heading with nothing under it. Asserted as an
    // absence of HEADINGS rather than as an absence of that one string, so a
    // future rename of the teams cannot make this pass while the bare heading
    // is still on screen.
    expect(within(flow).queryByRole("heading")).toBeNull();
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();

    // And the flow holds no bare heading text at all. The fixture's own name
    // still appears above the panel, in the modal's own title — what must not
    // happen is the flow restating it as a heading with nothing beneath.
    expect(flow.textContent?.trim()).toBe("");

    // The facts are untouched: the block is still there and still says the pick.
    // The heading went; the block did not.
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    expect(screen.getByText("Raptors is the pick.")).toBeInTheDocument();
  });

  it("keeps the flow's real sentences once the game is finished", async () => {
    await renderModal(finished);

    const flow = screen.getByTestId("fixture-flow");

    // MEASURED, and the reason this is not a blanket deletion: the finished
    // flow carries two genuine sentences that carry news the block does not
    // state. A fix that removed the whole flow would have taken these with it.
    expect(flow).toHaveTextContent("The result is a win for MIA.");
    expect(flow).toHaveTextContent("The pick rightness: the model's pick was wrong.");

    // No heading here either — the finished rows are sentences, not a name.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
  });

  it("says the score while the game is live, and no bare heading with it", async () => {
    // MEASURED before the fix: a game with a score and `completed: false` was
    // mapped to PRE-GAME, so it rendered the same bare `MIA vs TOR` heading a
    // not-yet-started game did — while its own data carried a score nobody had
    // been shown. It now takes the in-play branch, which is the state whose
    // sentences exist for exactly this case.
    await renderModal({ ...preTip, home_pts: 60, away_pts: 55 });

    const flow = screen.getByTestId("fixture-flow");

    // The score sentence, which is live content the old mapping threw away.
    expect(flow).toHaveTextContent("The score is MIA 60, TOR 55.");

    // And the rule still holds in this state too: a heading is only ever
    // rendered when there are rows beneath it, and these rows are sentences.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
  });
});

describe("each figure appears exactly once on the page", () => {
  it("draws the pick probability in the block and nowhere else", async () => {
    await renderModal(preTip);

    // The one-source rule, counted on the whole document. Before the fix this
    // was 3: the legacy header's `53%`, the block's `moneyline` tile and the
    // bar's own segment label. The block owns the figure, so the header's copy
    // is what goes.
    expect(occurrences("53%")).toBe(2); // the tile's value + the bar's label
    expect(occurrences("47%")).toBe(1); // the other side, bar only

    // Nothing above the block repeats them. Scoped to everything OUTSIDE the
    // block, which is the half of the page the block cannot own.
    const block = screen.getByTestId("instant-block");
    const outside = document.body.textContent!.replace(block.textContent ?? "", "");
    expect(outside.split("53%").length - 1).toBe(0);
  });

  it("keeps the header strip's two figures the block does NOT draw", async () => {
    await renderModal(preTip);

    // The field-by-field audit, as assertions. The game carries no market line,
    // so `panelFacts` builds no spread tile and no total tile — the block
    // genuinely has no copy of these two, and the header strip is their only
    // source. Deleting the strip wholesale would have removed real figures.
    expect(screen.getByText("TOR by 4.8")).toBeInTheDocument();
    expect(screen.getByText("230.6")).toBeInTheDocument();

    // ...and each is on the page once, not twice.
    expect(occurrences("TOR by 4.8")).toBe(1);
    expect(occurrences("230.6")).toBe(1);
  });

  it("still states the win probability once, in the block's own tile", async () => {
    await renderModal(preTip);

    // The figure the header used to repeat is not gone from the page — it is
    // gone from the header. The block's tile is where a reader now finds it,
    // and it is labelled with what the number means rather than asserted as a
    // separate "TOR to win" sentence.
    const tile = screen.getByTestId("tile-moneyline");
    expect(tile).toHaveTextContent("53%");
    expect(tile).toHaveTextContent("TOR");

    // The header's own wording is gone with it: it stated the pick twice over
    // (a percentage and a "to win" sentence) where the block states it once.
    expect(screen.queryByText(/to win$/)).toBeNull();
  });

  it("does not repeat the block's figures after the summary is pressed", async () => {
    await renderModal(preTip);
    vi.mocked(api.explainGame).mockResolvedValue({
      verdict: "Toronto is the pick on the road.",
      band: "moderate",
      factors: [{ key: "moneyline", direction: "up", headline: "Road edge", text: "Toronto away." }],
      source: "template" as const,
      model: "",
      generated_at: new Date().toISOString(),
      sport: "nba",
      pick_timing: "pre_kickoff" as const,
    });
    await screen.findByRole("button", { name: /ai summary/i });
    // Press it for real, so the summary state is the one under test.
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.click(screen.getByRole("button", { name: /ai summary/i }));
    const summary = await screen.findByTestId("fixture-summary");

    // The count is a whole-page count in the summary state too: the prose is
    // the button's product, and the figures are still the block's alone.
    expect(within(summary).queryByTestId("tile-moneyline")).toBeNull();
    expect(occurrences("53%")).toBe(2);
    expect(occurrences("TOR by 4.8")).toBe(1);
    expect(occurrences("230.6")).toBe(1);
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
  });

  it("keeps the finished game's review, which states its own figures", async () => {
    await renderModal(finished);

    // The post-match review is not a duplicate of the block: it compares the
    // prediction to what happened, which is the one thing the block cannot say.
    // It is a different set of figures (actual score, the error), so it stays.
    const review = screen.getByTestId("post-match-verdict");
    expect(review).toHaveTextContent("Predicted margin");
    expect(review).toHaveTextContent("Predicted total");

    // And the header strip, which stood in for those figures before the game,
    // is not also standing in for them after it.
    expect(screen.queryByText("Projected margin")).toBeNull();
    expect(screen.queryByText("Projected total points")).toBeNull();
  });
});
