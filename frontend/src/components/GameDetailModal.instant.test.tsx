import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
    // The gate's sibling route. Empty here: these tests assert each figure
    // appears exactly once, and the ranking block would add figures to the page.
    // Its own rendering is covered in TopCalls.test.tsx.
    getGameOutPlayers: vi.fn().mockResolvedValue([]),
    getGameDoubtfulPlayers: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn(),
    getTrackRecord: vi.fn(),
    explainGame: vi.fn(),
  },
}));

/** The winner-pick row, as `GET /hub/track-record` sends it.
 *
 *  `game_outcome` is the market that settles the model's own win/loss call, and
 *  `hub_service._settle_game_outcome` builds it from ALL counted picks. The
 *  headline includes picks made before AND after tip-off; the pre-tip subset
 *  is in the `pre_tip` sub-record. The live data shows 1373 total, 1365 rebuilt
 *  (made after tip-off), and only 8 pre-tip (3 correct).
 */
const winnerPick = (total: number, correct: number, nRebuilt: number, preTipTotal: number, preTipCorrect: number): TrackRecord => ({
  market: "game_outcome",
  total_predictions: total,
  correct_predictions: correct,
  hit_rate: total ? Number((correct / total).toFixed(3)) : null,
  settled: true,
  n_rebuilt: nRebuilt,
  n_pre_tip: preTipTotal,
  pre_tip: {
    total_predictions: preTipTotal,
    correct_predictions: preTipCorrect,
    hit_rate: preTipTotal ? Number((preTipCorrect / preTipTotal).toFixed(3)) : null,
    n_push: 0,
    weekly: [],
  },
});

/** The bookmaker-price row, present to prove it is not the one being quoted. */
const vsMarket: TrackRecord = {
  market: "h2h",
  total_predictions: 38,
  correct_predictions: 21,
  hit_rate: 0.553,
};

const preTip = {
  game_id: "g1",
  game_date: "2026-11-01",
  home_team: "BOS",
  away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false,
  home_pts: null,
  away_pts: null,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
  ],
  head_to_head: [],
  home_recent_form: [],
  away_recent_form: [],
};

const completed = { ...preTip, completed: true, home_pts: 113, away_pts: 105 };

afterEach(() => {
  vi.restoreAllMocks();
});

// `vi.restoreAllMocks()` wipes a mockResolvedValue set at module scope, and the
// box score's team split is a join against this feed, so re-arm per test.
beforeEach(() => {
  vi.mocked(api.getHubPlayers).mockResolvedValue([]);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([]);
  vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([]);
  // Live data: 1373 total, 1174 correct, 1365 rebuilt, 8 pre-tip (3 correct)
  vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(1373, 1174, 1365, 8, 3)]);
});

/** Render with the network hard-blocked: every `fetch` rejects. The block's
 *  whole claim is that it needs none of it, so the parity rule is that the
 *  figures are on screen anyway. */
async function renderOffline(detail: unknown) {
  const fetchSpy = vi.fn(() => Promise.reject(new Error("network blocked")));
  vi.stubGlobal("fetch", fetchSpy);
  vi.mocked(api.getGameDetail).mockResolvedValue(detail as never);
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  await screen.findByTestId("instant-block");
  return fetchSpy;
}

/** The one record line: the pre-tip winner record and its link. */
function recordLine() {
  return screen.getByText(/Model, picks made before tip-off:/);
}

/** A summary, shaped the way the service sends one. The prose is the only thing
 *  it carries: `pick` is deliberately absent, because a service that named a
 *  pick the bundle does not would put a second claim on the page. */
const summaryAnswer = {
  verdict: "Boston is the pick at home.",
  band: "moderate",
  factors: [{ key: "moneyline", direction: "up", headline: "Home edge", text: "Boston at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "nba",
  pick_timing: "pre_kickoff" as const,
};

/** Press the button and wait for the summary to land. */
async function pressAiSummary() {
  await userEvent.click(screen.getByRole("button", { name: /ai summary/i }));
  return screen.findByTestId("fixture-summary");
}

/** The fill colour each bar segment is painted in, in bar order.
 *
 *  `ProbabilityBar` resolves the pick's index itself and paints that segment
 *  `var(--color-pr-accent)`; jsdom carries the custom property through as text,
 *  which is what makes the accent assertable rather than merely visible. */
function segmentFills() {
  const fills = screen.getAllByTestId("pbar-fill");
  const labels = screen.getAllByTestId("pbar-label").map((el) => el.getAttribute("data-seg"));
  return fills.map((el, i) => ({
    label: labels[i],
    accent: (el.getAttribute("style") ?? "").includes("var(--color-pr-accent)"),
  }));
}

describe("GameDetailModal instant block", () => {
  it("renders the block from the bundle with the network blocked", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(40, 22, 3, 8, 3)]);
    const fetchSpy = await renderOffline(preTip);

    // The tiles NBA already had, unchanged.
    expect(screen.getByTestId("tile-moneyline")).toBeInTheDocument();
    // The verdict, from the bundle's own pick.
    expect(screen.getByText("Celtics is the pick.")).toBeInTheDocument();
    // The record, one line, from the winner-pick row's pre-tip block.
    expect(recordLine()).toHaveTextContent("Model, picks made before tip-off: 38%");

    // No request was made, and in particular not for the AI summary: the
    // button is still there to press.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(api.explainGame).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /ai summary/i })).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("says what the AI button adds, under the button", async () => {
    await renderOffline(preTip);
    // The promise is what the button buys, and it has to be on screen before
    // the button is pressed or the reader has nothing to decide against.
    const promise = screen.getByTestId("ai-promise");
    expect(promise).toHaveTextContent(/model vs line/i);
    expect(promise).toHaveTextContent(/who's out/i);
    // ...and after the block, not inside it: the block is facts, the button is
    // the one thing it cannot state.
    expect(
      screen.getByTestId("instant-block").compareDocumentPosition(promise) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    vi.unstubAllGlobals();
  });

  it("keeps the box score's team split out of the block", async () => {
    // The other half of "do not touch": the split belongs to the box score, and
    // the block must not grow a second copy of it.
    vi.mocked(api.getGamePlayers).mockResolvedValue([
      { player_id: "p1", player_name: "Jayson Tatum", stat: "points", predicted_value: 27.5, actual_value: null },
    ]);
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      { player_id: "p1", player_name: "Jayson Tatum", team: "BOS", position: "G", rating: 0, live_form_rating: 0, points_per_game: 0, rebounds_per_game: 0, assists_per_game: 0, fg_pct: 0, three_pt_pct: 0, ft_pct: 0, usage_rate: 0, minutes_per_game: 0 },
    ]);
    await renderOffline(preTip);

    expect(screen.getByRole("rowheader", { name: "Jayson Tatum" })).toBeInTheDocument();
    expect(within(screen.getByTestId("instant-block")).queryByRole("rowheader")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("replaces the PregamePick sentence with the timing badge, and never both", async () => {
    await renderOffline({ ...completed, rebuilt: true });

    // The badge says it once: when the pick was made, in the sport's own
    // moment. (predictor-ui reworded `Rebuilt after tip-off` to
    // `Made after tip-off` and dropped the "not counted" claim; the key
    // `rebuilt` and its meaning are unchanged.)
    expect(screen.getByText("Made after tip-off")).toBeInTheDocument();
    // The old prose is gone. If it were still here a reader would meet the
    // same fact twice, in two different sentences.
    expect(screen.queryByText(/Rebuilt after tip-off:/)).toBeNull();
    expect(screen.queryByText(/Not counted in the record\./)).toBeNull();
    expect(screen.queryByText(/not counted/i)).toBeNull();
    // ...and a rebuilt pick is still not judged.
    expect(screen.queryByTestId("post-match-verdict")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows the quiet chip for a pick made before tip-off", async () => {
    await renderOffline(completed);

    expect(screen.getByText("Made before tip-off")).toBeInTheDocument();
    // A pick made in time is not a badge case, and must not be dressed as one.
    expect(screen.queryByText("Made after tip-off")).toBeNull();
    expect(screen.queryByText(/not counted/i)).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows one pre-tip record line, not the headline's three numbers", async () => {
    await renderOffline(preTip);

    // The one line the games surfaces quote: the pre-tip subset, with a link.
    const line = recordLine();
    expect(line).toHaveTextContent("Model, picks made before tip-off: 38%");
    expect(within(line).getByRole("link", { name: "Track record" })).toHaveAttribute("href", "/track-record");
    // The headline (all picks, including reruns) and the rebuilt count are
    // the record page's business; a game card does not restate them.
    expect(screen.queryByText("Every pick")).toBeNull();
    expect(screen.queryByText("1174/1373")).toBeNull();
    expect(screen.queryByText("1365 of them made after tip-off")).toBeNull();
    expect(screen.queryByText("Before tip-off")).toBeNull();
    expect(screen.queryByTestId("record-fill")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("never quotes the bookmaker-price row as the pick's record", async () => {
    // `h2h` (38 graded, 21 right) is a real row in the same response. It
    // measures the same probability against a price, so putting it under the
    // verdict would state one pick two ways.
    vi.mocked(api.getTrackRecord).mockResolvedValue([vsMarket]);
    await renderOffline(preTip);

    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
    expect(screen.queryByText("21/38")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows no record strip when the track record has no winner-pick row", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    await renderOffline(preTip);

    // No row, no line. What must never appear is a fabricated 0/0.
    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("stays silent, not 0/0, when the winner-pick row has settled nothing", async () => {
    // 0 total, 0 correct, 0 rebuilt, 0 pre-tip, 0 pre-tip correct
    vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(0, 0, 0, 0, 0)]);
    await renderOffline(preTip);

    // A row that has graded nothing has no pre-tip figure to quote, so there
    // is no line. `0/0` would be a claim about a record that does not exist.
    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows no record strip while the track record is still loading", async () => {
    // A line that renders 0/0 for a moment and then corrects itself is a lie
    // a reader can catch. Nothing arrives until the numbers do.
    vi.mocked(api.getTrackRecord).mockReturnValue(new Promise(() => {}));
    await renderOffline(preTip);

    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("loses the record without losing the game", async () => {
    vi.mocked(api.getTrackRecord).mockRejectedValue(new Error("track record down"));
    await renderOffline(preTip);

    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    expect(screen.getByTestId("tile-moneyline")).toBeInTheDocument();
    expect(screen.queryByText(/couldn.t load this game/i)).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("still says plainly that no pick was made", async () => {
    await renderOffline({ ...preTip, prediction: null });

    expect(screen.queryByText("Celtics is the pick.")).toBeNull();
    expect(screen.getByText(/No pick was made for this fixture\./)).toBeInTheDocument();
    // No pick is not no facts: the block still carries what the site has.
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  // ---- The reviewer's four, for this site -------------------------------

  it("REQUIRED: shows the block before the button, having asked for nothing", async () => {
    const fetchSpy = await renderOffline(preTip);

    const block = await screen.findByTestId("instant-block");
    const button = screen.getByRole("button", { name: /ai summary/i });

    // The block is finished before the button is even on the page: it is above
    // it, and it is the reason the button is optional rather than the price of
    // admission.
    expect(
      block.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING,
      "the block must render above the summary button",
    ).toBeTruthy();

    // Every figure the block shows is on screen with no request spent...
    expect(within(block).getByTestId("tile-moneyline")).toBeInTheDocument();
    expect(within(block).getAllByTestId("pbar-fill")).toHaveLength(2);
    expect(within(block).getByText("Celtics is the pick.")).toBeInTheDocument();
    expect(screen.getByText(/Model, picks made before tip-off:/)).toBeInTheDocument();

    // ...and nothing was asked for to get them.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(api.explainGame).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("REQUIRED: each figure appears exactly once after the button is pressed", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(40, 22, 3, 10, 5)]);
    await renderOffline(preTip);
    vi.mocked(api.explainGame).mockResolvedValue(summaryAnswer as never);
    const summary = await pressAiSummary();

    // The block is the only place the figures live, so each is on screen once.
    // Counted across the whole panel, not inside the block: a second copy
    // outside it — in the summary, in a footer, in the flow — is the overlap
    // this phase exists to remove, and scoping the count to the block would
    // report green while the duplicate stood next to it.
    expect(screen.getAllByTestId("tile-moneyline")).toHaveLength(1);
    expect(screen.getAllByTestId("pbar-fill")).toHaveLength(2);
    expect(screen.getAllByTestId("pbar-label")).toHaveLength(2);
    expect(screen.getAllByText(/Model, picks made before tip-off:/)).toHaveLength(1);
    expect(screen.getAllByText("Celtics is the pick.")).toHaveLength(1);

    // And the old duplicate sections are gone from the summary state: it used
    // to re-render the tiles, the bar, the legend and the record.
    expect(within(summary).queryByTestId("tile-moneyline")).toBeNull();
    expect(within(summary).queryAllByTestId("pbar-fill")).toHaveLength(0);
    expect(within(summary).queryByTestId("pbar-legend")).toBeNull();
    expect(within(summary).queryByText(/Model, picks made before tip-off:/)).toBeNull();
    // The prose is still there — the summary is what the button buys.
    expect(within(summary).getByText("Boston is the pick at home.")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("REQUIRED: accents the bar segment the bundle's pick names", async () => {
    await renderOffline(preTip);

    // The bundle's pick is "BOS" and the segments are labelled with this site's
    // own team codes, so the label matches exactly and one segment is accented.
    // The accent is derived from `bundle.pick` inside the block; a site whose
    // pick label matched nothing would accent nothing at all, which is the
    // fail-closed behaviour — so this asserts the accent, not just the absence
    // of a crash.
    const segments = segmentFills();
    expect(segments.map((s) => s.label)).toEqual(["BOS", "MIA"]);
    expect(segments.filter((s) => s.accent).map((s) => s.label)).toEqual(["BOS"]);

    // The accent is on the segment the verdict names, which is the whole point:
    // the emphasis and the sentence must agree about which side was picked.
    const accented = segments.find((s) => s.accent)!;
    expect(screen.getByText("Celtics is the pick.")).toBeInTheDocument();
    expect(accented.label).toBe("BOS");
    vi.unstubAllGlobals();
  });

  it("REQUIRED: accents the away segment when the away team is the pick", async () => {
    // The companion to the test above, and the one that could actually pass by
    // accident: with MIA the favourite the accent has to move to the SECOND
    // segment. A bar that painted whichever segment came first would satisfy
    // the previous test and fail this one.
    await renderOffline({
      ...preTip,
      prediction: { home_win_probability: 0.38, predicted_margin: -3.5, predicted_total: 224.5 },
    });

    const segments = segmentFills();
    expect(segments.map((s) => s.label)).toEqual(["BOS", "MIA"]);
    expect(segments.filter((s) => s.accent).map((s) => s.label)).toEqual(["MIA"]);
    vi.unstubAllGlobals();
  });

  it("REQUIRED: draws no bar, and so accents nothing, when the model has no split", async () => {
    // There is no probability to split, so there is no bar. Nothing to accent
    // is the honest outcome here rather than a failure to accent: the block
    // fails closed instead of pointing at the widest segment it can see.
    await renderOffline({ ...preTip, prediction: null });

    expect(screen.queryByTestId("pbar-fill")).toBeNull();
    expect(screen.queryByTestId("pbar-label")).toBeNull();
    expect(screen.getByText(/No pick was made for this fixture\./)).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("keeps the record on a fixture the model made no pick for", async () => {
    // A record is a fact about the season, not about this fixture's pick, so it
    // stands on its own. An earlier guard that required a verdict dropped it
    // silently whenever the bundle had no pick.
    vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(40, 22, 3, 8, 3)]);
    await renderOffline({ ...preTip, prediction: null });

    expect(screen.getByText(/No pick was made for this fixture\./)).toBeInTheDocument();
    expect(recordLine()).toHaveTextContent("Model, picks made before tip-off: 38%");
    vi.unstubAllGlobals();
  });

  it("puts the block above the summary once the summary is in", async () => {
    await renderOffline(preTip);
    vi.mocked(api.explainGame).mockResolvedValue(summaryAnswer as never);
    const summary = await pressAiSummary();

    // Facts first, interpretation after: the block is the finished "what", so
    // the summary reads as what it adds and never as a second copy of the page.
    // The flow is mounted throughout, so the panel is never empty.
    expect(
      screen.getByTestId("instant-block").compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.getByTestId("fixture-flow")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
