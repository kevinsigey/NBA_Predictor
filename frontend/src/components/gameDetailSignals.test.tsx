/**
 * NBA's game modal fetches and renders its `absence` row.
 *
 * `/api/signals/{game_id}` shipped in #33 and is **live in production** — but
 * nothing on this page asked for it, so the endpoint served rows to no reader. The
 * vendored `SignalRows` can draw `absence_strip`; this is the hop between them, and
 * it mirrors Sports' `GameDetailModal` and PL's `FixtureModal` so "a fixture page
 * shows its signals" stays one implementation across all three.
 *
 * ## Why this is worth more on NBA than on the other two
 *
 * NBA is the sport where the absence was *invisible*: `api/routes.get_game_players`
 * drops an out player's rows before serialising, and `OutPlayerOut` carries no
 * projection, no stat and no rank. So today the page shows the out player as a
 * one-line mention under the lists and **nothing else** — while every player he
 * outranks silently gains a place. This row is the first thing on the page that says
 * what he was expected to do.
 *
 * ## EVERY failure is silence, and the failures are indistinguishable
 *
 * A 404, a network error and an honest empty list all leave `[]`. Spec §2's "no data,
 * no row" forbids a placeholder, and a signal is an *enhancement* on this page: it
 * must never become the page's error state.
 *
 * ## The gate is `detail.completed`, not a truthiness test
 *
 * `/api/signals/{game_id}` answers `[]` for a game that has started, and the reason
 * is the same one that empties the props feed: a started game's roster is the one
 * known *before* tip-off, so quoting it afterwards is hindsight. Asking is skipped
 * outright, once per started game, rather than asking for an answer already known.
 *
 * **This heading used to say `detail.game_status`, and that was the bug.** This
 * API sends `completed: boolean` and has no `game_status` field at all, so the
 * docstring described a gate that never existed. The first version of the component
 * matched it, and the first version of this fixture set `game_status: "final"` too --
 * so the test and the component agreed on the same fiction and the suite was green
 * against a field the backend does not send. `tsc --noEmit` was what broke the tie,
 * because the property is not on the `GameDetail` type. Both are corrected here, and
 * the fixture below carries `completed` only.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(),
    getGameOutPlayers: vi.fn(),
    getGameDoubtfulPlayers: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn(),
    getHubTeams: vi.fn(),
    getHubRankings: vi.fn(),
    getHubStandings: vi.fn(),
    getTrackRecord: vi.fn(),
    getVsMarket: vi.fn(),
    getManifest: vi.fn(),
    getCalibration: vi.fn(),
    explainGame: vi.fn(),
    // The signal fetch. A bare `vi.fn()` returns `undefined`, which the modal
    // calls `.then` on -- the same trap PL's five client mocks fell into.
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "401", signals: [] })),
    // The post-game review's own feed: null is its absence (no counted props, or unreadable).
    getGamePropGrading: vi.fn(async () => null),
  },
}));

/** The one absence row NBA's adapter can produce, in the shape it sends. */
const ABSENCE = {
  kind: "absence",
  sport: "nba",
  game_id: "401",
  headline: { text: "Out: Big, our #1 Points projection (31.5 pts)", figures: { projection: 31.5 } },
  n: 1,
  source: "ESPN injury report, 2026-10-04",
  as_of: "2026-10-04",
  strength: 1.0,
  pre_kickoff_only: true,
  visual: "absence_strip",
};

const detail = {
  game_id: "401",
  game_date: "2026-10-05",
  home_team: "BOS",
  away_team: "LAL",
  completed: false,
  home_pts: null,
  away_pts: null,
  markets: [],
  head_to_head: [],
  home_recent_form: [],
  away_recent_form: [],
};

/** Props with enough players that the projections section renders at all. */
const PLAYERS = [
  { player_id: "p1", player_name: "A. Player", stat: "points", predicted_value: 28.0, actual_value: null },
  { player_id: "p2", player_name: "B. Player", stat: "points", predicted_value: 24.0, actual_value: null },
];

function mockApi(over: Record<string, unknown> = {}) {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail } as never);
  vi.mocked(api.getGamePlayers).mockResolvedValue(PLAYERS as never);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getHubPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getHubTeams).mockResolvedValue([] as never);
  vi.mocked(api.getHubRankings).mockResolvedValue([] as never);
  vi.mocked(api.getHubStandings).mockResolvedValue([] as never);
  vi.mocked(api.getTrackRecord).mockResolvedValue([] as never);
  vi.mocked(api.getVsMarket).mockResolvedValue(null as never);
  vi.mocked(api.getManifest).mockResolvedValue(null as never);
  vi.mocked(api.getCalibration).mockResolvedValue([] as never);
  vi.mocked(api.explainGame).mockRejectedValue(new Error("not needed"));
  Object.assign(api, over);
}

async function openModal() {
  render(<GameDetailModal gameId="401" onClose={() => {}} />);
  // The props have to be in before the absence row means anything: this page hides
  // its projections entirely when the availability feed has not loaded.
  await waitFor(() => expect(api.getGamePlayers).toHaveBeenCalled());
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("NBA's absence row on the game modal", () => {
  it("asks for the signals when an upcoming game opens", async () => {
    mockApi();
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalledWith("401"));
  });

  it("renders the row the endpoint sent", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    // Read back out of the DOM: a screenshot proves a thing was drawn, only the
    // text says whether it was the RIGHT thing.
    expect(await screen.findByText(/Out: Big/)).toBeInTheDocument();
  });

  it("draws the figure beside the headline", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    // `SignalRows` REFUSES a row whose words do not state this figure, so its
    // presence proves the marker and the headline agree.
    const marker = await screen.findByTestId("signal-absence");
    expect(marker).toHaveTextContent("31.5");
  });

  it("renders NO row and no empty state when the endpoint has none", async () => {
    mockApi();
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalled());
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    expect(screen.queryByText(/Out:/)).not.toBeInTheDocument();
  });

  it("renders nothing when the fetch rejects, and the page survives", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockRejectedValue(new Error("404"));
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalled());
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    // The page itself is still there -- the row is an enhancement, never the page's
    // error state.
    expect(await screen.findByText(/Player projections/)).toBeInTheDocument();
  });

  it("does not ask about a completed game", async () => {
    // The endpoint answers [] for those: a completed game's roster is the one known
    // before tip-off, and quoting it afterwards is hindsight.
    //
    // `completed`, not a `game_status` this API does not send. An earlier version of
    // this fixture set `game_status: "final"`, and the test passed -- because the
    // COMPONENT was reading that same non-existent field, so the two agreed on
    // fiction and the gate looked like it worked. `tsc --noEmit` caught it: the
    // property is not on `GameDetail`. Against the real backend the gate was inert
    // and every completed game would have been asked.
    mockApi();
    vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, completed: true } as never);
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    await waitFor(() => expect(api.getGameDetail).toHaveBeenCalled());
    expect(api.getGameSignals).not.toHaveBeenCalled();
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
  });

  it("does not read one game's completion to decide about the next", async () => {
    // The stale-status defect, and it only shows when the modal is REUSED for a
    // different game -- which it is, from the schedule list. The reset is a
    // `setDetail(null)` inside another effect, and an effect reads the value
    // captured by its own render, so on the render where `gameId` changes `detail`
    // is still the PREVIOUS game's.
    //
    // Asserted as a CALL COUNT because that is the whole of it: the fetch is keyed
    // on `gameId` and every write is guarded by `cancelled`, so no previous game's
    // data can reach the page. A `!detail` gate reads the old game's `completed`
    // and fires for the new id immediately, then again once its detail lands.
    mockApi();
    const { rerender } = render(<GameDetailModal gameId="401" onClose={() => {}} />);
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalledTimes(1));

    // A COMPLETED previous game, and a new id whose detail never lands.
    vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, completed: true } as never);
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}) as never);
    rerender(<GameDetailModal gameId="402" onClose={() => {}} />);
    await waitFor(() => expect(api.getGameDetail).toHaveBeenCalled());
    // Still exactly one: the completed "401" detail did not authorise a fetch, and
    // "402" has not arrived.
    expect(api.getGameSignals).toHaveBeenCalledTimes(1);
    expect(api.getGameSignals).toHaveBeenCalledWith("401");
  });

  it("does not ask while the game detail is still in flight", async () => {
    // PL's effect had this bug: a `detail?.status` gate is `undefined` -- falsy --
    // on the first render, so the fetch fired for every game including completed
    // ones, and only the re-run after the detail landed suppressed it.
    mockApi();
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}) as never);
    render(<GameDetailModal gameId="401" onClose={() => {}} />);
    await waitFor(() => expect(api.getGamePlayers).toHaveBeenCalled());
    expect(api.getGameSignals).not.toHaveBeenCalled();
  });

  // --- Review findings on #36 -------------------------------------------
  //
  // These three are not new behaviour; they are the difference between what this
  // PR CLAIMED and what it shipped. Each was caught by CodeRabbit and each was
  // correct. Two of them are the recurring pattern in this repo's history: a
  // comment that states an intent the code does not implement, and a reference to
  // something that does not exist.

  it("puts the signal rows ABOVE the player projections", async () => {
    // The comment on the block says "ABOVE the projections -- the same position
    // Sports' `GameDetailModal` and PL's `FixtureModal` put them, so a fixture
    // page reads the same way in all three". The block shipped BELOW `TopCalls`.
    //
    // So the claim was false and the cross-sport consistency was the reverse of
    // what was asserted: this was the one modal of the three with the signals
    // last. A comment asserting a position is not a position, which is why this
    // reads the DOM order rather than the comment.
    //
    // ORDER is the whole assertion, so it compares document position of the two
    // rows. `compareDocumentPosition` is the DOM's own answer to "which comes
    // first" and cannot be satisfied by both being present.
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    // The projections section is gated on the out feed having landed.
    vi.mocked(api.getGameOutPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getGameDoubtfulPlayers).mockResolvedValue([] as never);
    await openModal();

    const signal = await screen.findByTestId("signal-absence");
    const calls = await screen.findByText("Player projections");
    // DOCUMENT_POSITION_FOLLOWING === the signal precedes the projections.
    expect(
      signal.compareDocumentPosition(calls) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("does not mount a section labelled by an element that does not exist", async () => {
    // `aria-labelledby={`${titleId}-signals`}` with no `id={`${titleId}-signals`}`
    // anywhere. A dangling `aria-labelledby` resolves to the empty string, so the
    // section is announced with NO name -- worse than the unlabelled section it was
    // meant to be, because it looks labelled to the code that reads it.
    //
    // The sibling sections (`-box`, `-calls`) each pair their `aria-labelledby`
    // with a real `<h3 id=...>`. This one has the attribute and not the element.
    //
    // A11y basics are not "simplified away", so this is fixed rather than waived.
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    await screen.findByTestId("signal-absence");

    // Every `aria-labelledby` on the page must name an element that exists.
    const dangling = Array.from(document.querySelectorAll("[aria-labelledby]"))
      .map((el) => el.getAttribute("aria-labelledby") || "")
      // `aria-labelledby` may list several ids; every one of them must resolve.
      .flatMap((value) => value.split(/\s+/))
      .filter((id) => id.length > 0 && !document.getElementById(id));
    expect(dangling).toEqual([]);
  });

  it("renders no section at all when every signal row is filtered out", async () => {
    // `{signals.length > 0 && ...}` guards on the RAW list. `SignalRows` drops any
    // row it cannot draw and renders NOTHING when none survive (`drawn.length === 0`
    // -> `return null`), so a nonempty list that filters to empty still mounted a
    // `<section>` with no heading and no rows inside it.
    //
    // The drop path used here is an EMPTY HEADLINE TEXT, which is the one an
    // absence row can actually take. NOT a missing `figures.projection`: that
    // raises `SignalFigureError` rather than being filtered, so it would have
    // crashed this test instead of failing the assertion -- the adapter's job is to
    // refuse to emit such a row, and the component's job is to refuse to draw it.
    // The `n >= 30` rate floor is not available as a drop either:
    // `DRAWS_A_RATE.absence_strip === false`, so it never applies here.
    //
    // Spec §2: a fixture with nothing to say renders NO rows. An empty section is
    // the gap that rule exists to prevent, and it shows as a stray band of padding.
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({
      sport: "nba",
      id: "401",
      signals: [{ ...ABSENCE, headline: { text: "   ", figures: { projection: 31.5 } } }],
    } as never);
    await openModal();

    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalled());
    // Settled, and nothing on the page claims to be a signal section.
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    const named = Array.from(document.querySelectorAll("[aria-labelledby]")).map((el) =>
      el.getAttribute("aria-labelledby"),
    );
    expect(named.filter((v) => v && v.includes("-signals"))).toEqual([]);
  });
});
