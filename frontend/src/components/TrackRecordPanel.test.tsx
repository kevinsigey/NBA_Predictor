import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TrackRecordPanel from "./TrackRecordPanel";
import { wilsonInterval } from "../predictor-ui";
import { api, type PropGrading, type TrackRecord, type TrackRecordPick, type TrackRecordWeek, type VsMarket } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getTrackRecord: vi.fn(), getVsMarket: vi.fn(), getTrackRecordPicks: vi.fn(), getSnapshotMeta: vi.fn(), getPropGrading: vi.fn() },
}));

// The grading payload's aggregates: the prop model's tracking over the
// counted pre-tip record. Points ran high (bias is actual minus projected,
// so a negative bias is projections above what players did); rebounds had
// no lean. Three game days of daily miss for the trend.
const GRADING = {
  bands: { points: 4.6, rebounds: 1.9 },
  counting: "", confidence_basis: "", grade_basis: "",
  latest_day: null, most_confident: [], biggest_surprises: [], n_with_baseline: 0,
  aggregates: {
    basis: "Prop-model tracking over the whole counted record.",
    from_date: "2026-10-21", to_date: "2026-11-01", n_days: 3, total: 120,
    per_stat: [
      { stat: "points", n: 60, mae: 4.3, bias: -0.31, holdout_mae: 4.6, band_n: 60,
        bands: [
          { grade: "On the number", n: 15, pct: 25 },
          { grade: "Close", n: 21, pct: 35 },
          { grade: "Off", n: 15, pct: 25 },
          { grade: "Way off", n: 9, pct: 15 },
        ] },
      { stat: "rebounds", n: 60, mae: 1.9, bias: -0.02, holdout_mae: 1.9, band_n: 60,
        bands: [
          { grade: "On the number", n: 18, pct: 30 },
          { grade: "Close", n: 18, pct: 30 },
          { grade: "Off", n: 15, pct: 25 },
          { grade: "Way off", n: 9, pct: 15 },
        ] },
    ],
    daily: [
      { date: "2026-10-21", stats: { points: { n: 20, mae: 4.9 }, rebounds: { n: 20, mae: 2.1 } } },
      { date: "2026-10-28", stats: { points: { n: 20, mae: 4.4 }, rebounds: { n: 20, mae: 1.8 } } },
      { date: "2026-11-01", stats: { points: { n: 20, mae: 3.9 }, rebounds: { n: 20, mae: 1.9 } } },
    ],
  },
} as unknown as PropGrading;

const wk = (week_start: string, n: number, correct: number, tracked = true): TrackRecordWeek => ({
  week_start, n, correct, hit_rate: n ? correct / n : null, tracked,
});
const pick = (over: Partial<TrackRecordPick>): TrackRecordPick => ({
  game_id: "g1", market: "h2h", pick: "BOS", actual: "BOS", hit: true, made_before_tip: true,
  created_at: "2026-10-05T03:00:00Z", counted: true, gameday: "2026-10-08", point: null, ...over,
});
type Row = TrackRecord & { confidence_buckets?: unknown };
const row = (over: Partial<Row> & { market: string }): Row => ({
  total_predictions: 0, correct_predictions: 0, hit_rate: null, n_rebuilt: 0, n_push: 0, settled: true, weekly: [], ...over,
});
const vs = (over: Partial<VsMarket> = {}): VsMarket => ({
  market: "h2h", n: 10, mean_model_probability: 0.6, mean_market_probability: 0.58, mean_edge_points: 2,
  disagreement_n: 2, disagreement_hit_rate: 0.5, disagreement_game_ids: ["g1", "g2"], weekly: [],
  scope: { population: "x", weekly_from: null, weekly_through: null, n_games_total: 10, n_games_in_weekly: 10, n_games_outside_weekly: 0 },
  method: {}, ...over,
});

// The recorded moneyline rows: two counted pre-tip picks and one rerun that
// lost the earliest-pick contest. The summary ships the counted pre-tip pair
// as `recent_picks`; all three live behind the picks endpoint.
const ALL_PICKS: TrackRecordPick[] = [
  pick({ game_id: "g1" }),
  pick({ game_id: "g2", hit: false, pick: "ATL", actual: "MEM" }),
  pick({ game_id: "old", counted: false }),
];

// The live shape: the winner pick is mostly rebuilt after tip-off, with a small counted slice.
const LIVE: Row[] = [
  row({
    market: "game_outcome", total_predictions: 1389, correct_predictions: 1182, hit_rate: 0.851, n_rebuilt: 1365,
    pre_tip: { total_predictions: 24, correct_predictions: 11, hit_rate: 0.458, weekly: [wk("2026-09-21", 0, 0, false), wk("2026-09-28", 3, 2), wk("2026-10-05", 21, 9)] },
    weekly: [],
  }),
  row({
    market: "h2h", total_predictions: 10, correct_predictions: 5, hit_rate: 0.5, n_rebuilt: 0,
    pre_tip: { total_predictions: 10, correct_predictions: 5, hit_rate: 0.5 },
    recent_picks: ALL_PICKS.filter((p) => p.counted),
    confidence_buckets: [
      { bucket: "50-60%", total_predictions: 4, correct_predictions: 2, hit_rate: 0.5 },
      { bucket: "60-70%", total_predictions: 3, correct_predictions: 2, hit_rate: 0.667 },
      { bucket: "70%+", total_predictions: 3, correct_predictions: 1, hit_rate: 0.333 },
    ],
  }),
  row({ market: "player_props", settled: false, total_predictions: 121148,
        per_stat: [{ stat: "points", n: 20000, mae: 4.8, mean_signed_error: 0.31 }, { stat: "rebounds", n: 20000, mae: 1.9, mean_signed_error: -0.02 }] } as Parameters<typeof row>[0]),
];

beforeEach(() => {
  vi.mocked(api.getTrackRecord).mockResolvedValue(LIVE as TrackRecord[]);
  vi.mocked(api.getVsMarket).mockResolvedValue(vs());
  vi.mocked(api.getPropGrading).mockResolvedValue(GRADING);
  // No stamp by default: the as-of line is absent, as it is when the meta
  // read fails or nothing has been written yet.
  vi.mocked(api.getSnapshotMeta).mockResolvedValue({ generated_at: null, source: "live" });
  // The picks endpoint, over the fixture rows: gameIds narrows to those
  // games (the odds block), otherwise the table's pages over all of them.
  vi.mocked(api.getTrackRecordPicks).mockImplementation(async (params = {}) => {
    const pool = params.gameIds ? ALL_PICKS.filter((p) => params.gameIds!.includes(p.game_id)) : ALL_PICKS;
    const offset = params.offset ?? 0;
    const limit = params.limit ?? pool.length;
    return { total: pool.length, limit, offset, picks: pool.slice(offset, offset + limit) };
  });
});

describe("TrackRecordPanel", () => {
  it("leads with the counted pre-tip record, not the rebuilt 85%", async () => {
    render(<TrackRecordPanel />);
    const lead = await screen.findByTestId("tally-game_outcome");
    expect(lead).toHaveTextContent("11 of 24");
    // The blended 85% (counted + rebuilt) is shown nowhere: the headline is the pre-tip slice only.
    expect(document.body).not.toHaveTextContent("85%");
    expect(lead).toHaveTextContent("46%");
  });

  it("says plainly that a small sample is early", async () => {
    render(<TrackRecordPanel />);
    expect(await screen.findByTestId("early-sample")).toHaveTextContent("24 counted picks");
  });

  it("keeps rebuilt-after-tip rows in a separate block that never feeds the headline", async () => {
    render(<TrackRecordPanel />);
    const block = await screen.findByTestId("backtest-game_outcome");
    expect(block).toHaveTextContent("1,171 of 1,365");
    expect(screen.getByRole("heading", { name: /backtest, not counted/i })).toBeInTheDocument();
  });

  it("draws one cell per tracked week and starts at the first graded week", async () => {
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getAllByTestId("week-cell")).toHaveLength(2));
  });

  it("plots a calibration point per graded bucket", async () => {
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getAllByTestId("calibration-point")).toHaveLength(3));
  });

  it("lists the disagreement games with how each ended, and only counted picks", async () => {
    render(<TrackRecordPanel />);
    const rows = await screen.findAllByTestId("disagreement");
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent("ATL");
    expect(rows[1]).toHaveTextContent("Wrong");
  });

  it("lists the latest counted picks and leaves out the one that lost the earliest-pick contest", async () => {
    render(<TrackRecordPanel />);
    const rows = await screen.findAllByTestId("recent-pick");
    expect(rows).toHaveLength(2);
  });

  it("tracks the prop model from the grading aggregates, never as a hit rate", async () => {
    render(<TrackRecordPanel />);
    const points = await screen.findByTestId("prop-stat-points");
    // The percentage leads: On the number (25%) + Close (35%) landed within
    // the model's own error.
    expect(points).toHaveTextContent("60%");
    expect(points).toHaveTextContent("within its own error");
    expect(points).toHaveTextContent("off by 4.3 on average");
    expect(points).toHaveTextContent("holdout error 4.6, tracking inside it");
    // Bias is actual minus projected: -0.31 means the projections ran high.
    expect(points).toHaveTextContent("runs high by 0.3");
    expect(points).toHaveTextContent("On the number 25%");
    expect(points).toHaveTextContent("Way off 15%");
    expect(screen.getByTestId("prop-stat-rebounds")).toHaveTextContent("no lean");
    expect(screen.queryByText(/no results rule yet/i)).toBeNull();
  });

  it("keeps a broken grading read from taking the record down", async () => {
    vi.mocked(api.getPropGrading).mockRejectedValue(new Error("x"));
    render(<TrackRecordPanel />);
    expect(await screen.findByTestId("tally-game_outcome")).toBeInTheDocument();
    expect(screen.queryByTestId("prop-stat-points")).toBeNull();
  });

  it("never prints a rate for a market the backend cannot judge", async () => {
    render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    expect(screen.queryByTestId("tally-player_props")).toBeNull();
  });

  it("falls back to a plain message when nothing counted has graded", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([row({ market: "h2h" })] as TrackRecord[]);
    render(<TrackRecordPanel />);
    expect(await screen.findByText(/no pick made before tip-off has been graded/i)).toBeInTheDocument();
  });

  it("keeps a broken odds comparison from taking the record down", async () => {
    vi.mocked(api.getVsMarket).mockRejectedValue(new Error("x"));
    render(<TrackRecordPanel />);
    expect(await screen.findByTestId("tally-game_outcome")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(/comparison with the odds/i);
  });

  it("shows the error state with a retry when the record cannot load", async () => {
    vi.mocked(api.getTrackRecord).mockRejectedValue(new Error("x"));
    render(<TrackRecordPanel />);
    expect(await screen.findByText(/couldn't load the track record/i)).toBeInTheDocument();
  });

  it("leads each headline line with the percentage, the tally under it", async () => {
    render(<TrackRecordPanel />);
    const lead = await screen.findByTestId("tally-game_outcome");
    const rate = within(lead).getByTestId("rate-game_outcome");
    expect(rate).toHaveTextContent("46%");
    expect(lead).toHaveTextContent("11 of 24");
  });

  it("stamps the record with when its data was written", async () => {
    vi.mocked(api.getSnapshotMeta).mockResolvedValue({ generated_at: "2026-11-02T04:15:00Z", source: "live" });
    render(<TrackRecordPanel />);
    expect(await screen.findByText(/record as of/i)).toHaveTextContent("Nov 2, 2026");
  });

  it("fetches the every-pick rows only when the fold opens, never with the record", async () => {
    render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    await screen.findAllByTestId("disagreement");
    const tableCalls = () => vi.mocked(api.getTrackRecordPicks).mock.calls.filter((call) => !call[0]?.gameIds);
    expect(tableCalls()).toHaveLength(0);
    await userEvent.click(screen.getByText("Every recorded pick"));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(tableCalls()).toEqual([[{ limit: 200, offset: 0 }]]);
  });

  it("pages the every-pick table with a show-more button until the total is reached", async () => {
    const four = [0, 1, 2, 3].map((i) => pick({ game_id: `p${i}`, created_at: `2026-10-0${5 + i}T03:00:00Z` }));
    vi.mocked(api.getTrackRecordPicks).mockImplementation(async (params = {}) => {
      if (params.gameIds) return { total: 0, limit: 0, offset: 0, picks: [] };
      const offset = params.offset ?? 0;
      return { total: four.length, limit: 2, offset, picks: four.slice(offset, offset + 2) };
    });
    render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    await userEvent.click(screen.getByText("Every recorded pick"));
    const more = await screen.findByRole("button", { name: /show more picks \(2 of 4\)/i });
    expect(within(screen.getByRole("table")).getAllByRole("row")).toHaveLength(3); // header + 2
    await userEvent.click(more);
    await waitFor(() => expect(screen.queryByRole("button", { name: /show more picks/i })).toBeNull());
    expect(within(screen.getByRole("table")).getAllByRole("row")).toHaveLength(5); // header + 4
  });
});

describe("wilson", () => {
  it("is a wide range for a small sample and null for none", () => {
    const [lo, hi] = wilsonInterval(11, 24)!;
    expect(lo).toBeGreaterThan(0.25);
    expect(lo).toBeLessThan(0.3);
    expect(hi).toBeGreaterThan(0.64);
    expect(hi).toBeLessThan(0.68);
    expect(wilsonInterval(0, 0)).toBeNull();
  });
});

describe("parity with the page before the kit", () => {
  // The exact text the panel rendered before it moved onto the shared kit,
  // captured from the fixture above. Every number, label and section must stay
  // put; the only differences allowed are the corrections applied below.
  const BEFORE = "Does 70% mean 70%?0%50%100%50%60%70%80%50-60% confidence: 2 of 4 right60-70% confidence: 2 of 3 right70%+ confidence: 1 of 3 rightAcross: how sure the model said it was. Up: how often it was right. On the dashed line is honest; bigger dots rest on more picks; faint dots rest on fewer than 5.Counted record: picks made before tip-offWinner pick11 of 2446%Could plausibly be 28%–65%Early sample: 24 counted picks is not enough to call this model good or bad. The bar shows the range the true rate could sit in; the thin line is 50%.Moneyline5 of 1050%Could plausibly be 24%–76%Week by weekMon 28 Sep: 2 of 3Mon 5 Oct: 9 of 2150% or betterunder 50%Taller = more picks that weekAgainst the odds1 of 250%when the model backed the side the odds did not.Thu 8 OctBOS · was BOSRightThu 8 OctATL · was MEMWrongLatest counted picksThu 8 OctBOS · Moneyline · was BOSRightThu 8 OctATL · Moneyline · was MEMWrongPlayer projections: how far offProps have no win/lose result, so they are judged by the average miss between the projection and what the player did, over 121,148 projections that have since been played.Pointsoff by 4.8 on averageruns high by 0.3 · 20,000 gamesReboundsoff by 1.9 on averageno lean · 20,000 gamesBacktest, not countedPicks rebuilt at or after tip-off by a later run of the model. They show what the model would have said, not what it called in time, so they never enter the record above.Winner pick1,171 of 1,365Every recorded pickEvery recorded pickGameMarketPickActualResultMadeThu 8 OctMoneylineBOSBOSRightBefore tip-offThu 8 OctMoneylineATLMEMWrongBefore tip-offThu 8 OctMoneylineBOSBOSRightBefore tip-offHow this is countedOne counted pick per game and market: the earliest recorded. A re-run of the model never replaces it and is never counted twice. The counted record above only includes picks made before tip-off.";

  // Each change, and why it is the only thing that may differ.
  const CORRECTIONS: [string, string][] = [
    // The headline leads with the percentage now, the tally under it
    // (percentage-forward headline, 2026-11-01): "11 of 24" then "46%" became
    // "46%" then "11 of 24". Same figures, read in the other order.
    ["11 of 2446%", "46%11 of 24"],
    ["5 of 1050%", "50%5 of 10"],
    // The headline no longer calls the pre-tip subset the whole "counted record".
    ["Counted record: picks made before tip-off", "Picks made before tip-off"],
    // Rebuilt picks do feed a count elsewhere; it is only the headline that excludes them.
    [", so they never enter the record above", ""],
    // The old sentence named the pre-tip subset "the counted record".
    [
      "The counted record above only includes picks made before tip-off.",
      "The record above shows every pick made before tip-off. Picks rebuilt after tip-off are listed separately, below.",
    ],
    // The shared "How this is counted" states the counting rule and the meaning of
    // the headline as two paragraphs, so the space that joined them in the old
    // single paragraph is gone. Same words, one line break.
    ["never counted twice. The record above", "never counted twice.The record above"],
    // The props block was replaced outright (2026-11-02, the grading
    // relocation): the old block averaged every played projection whatever
    // its timing; the new one is the grading payload's tracking over the
    // counted pre-tip record only -- the share within the model's own error
    // leading, the average miss against the holdout reference, the grade
    // distribution and the daily trend. The fixture above (GRADING) is what
    // renders the new text.
    [
      "Player projections: how far offProps have no win/lose result, so they are judged by the average miss between the projection and what the player did, over 121,148 projections that have since been played.Pointsoff by 4.8 on averageruns high by 0.3 · 20,000 gamesReboundsoff by 1.9 on averageno lean · 20,000 games",
      "Player projections: how the model is trackingProp-model tracking over the whole counted record.120 counted lines, Oct 21, 2026 – Nov 1, 2026, over 3 game days.Points60% within its own erroroff by 4.3 on average · holdout error 4.6, tracking inside itruns high by 0.3 · 60 countedOn the number 25% · Close 35% · Off 25% · Way off 15%Daily miss, last 3 game days · dashed line is the holdout errorRebounds60% within its own erroroff by 1.9 on average · holdout error 1.9, tracking inside itno lean · 60 countedOn the number 30% · Close 30% · Off 25% · Way off 15%Daily miss, last 3 game days · dashed line is the holdout error",
    ],
  ];

  it("renders everything as before, except the corrections", async () => {
    const { container } = render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    await userEvent.click(screen.getByText("Every recorded pick"));
    await screen.findAllByRole("table");
    const after = CORRECTIONS.reduce((text, [from, to]) => text.split(from).join(to), BEFORE);
    expect(container.textContent).toBe(after);
    expect(container.textContent).not.toBe(BEFORE);
  });

  it("builds the every-pick table only when it is opened (it is ~1,400 rows on the live page)", async () => {
    render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    expect(screen.queryByRole("table")).toBeNull();
    await userEvent.click(screen.getByText("Every recorded pick"));
    expect(await screen.findByRole("table")).toBeInTheDocument();
  });

  it("made every correction", async () => {
    const { container } = render(<TrackRecordPanel />);
    await screen.findByTestId("tally-game_outcome");
    const text = container.textContent ?? "";
    for (const [from] of CORRECTIONS) expect(text, from).not.toContain(from);
    expect(text).toContain("not what it called in time.");
    expect(text).toContain("Picks rebuilt after tip-off are listed separately, below.");
  });
});
