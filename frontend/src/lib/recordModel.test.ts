import { describe, expect, it } from "vitest";
import { toRecordModel } from "./recordModel";
import type { TrackRecord, TrackRecordPick } from "../api/client";

const pick = (over: Partial<TrackRecordPick>): TrackRecordPick => ({
  game_id: "g1", market: "h2h", pick: "BOS", actual: "BOS", hit: true, made_before_tip: true,
  created_at: "2026-10-05T03:00:00Z", counted: true, gameday: "2026-10-08", point: null, ...over,
});

const rows: (TrackRecord & { confidence_buckets?: unknown })[] = [
  {
    market: "game_outcome", total_predictions: 1389, correct_predictions: 1182, hit_rate: 0.851, n_rebuilt: 1365,
    pre_tip: { total_predictions: 24, correct_predictions: 11, hit_rate: 0.458, weekly: [{ week_start: "2026-09-21", n: 0, correct: 0, hit_rate: null, tracked: false }, { week_start: "2026-09-28", n: 3, correct: 2, hit_rate: 0.667, tracked: true }] },
    settled: true,
  },
  {
    market: "h2h", total_predictions: 10, correct_predictions: 5, hit_rate: 0.5, n_rebuilt: 0, settled: true,
    pre_tip: { total_predictions: 10, correct_predictions: 5, hit_rate: 0.5 },
    recent_picks: [pick({ game_id: "older", gameday: "2026-10-01" }), pick({ game_id: "newer", gameday: "2026-10-09" }), pick({ game_id: "rerun", counted: false })],
    confidence_buckets: [{ bucket: "50-60%", total_predictions: 4, correct_predictions: 2, hit_rate: 0.5 }],
  },
  { market: "player_props", total_predictions: 121148, correct_predictions: 0, hit_rate: null, settled: false },
];

describe("toRecordModel", () => {
  const model = toRecordModel(rows as TrackRecord[]);

  it("keeps NBA's headline on the pre-tip subset, with the whole record beside it", () => {
    expect(model.headline).toBe("preKickoff");
    expect(model.lead).toBe("game_outcome");
    expect(model.markets[0]).toMatchObject({ key: "game_outcome", preKickoff: { correct: 11, n: 24 }, counted: { correct: 1182, n: 1389 } });
  });

  it("drops a market the backend cannot judge, and never prints a raw key", () => {
    expect(model.markets.map((m) => m.key)).toEqual(["game_outcome", "h2h"]);
    expect(model.markets.every((m) => !["game_outcome", "h2h", "player_props"].includes(m.label))).toBe(true);
  });

  it("carries only the lead market's pre-tip weeks, as week points", () => {
    expect(model.weeks).toEqual([
      { weekStart: "2026-09-21", n: 0, correct: 0, tracked: false },
      { weekStart: "2026-09-28", n: 3, correct: 2, tracked: true },
    ]);
  });

  it("maps confidence buckets to the kit's n/correct", () => {
    expect(model.calibration).toEqual([{ key: "h2h", label: "Moneyline", buckets: [{ bucket: "50-60%", n: 4, correct: 2 }] }]);
  });

  it("hands recent picks over counted, pre-tip and newest first", () => {
    expect(model.recent!.map((p) => p.id)).toEqual(["h2h-newer-2026-10-05T03:00:00Z", "h2h-older-2026-10-05T03:00:00Z"]);
    expect(model.recent!.every((p) => p.madeBeforeKickoff && p.label === "Moneyline")).toBe(true);
  });
});
