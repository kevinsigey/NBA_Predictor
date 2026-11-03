import { useEffect, useRef, useState } from "react";
import { api, type TrackRecord, type TrackRecordPick, type VsMarket } from "../api/client";
import {
  CalibrationBlock,
  EmptyState,
  ErrorState,
  HowCounted,
  RecordBlock,
  RecordHeadline,
  RecentPicks,
  Skeleton,
  StatTable,
  WeekStrip,
  kickoff,
  marketRank,
  momentFor,
} from "../predictor-ui";
import { labelFor, toRecordModel } from "../lib/recordModel";
import { Disagreements } from "./trackRecordViews";

const STAT_LABEL: Record<string, string> = { points: "Points", rebounds: "Rebounds", assists: "Assists", threes: "Threes", steals: "Steals", blocks: "Blocks" };

const n0 = (x: number) => x.toLocaleString("en-US");

/** Rows per page of the every-pick table: enough to read, small enough to fetch on a phone. */
const PICKS_PAGE = 200;

/**
 * The track record, led by the only number that cannot flatter the model: picks
 * made before tip-off, counted once per game and market. Everything the model
 * "got right" after the game started is a rebuilt backtest, and lives in its own
 * block that never feeds the headline.
 *
 * The page is the shared Track Record kit over NBA's adapter (`toRecordModel`);
 * only the player-projection block, the comparison-with-the-odds block and the
 * every-pick table are NBA's own.
 */
export default function TrackRecordPanel() {
  const [rows, setRows] = useState<TrackRecord[] | null>(null);
  // The every-pick table is tens of thousands of rows, so it is neither shipped
  // with the record nor built until opened: the rows are paged from
  // /hub/track-record/picks, a page at a time, once the fold first opens.
  const [picksOpen, setPicksOpen] = useState(false);
  const [picks, setPicks] = useState<TrackRecordPick[] | null>(null);
  const [picksTotal, setPicksTotal] = useState(0);
  const [picksLoading, setPicksLoading] = useState(false);
  // Whether page one of the disclosure has been requested for the current
  // record load; the gate that keeps the toggle, the effect below and a
  // retry from racing each other into duplicate fetches.
  const picksRequested = useRef(false);
  const [vsMarket, setVsMarket] = useState<VsMarket | null>(null);
  // The counted moneyline rows for exactly the games the odds block lists,
  // fetched when the comparison lands (the summary no longer inlines them).
  const [disagreementPicks, setDisagreementPicks] = useState<TrackRecordPick[] | null>(null);
  const [asOf, setAsOf] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const [vsError, setVsError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    setVsError(false);
    setRows(null);
    setVsMarket(null);
    setDisagreementPicks(null);
    setPicks(null);
    setPicksTotal(0);
    picksRequested.current = false;
    setAsOf(null);
    // Two requests, each with its own failure: a broken comparison block must
    // not take the record above it down with it.
    api.getTrackRecord().then(setRows).catch(() => setError(true));
    api.getVsMarket().then(setVsMarket).catch(() => setVsError(true));
    // The as-of stamp fails silently: the record without a stamp is honest,
    // a stamp that blocks the page is not.
    api.getSnapshotMeta()
      .then((meta) => { if (meta.generated_at) setAsOf(meta.generated_at); })
      .catch(() => {});
  }, [reloadKey]);

  useEffect(() => {
    if (vsMarket === null) return;
    const ids = vsMarket.disagreement_game_ids;
    if (ids.length === 0) {
      setDisagreementPicks([]);
      return;
    }
    let cancelled = false;
    api.getTrackRecordPicks({ market: "h2h", gameIds: ids, limit: ids.length })
      .then((page) => { if (!cancelled) setDisagreementPicks(page.picks); })
      .catch(() => { if (!cancelled) setDisagreementPicks([]); });
    return () => { cancelled = true; };
  }, [vsMarket]);

  const loadPicks = (offset: number) => {
    setPicksLoading(true);
    api.getTrackRecordPicks({ limit: PICKS_PAGE, offset })
      .then((page) => {
        setPicks((prev) => (offset === 0 || prev === null ? page.picks : [...prev, ...page.picks]));
        setPicksTotal(page.total);
      })
      .catch(() => { if (offset === 0) setPicks([]); })
      .finally(() => setPicksLoading(false));
  };

  // Page one loads when the fold first opens (and again after a reload reset
  // the rows while the fold stayed open).
  useEffect(() => {
    if (!picksOpen) picksRequested.current = false;
    if (!picksOpen || picks !== null || picksRequested.current) return;
    picksRequested.current = true;
    loadPicks(0);
    // loadPicks is stable enough for this guard's purpose; the ref is the gate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picksOpen, picks]);

  if (error) return <ErrorState message="We couldn't load the track record. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  if (rows === null) return <Skeleton label="Loading track record…" />;
  if (rows.length === 0) return <EmptyState message="No tracked predictions yet. Picks are graded here once their games are final." />;

  const model = toRecordModel(rows);
  const moment = momentFor("nba");

  const props = rows.find((r) => r.market === "player_props" && !r.settled && (r.per_stat?.length ?? 0) > 0);
  // Props are judged by error, shown in their own block; "no results rule yet" would be the wrong thing to say about them.
  const unsettled = rows.filter((r) => !r.settled && r !== props);

  // Backtest: everything counted but made at or after tip-off. Hit rate only: these rows carry no stated confidence.
  const backtest = rows
    .filter((r) => (r.n_rebuilt ?? 0) > 0 && r.pre_tip)
    .sort((a, b) => marketRank(a.market) - marketRank(b.market))
    .map((r) => ({
      key: r.market,
      label: labelFor(r.market),
      n: r.n_rebuilt ?? 0,
      correct: r.correct_predictions - r.pre_tip!.correct_predictions,
    }));

  // NBA's own blocks: the every-pick table (it also shows counted vs rebuilt,
  // which the view model does not carry -- its rows are paged in on open) and
  // the pushes/unsettled disclosure.
  const pushes = rows.reduce((n, r) => n + (r.n_push ?? 0), 0);

  return (
    <div className="flex flex-col gap-8">
      {/* Calibration chart as hero: full width under the tabs, above the record. */}
      <CalibrationBlock series={model.calibration ?? []} markets={model.markets} />

      <RecordBlock id="tr-record" title="Picks made before tip-off">
        {model.markets.some((m) => m.preKickoff) ? (
          <RecordHeadline markets={model.markets} mode={model.headline} lead={model.lead} moment={moment} />
        ) : (
          <p className="text-sm text-pr-text-dim">No pick made before tip-off has been graded yet.</p>
        )}
      </RecordBlock>

      <RecordBlock id="tr-week" title="Week by week">
        <WeekStrip weeks={model.weeks ?? []} />
      </RecordBlock>

      <div className="grid gap-8 lg:grid-cols-2">
        <RecordBlock id="tr-market" title="Against the odds">
          {vsError ? (
            <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
              <span>We couldn&apos;t load the comparison with the odds.</span>
              <button type="button" onClick={() => setReloadKey((k) => k + 1)} className="rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text hover:border-pr-accent">
                Try again
              </button>
            </div>
          ) : vsMarket === null || (vsMarket.n > 0 && disagreementPicks === null) ? (
            <Skeleton label="Loading comparison…" />
          ) : vsMarket.n === 0 ? (
            <p className="text-sm text-pr-text-dim">No finished game has a pre-tip price to compare with yet.</p>
          ) : (
            <Disagreements
              gameIds={vsMarket.disagreement_game_ids}
              picks={disagreementPicks ?? []}
              n={vsMarket.disagreement_n}
              hitRate={vsMarket.disagreement_hit_rate}
            />
          )}
        </RecordBlock>
        <RecordBlock id="tr-recent" title="Latest counted picks">
          <RecentPicks picks={model.recent ?? []} moment={moment} />
        </RecordBlock>
      </div>

      {props && (
        <RecordBlock id="tr-props" title="Player projections: how far off">
          <p className="max-w-prose text-sm text-pr-text-dim">
            Props have no win/lose result, so they are judged by the average miss between the projection and what the player
            did, over {n0(props.total_predictions)} projections that have since been played.
          </p>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {props.per_stat!.map((s) => (
              <li key={s.stat} data-testid={`prop-stat-${s.stat}`} className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3">
                <p className="text-xs text-pr-text-dim">{STAT_LABEL[s.stat] ?? s.stat}</p>
                <p className="font-pr-display text-lg font-semibold tabular-nums text-pr-text">
                  off by {s.mae.toFixed(1)} <span className="text-pr-text-faint">on average</span>
                </p>
                <p className="text-xs text-pr-text-faint">
                  {s.mean_signed_error > 0.05 ? `runs high by ${s.mean_signed_error.toFixed(1)}` : s.mean_signed_error < -0.05 ? `runs low by ${Math.abs(s.mean_signed_error).toFixed(1)}` : "no lean"} · {n0(s.n)} games
                </p>
              </li>
            ))}
          </ul>
        </RecordBlock>
      )}

      {backtest.length > 0 && (
        <RecordBlock id="tr-backtest" title="Backtest, not counted">
          <p className="max-w-prose text-sm text-pr-text-dim">
            Picks rebuilt at or after tip-off by a later run of the model. They show what the model would have said, not
            what it called in time.
          </p>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {backtest.map((b) => (
              <li key={b.key} data-testid={`backtest-${b.key}`} className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3">
                <p className="text-xs text-pr-text-dim">{b.label}</p>
                <p className="font-pr-display text-lg font-semibold tabular-nums text-pr-text">
                  {n0(b.correct)} <span className="text-pr-text-faint">of</span> {n0(b.n)}
                </p>
              </li>
            ))}
          </ul>
        </RecordBlock>
      )}

      <details
        className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm"
        onToggle={(e) => setPicksOpen((e.currentTarget as HTMLDetailsElement).open)}
      >
        <summary className="cursor-pointer font-semibold text-pr-text">Every recorded pick</summary>
        <div className="mt-3">
          {!picksOpen ? null : picks === null ? (
            <p className="text-pr-text-dim">Loading picks…</p>
          ) : picks.length === 0 ? (
            <p className="text-pr-text-dim">No pick-level data in this response.</p>
          ) : (
            <>
            <StatTable
              rows={[...picks].sort((a, b) => b.created_at.localeCompare(a.created_at))}
              rowKey={(r) => `${r.market}-${r.game_id}-${r.created_at}`}
              caption="Every recorded pick"
              columns={[
                { key: "gameday", label: "Game", value: (r) => r.gameday ?? r.created_at, render: (r) => (r.gameday ? kickoff(r.gameday) : <span className="text-pr-text-dim">Unknown</span>) },
                { key: "market", label: "Market", value: (r) => r.market, render: (r) => labelFor(r.market) },
                { key: "pick", label: "Pick", value: (r) => r.pick },
                { key: "actual", label: "Actual", value: (r) => r.actual },
                {
                  key: "hit",
                  label: "Result",
                  value: (r) => (r.hit === null ? 2 : r.hit ? 1 : 0),
                  render: (r) => (r.hit === null ? <span className="text-pr-text-faint">Push</span> : r.hit ? <span className="font-semibold text-pr-win">Right</span> : <span className="font-semibold text-pr-loss">Wrong</span>),
                },
                {
                  key: "timing",
                  label: "Made",
                  value: (r) => (r.made_before_tip ? 1 : r.counted ? 0 : -1),
                  render: (r) => (r.made_before_tip ? "Before tip-off" : r.counted ? "After tip-off" : "Rebuilt (not counted)"),
                },
              ]}
            />
            {picks.length < picksTotal && (
              <button
                type="button"
                onClick={() => loadPicks(picks.length)}
                disabled={picksLoading}
                className="mt-3 rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text hover:border-pr-accent disabled:opacity-50"
              >
                {picksLoading ? "Loading…" : `Show more picks (${n0(picks.length)} of ${n0(picksTotal)})`}
              </button>
            )}
            </>
          )}
        </div>
      </details>

      <HowCounted moment={moment} headline={model.headline} asOf={asOf}>
        {pushes > 0 && <p>{n0(pushes)} pick{pushes === 1 ? "" : "s"} landed on the line and {pushes === 1 ? "is" : "are"} left out of every rate, not scored as a miss.</p>}
        {unsettled.length > 0 && (
          <p>
            {unsettled.map((r) => labelFor(r.market)).join(", ")} {unsettled.length === 1 ? "has" : "have"} no results rule yet, so no rate is shown.
          </p>
        )}
        {vsMarket && Object.keys(vsMarket.method).length > 0 && (
          <dl className="mt-1 flex flex-col gap-2">
            {Object.entries(vsMarket.method).map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs font-semibold uppercase tracking-wide text-pr-text-faint">{k.replace(/_/g, " ")}</dt>
                <dd className="text-xs leading-relaxed">{String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </HowCounted>
    </div>
  );
}
