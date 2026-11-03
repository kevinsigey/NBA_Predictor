import { useEffect, useId, useMemo, useRef, useState } from "react";
import { api, type GameDetail, type OutPlayer, type DoubtfulPlayer, type PlayerProp, type PlayerHubRow, type MarketPrediction, type PropGameGrading, type TrackRecord, type Prediction } from "../api/client";
import { ModelRecordLine } from "./ModelRecordLine";
import TeamLogo from "./TeamLogo";
import TopCalls from "./TopCalls";
import TeamRatings from "./TeamRatings";
import Availability from "./Availability";
import PostGameReview from "./PostGameReview";
import { favourite } from "../lib/pick";
import { teamName } from "../lib/teams";
import { ErrorState, FixtureExplainer, SignalRows, Skeleton, StatusBadge, kickoff, pct, signalIsDrawn, stat, statusWords, type Signal } from "../predictor-ui";
import { panelFacts } from "../predictor-ui/lib/panelFacts";
import PlayerBoxScore from "./PlayerBoxScore";
import { signedLine } from "../lib/lines";

interface GameDetailModalProps {
  gameId: string;
  onClose: () => void;
}

interface FavoredTeam {
  team: string;
  value: number;
}

function favoredTeam(margin: number, homeTeam: string, awayTeam: string): FavoredTeam {
  return margin >= 0 ? { team: homeTeam, value: margin } : { team: awayTeam, value: -margin };
}

interface PostMatchVerdict {
  winnerCorrect: boolean;
  predictedMargin: FavoredTeam;
  actualMargin: FavoredTeam;
  marginDiff: number;
  predictedTotal: number;
  actualTotal: number;
  totalDiff: number;
}

export function computePostMatchVerdict(detail: GameDetail): PostMatchVerdict | null {
  // A pick rebuilt after tip-off is shown for reference but never judged.
  if (!detail.completed || !detail.prediction || detail.rebuilt || detail.home_pts === null || detail.away_pts === null) {
    return null;
  }
  const actualMarginValue = detail.home_pts - detail.away_pts;
  const actualTotal = detail.home_pts + detail.away_pts;
  return {
    winnerCorrect: detail.prediction.home_win_probability >= 0.5 === actualMarginValue > 0,
    predictedMargin: favoredTeam(detail.prediction.predicted_margin, detail.home_team, detail.away_team),
    actualMargin: favoredTeam(actualMarginValue, detail.home_team, detail.away_team),
    marginDiff: Math.abs(detail.prediction.predicted_margin - actualMarginValue),
    predictedTotal: detail.prediction.predicted_total,
    actualTotal,
    totalDiff: Math.abs(detail.prediction.predicted_total - actualTotal),
  };
}

/**
 * The game's own state, in one chip.
 *
 * A finished game says so in the score block underneath, so it gets no chip
 * here. A game with a score is Live; anything else is still to come.
 *
 * The rule reads the data the page already holds and never the browser's
 * clock: comparing `tip_off` to `Date.now()` would put the server's status
 * rule on the wrong side of a network boundary, and the two could disagree.
 */
function GameStatusChip({ completed, inPlay }: { completed: boolean; inPlay: boolean }) {
  if (completed) return null;
  return <StatusBadge status={inPlay ? "live" : "next"} moment="tip-off" />;
}

function FormBadge({ result }: { result: string }) {
  return (
    <span
      data-testid="form-badge"
      className={
        result === "W"
          ? "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-win)]/20 text-xs text-[var(--color-win)]"
          : "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-shotclock)]/20 text-xs text-[var(--color-shotclock)]"
      }
    >
      {result}
    </span>
  );
}

export default function GameDetailModal({ gameId, onClose }: GameDetailModalProps) {
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [players, setPlayers] = useState<PlayerProp[] | null>(null);
  // Who the availability gate removed from this game's ranking. `null` is a
  // third state on purpose, and it is not "nobody is out": while it is loading,
  // and after a failure, the ranking is not shown at all. A ranking whose
  // exclusions could not be read is a ranking nobody can vouch for, and the API
  // answers an unreadable feed with a 503 rather than an empty report precisely
  // so the two cases stay apart. Showing the top three of a list we cannot prove
  // is complete would be the dishonest reading, not the cautious one.
  const [outPlayers, setOutPlayers] = useState<OutPlayer[] | null>(null);
  // The players the gate flagged but did not remove: they stay in the ranking,
  // and the page says they are there.
  const [doubtfulPlayers, setDoubtfulPlayers] = useState<DoubtfulPlayer[] | null>(null);
  // Spec §4's signal rows. `[]` and never null once settled -- and `[]` is also
  // what every failure leaves behind, see the effect below.
  const [signals, setSignals] = useState<Signal[]>([]);
  // The rows that will ACTUALLY render, computed by the same exported predicate
  // `SignalRows` filters with. Without this the section mounted on the raw count
  // and rendered an empty band whenever every row was undrawable; see the block's
  // comment. `useMemo` because `signalIsDrawn` walks every row on every render.
  const drawnSignals = useMemo(() => signals.filter((s) => signalIsDrawn(s)), [signals]);
  // The per-game player feed carries no team, so the box score's split comes
  // from the season hub feed. Fetched separately, and allowed to fail: losing
  // the split must not take the game detail down with it.
  const [hubPlayers, setHubPlayers] = useState<PlayerHubRow[]>([]);
  // The season's winner-pick record, for the block's record strip. Fetched once
  // with the modal and allowed to fail, like the hub feed above: losing the
  // record must not take the game detail down, and a record that cannot be read
  // is better absent than wrong. `null` while loading and after a failure, so
  // the block is passed no record at all rather than an empty one.
  const [trackRecord, setTrackRecord] = useState<TrackRecord[] | null>(null);
  // This game's counted prop grading, for the post-game review. `null` while
  // loading, after a failure, and for a game with no counted props -- all
  // three are the same rendering (no review section), so they share a state.
  const [gameGrading, setGameGrading] = useState<PropGameGrading | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setDetail(null);
    setPlayers(null);
    setOutPlayers(null);
    setError(null);
    Promise.all([api.getGameDetail(gameId), api.getGamePlayers(gameId)])
      .then(([detailResult, playersResult]) => {
        setDetail(detailResult);
        setPlayers(playersResult);
      })
      .catch(() => setError("We couldn't load this game. Check your connection and try again."));
  }, [gameId, reloadKey]);

  // The signal rows (spec §4). Its own fetch, allowed to fail on its own, and
  // GATED ON `detail.game_id === gameId` -- the same shape PL's `FixtureModal` uses,
  // and for PL's reason: this modal keeps the PREVIOUS game in state until the new
  // one lands (the reset is a `setDetail(null)` inside the effect above, and an
  // effect reads the value captured by ITS OWN render), so a `!detail` or
  // `detail?.game_status` gate would read the OLD game's status to decide about the
  // new one.
  //
  // A COMPLETED game is not asked about at all. The endpoint answers `[]` for one,
  // because a completed game's roster is the one known BEFORE tip-off -- the same
  // reason `get_game_players` withholds those rows and the same reason this page's
  // own `outPlayers` list stops being information. Asking for an answer already known
  // is a request per game for nothing.
  //
  // **`completed` AND NOT "has started".** The endpoint's own rule is
  // `facts._status(game, now) in ("live", "final")`, and `completed` is only the
  // `final` half of it: a LIVE game has `completed=False` with a tip-off in the
  // past. So a live game is still asked, and the endpoint answers `[]`. That is
  // deliberate -- reproducing `_status` on the client means comparing `tip_off`
  // against the browser's clock, which duplicates the server's rule on the wrong
  // side of a network boundary for the sake of one avoided request. One extra
  // request per live game is cheaper than a status rule that can disagree with the
  // endpoint it is mirroring.
  //
  // EVERY failure is silence, and the failures are deliberately indistinguishable:
  // a 404, a network error and an honest empty list all leave `[]`. Spec §2 forbids a
  // placeholder, and a signal is an enhancement on this page -- it must never become
  // the page's error state. Sports' `GameDetailModal` does the same at the same
  // place, so this is one implementation and not one per sport.
  useEffect(() => {
    let cancelled = false;
    setSignals([]);
    if (!detail || detail.game_id !== gameId) return;
    if (detail.completed) return;
    api
      .getGameSignals(gameId)
      .then((response) => {
        if (!cancelled) setSignals(response?.signals ?? []);
      })
      // Swallow only. The `[]` at this effect's head already covers every re-run, so
      // repeating it here would be a second place to keep the same rule -- PL's
      // proved that redundant, by passing its whole suite with the clear deleted.
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [gameId, detail, reloadKey]);

  // The game's counted prop grading, for the post-game review. Its own
  // fetch, gated on `detail.game_id === gameId` for the signals effect's
  // reason (this modal keeps the previous game in state until the new one
  // lands), and asked only of a COMPLETED game: an unplayed game has no
  // counted rows to grade, and asking would be a request per game for a
  // known empty. The client call already folds every failure into null --
  // a game with no counted props and a grading that could not be read are
  // the same rendering (no review section), and the review must never
  // become the page's error state.
  useEffect(() => {
    let cancelled = false;
    setGameGrading(null);
    if (!detail || detail.game_id !== gameId) return;
    if (!detail.completed) return;
    api
      .getGamePropGrading(gameId)
      .then((grading) => {
        if (!cancelled) setGameGrading(grading);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [gameId, detail, reloadKey]);

  // Its own fetch, allowed to fail on its own. Deliberately NOT folded into the
  // Promise.all above: that one failing would take the whole game detail down,
  // and losing the exclusion list must not cost the reader the box score.
  useEffect(() => {
    let cancelled = false;
    api
      .getGameOutPlayers(gameId)
      .then((result) => {
        if (!cancelled) setOutPlayers(result);
      })
      .catch(() => {
        if (!cancelled) setOutPlayers(null);
      });
    return () => {
      cancelled = true;
    };
  }, [gameId, reloadKey]);

  // The doubtful players: still ranked, but said out loud. The same gate, the
  // same failure rule -- an unreadable feed is not an empty report.
  useEffect(() => {
    let cancelled = false;
    api
      .getGameDoubtfulPlayers(gameId)
      .then((result) => {
        if (!cancelled) setDoubtfulPlayers(result);
      })
      .catch(() => {
        if (!cancelled) setDoubtfulPlayers(null);
      });
    return () => {
      cancelled = true;
    };
  }, [gameId, reloadKey]);

  useEffect(() => {
    api.getHubPlayers()
      .then(setHubPlayers)
      .catch(() => setHubPlayers([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .getTrackRecord()
      .then((rows) => { if (!cancelled) setTrackRecord(rows); })
      .catch(() => { if (!cancelled) setTrackRecord([]); });
    return () => { cancelled = true; };
  }, []);

  // Focus moves into the dialog on open, and back to the card on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => opener?.focus?.();
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const verdict = detail ? computePostMatchVerdict(detail) : null;
  // The ratings: what each side is, and where that puts it in the league --
  // the same figures the summary reads.
  //
  // **Where they sit depends on the game.** Before a game they are under the
  // pick, because they are what the pick is made of. After it they are under
  // the verdict, because then they are the form the game was won or lost on.
  const ratingsBlock = detail?.ratings ? (
    <div className="mb-5 border-b border-[var(--color-line)] pb-5">
      <TeamRatings ratings={detail.ratings} awayTeam={detail.away_team} homeTeam={detail.home_team} />
    </div>
  ) : null;
  // The two teams in this game. ESPN's report covers the whole league, so a
  // player who is playing for neither of them is not this game's news. The
  // API already filters its feeds; this is the page's own rule, enforced here
  // so a stale or partly-applied feed cannot put a third team's player on a
  // game page.
  const outForGame = useMemo(() => {
    if (!detail) return [];
    const teams = [detail.home_team, detail.away_team];
    return (outPlayers ?? []).filter((p) => teams.includes(p.team));
  }, [outPlayers, detail]);
  const doubtfulForGame = useMemo(() => {
    if (!detail) return [];
    const teams = [detail.home_team, detail.away_team];
    return (doubtfulPlayers ?? []).filter((p) => teams.includes(p.team));
  }, [doubtfulPlayers, detail]);
  // The flow's facts: what this modal already holds, no request. The away
  // probability is the site's own convention (`favourite` derives the pair the
  // same way for the pick it prints), and the game carries no market line --
  // 142 of 142 bundles have none -- so the flow says the pick and stops. That
  // thinness is data, not a defect in the panel.
  const finite = (x: unknown): number | undefined =>
    typeof x === "number" && Number.isFinite(x) ? x : undefined;
  const flowBundle = useMemo(() => {
    if (!detail) return null;
    const hw = finite(detail.prediction?.home_win_probability);
    const aw = hw !== undefined ? 1 - hw : undefined;
    const fav = hw !== undefined && detail.prediction ? favourite(detail.prediction, detail.home_team, detail.away_team) : null;
    const wasRight = verdict ? verdict.winnerCorrect : undefined;
    const score =
      detail.home_pts != null && detail.away_pts != null
        ? { home: detail.home_pts, away: detail.away_pts }
        : undefined;
    /* THE BARE HEADING, and why the sides are spelled two ways here.
       Measured across the three states rather than assumed:
         - pre-game  — the flow's ONLY row was the fixture's own name, so the
                       live modal showed an `MIA vs TOR` heading with nothing
                       under it, directly above the AI button.
         - finished  — two REAL sentences (the result, and the pick's
                       rightness). These are live content and they stay.
       So this is not a deletion of the flow; it is the heading being
       conditional on rows existing beneath it, expressed in the only place a
       site may express it -- the bundle it hands over.

       `FixtureFlow` reads the sides under `home_team`/`away_team` for BOTH its
       pre-game name row AND its finished result sentence, so one spelling
       cannot suppress the first without breaking the second. `bundleFacts`
       (which is what the block's verdict sentence goes through) reads
       `home_team ?? team_home`, and both spellings are part of that package's
       documented contract -- its own header names PL's `team_home` and Sports'
       `home_team` as the two shapes it exists to bridge. So:

         - ALWAYS carry `team_home`/`team_away`, which the block's
           `fullTeamName` reads to expand "TOR" into "Raptors is the pick."
           Without this the block would fall back to the bare code.
         - carry `home_team`/`away_team` ONLY when the flow has a sentence that
           needs them -- i.e. once there is a score or a result. Pre-game there
           is no such sentence, and withholding the spelling the name row reads
           is what leaves the flow empty instead of bare.

       The block's figures do not depend on either key: `panelFacts` is handed
       this site's own game object, not the bundle, so the tile and the bar are
       unchanged by which spelling is present. */
    // The one condition the pre-game name row keys on, and nothing else: only
    // `home_team`/`away_team` make it exist, and only `in-play`/`finished` rows
    // need those keys to speak. Pre-game has neither, so withholding the keys
    // is what makes that row impossible rather than merely unlikely.
    const namesForSentences = Boolean(score) || detail.completed;
    return {
      team_home: detail.home_team,
      team_away: detail.away_team,
      ...(namesForSentences
        ? { home_team: detail.home_team, away_team: detail.away_team }
        : {}),
      // The codes are this modal's own vocabulary, and the title above already
      // spells them out with the same helper. The block's verdict sentence reads
      // far better as "Celtics is the pick." than as "BOS is the pick." — the
      // shared `verdictSentence` uses these when the pick matches a side, and
      // falls back to the code when the site does not carry a full name.
      // Spelled under BOTH keys for the reason above: the block reads whichever
      // alias `fullTeamName` resolves, and the flow's finished sentences read
      // the `home_team` one.
      team_home_full: teamName(detail.home_team),
      team_away_full: teamName(detail.away_team),
      home_team_full: teamName(detail.home_team),
      away_team_full: teamName(detail.away_team),
      home_win_prob: hw,
      away_win_prob: aw,
      pick: fav
        ? { label: fav.team, prob: fav.prob, ...(typeof wasRight === "boolean" ? { was_right: wasRight } : {}) }
        : undefined,
      pick_timing: detail.rebuilt ? "rebuilt" : undefined,
      score,
      result: !score
        ? undefined
        : score.home === score.away
          ? "draw"
          : score.home > score.away
            ? "home_win"
            : "away_win",
    };
  }, [detail, verdict]);
  // The panel's figures, from the SHARED adapter: moneyline segments for the
  // bar, and nothing else, because the game carries no market line for a tile.
  // The away probability is the site's own convention -- `favourite` derives
  // the pair the same way for the pick it prints -- so the bar this draws
  // agrees with the pick above it by construction, not by coincidence.
  const panel = useMemo(() => {
    if (!detail) return { tiles: [], segments: [], legend: undefined };
    const hw = finite(detail.prediction?.home_win_probability);
    return panelFacts({
      kind: "SP",
      game: { home_team: detail.home_team, away_team: detail.away_team, spread_line: null, total_line: null },
      prediction: detail.prediction
        ? { home_win_prob: hw ?? null, away_win_prob: hw !== undefined ? 1 - hw : null }
        : null,
    });
  }, [detail]);
  // The flow's state, derived ONCE and used for both the bundle and the prop,
  // because the bare-heading rule is a rule about the PAIR: the bundle withholds
  // `home_team` when the flow has nothing to say, and the prop has to be asking
  // the flow for that same nothing. Two separate derivations of "is this
  // finished?" is how the two drifted apart in the first place -- a live game
  // (`completed: false`, score present) took the PRE-GAME branch while the
  // bundle had already handed over the team names the pre-game name row reads,
  // which put the bare heading back on screen for exactly the games that have a
  // score to say.
  //
  // `in-play` is not decoration: `FixtureFlow`'s in-play rows are the score
  // sentence and (where a line exists) how it stands, and this site's data does
  // carry a partial score on an un-finished game. Those are real sentences, so
  // a live game asks for them rather than asking for a name.
  const flowState = !detail
    ? ("pre-game" as const)
    : detail.completed
      ? ("finished" as const)
      : detail.home_pts != null && detail.away_pts != null
        ? ("in-play" as const)
        : ("pre-game" as const);
  const pickFav = detail?.prediction ? favourite(detail.prediction, detail.home_team, detail.away_team) : null;
  const marginFav = detail?.prediction ? favoredTeam(detail.prediction.predicted_margin, detail.home_team, detail.away_team) : null;
  // The win and margin numbers come from separate models. When they point at
  // different teams, or the margin rounds to nothing, don't print a
  // contradiction ("BOS to win" beside "MIA by 0.6") or "BOS by 0.0".
  const marginLabel = marginFav && pickFav && marginFav.team === pickFav.team && marginFav.value >= 0.5
    ? `${marginFav.team} by ${marginFav.value.toFixed(1)}`
    : "Toss-up";

  // Show legacy header strip only when: pre-tip game, NO cover probabilities
  // available. If ANY cover probability exists (meaning a market line exists),
  // the "Other model markets" block below is the canonical source and owns all
  // figures (margin/total ±, cover probs, σ). This satisfies the one-source rule.
  // Use falsy check to handle both null and undefined.
  const hasCoverProb = detail?.prediction && (
    detail.prediction.cover_prob_spread || detail.prediction.cover_prob_total
  );
  const showLegacyStrip = (
    detail && detail.prediction && !detail.completed && !hasCoverProb
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id={titleId} className="flex flex-wrap items-center gap-2 text-xl font-bold uppercase tracking-wide">
              {detail ? (
                <>
                  <TeamLogo team={detail.away_team} size={40} ariaHidden />
                  <span>{teamName(detail.away_team)}</span>
                  <span className="font-normal normal-case"> at </span>
                  <TeamLogo team={detail.home_team} size={40} ariaHidden />
                  <span>{teamName(detail.home_team)}</span>
                </>
              ) : (
                "Game detail"
              )}
            </h2>
            {detail && (
              <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-pr-text-dim">
                <span>{kickoff(detail.tip_off ?? detail.game_date)}</span>
                <GameStatusChip
                  completed={detail.completed}
                  inPlay={detail.home_pts !== null && detail.away_pts !== null}
                />
              </p>
            )}
          </div>
          <button ref={closeRef} aria-label="Close" onClick={onClose} className="text-xl leading-none text-[var(--color-net-dim)]">
            ×
          </button>
        </div>

        {error && <ErrorState message={error} onRetry={() => setReloadKey((k) => k + 1)} />}
        {!error && !detail && <Skeleton label="Loading…" />}

        {detail?.completed ? (
          <div className="mb-5 flex items-center justify-center gap-6 border-b border-[var(--color-line)] pb-5">
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.away_pts}</div>
              <div className="mt-1 flex items-center justify-center gap-1 text-sm text-[var(--color-net-faint)]">
                <TeamLogo team={detail.away_team} size={16} />
                {detail.away_team}
              </div>
            </div>
            <div className="text-xs uppercase text-[var(--color-net-faint)]">Final</div>
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.home_pts}</div>
              <div className="mt-1 flex items-center justify-center gap-1 text-sm text-[var(--color-net-faint)]">
                <TeamLogo team={detail.home_team} size={16} />
                {detail.home_team}
              </div>
            </div>
          </div>
        ) : (
          showLegacyStrip && (
            /* THE FIELD-BY-FIELD AUDIT, and the reason this strip is two cells
               rather than three or none.
               This used to read `53% TOR to win | TOR by 4.8 | 230.6` ABOVE the
               instant block, so the pick probability was on the page three
               times: here, in the block's `moneyline` tile, and on the bar. Per
               the one-source rule the block owns the figures, so that cell is
               GONE -- the block's tile states the probability and which side it
               is, which is what this cell was saying in two pieces.

               The other two WERE KEPT when the game carried no market line and
               the "Other model markets" block had no cover probabilities. Now
               that block draws them with ± and σ, so this legacy strip is
               hidden when cover probabilities are available -- it only renders
               as a fallback for games with no market line and no cover probs.
               Finished games never render it (the Final block owns that state). */
            <div className="mb-5 grid grid-cols-2 gap-4 border-b border-[var(--color-line)] pb-5">
              <div>
                <div className="stat-display text-2xl leading-none">{marginLabel}</div>
                <div className="mt-1 text-sm text-[var(--color-net-dim)]">Projected margin</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">{detail.prediction?.predicted_total.toFixed(1)}</div>
                <div className="mt-1 text-sm text-[var(--color-net-dim)]">Projected total points</div>
              </div>
            </div>
          )
        )}

        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={`font-pr-display font-semibold uppercase tracking-wide ${verdict.winnerCorrect ? "text-pr-win" : "text-pr-loss"}`}>
                {statusWords(verdict.winnerCorrect ? "called" : "missed")}
              </span>
              <span className="text-pr-text-dim">winner pick</span>
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted margin: {verdict.predictedMargin.team} +{verdict.predictedMargin.value.toFixed(1)} —{" "}
              Actual: {verdict.actualMargin.team} +{verdict.actualMargin.value.toFixed(1)} (off by {verdict.marginDiff.toFixed(1)})
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted total: {verdict.predictedTotal.toFixed(1)} — Actual: {verdict.actualTotal} (off by {verdict.totalDiff.toFixed(1)})
            </div>
          </div>
        )}

        {detail?.completed && ratingsBlock}

        {detail && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5">
            {/* In plain English: the flow renders from facts this modal already
                holds, with no request; the AI summary sits behind the button
                and costs nothing until a reader asks. Thin by data, not by
                defect: the game carries no market line, so the flow says the
                pick and stops, and the bar below is the model's own split.

                The block above the button carries the same figures as figures:
                timing (from `pick_timing`, so a rebuilt pick is badged rather
                than described twice), the verdict, the tiles and the record.

                FixtureExplainer renders the flow itself and adds the button —
                mounting a bare FixtureFlow alongside it produced two flows and
                no button, which is how three tests here spent a cycle failing
                to find "Get the AI summary". */}
            <FixtureExplainer
              sport="nba"
              state={flowState}
              bundle={flowBundle}
              request={() => api.explainGame(gameId)}
              // No loadContext: the kit's rank-only Matchup table would sit beside the site's own ratings table
              // (value + rank, below) and say the same thing with less. The AI read still gets the facts server-side.
              fixtureId={gameId}
              extras={{ tiles: panel.tiles, segments: panel.segments, moment: "tip-off" }}
            />
            <ModelRecordLine rows={trackRecord} format="percent" className="mt-3" />
          </div>
        )}

        {!detail?.completed && ratingsBlock}

        {detail && (detail.home_recent_form.length > 0 || detail.away_recent_form.length > 0) && (
          <div className="mb-5 flex items-center justify-between border-b border-[var(--color-line)] pb-5 text-sm">
            <div>
              <div className="mb-1 text-sm text-[var(--color-net-faint)]">{detail.away_team} form</div>
              <div className="flex gap-1">
                {detail.away_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
            <div className="text-right">
              <div className="mb-1 text-sm text-[var(--color-net-faint)]">{detail.home_team} form</div>
              <div className="flex justify-end gap-1">
                {detail.home_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Other model markets: cover chances and projected margin ±, total ±, σ */}
        {detail && detail.prediction && hasCoverProb && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm">
            <h3 className="mb-2 text-xs uppercase tracking-wide text-[var(--color-net-faint)]">
              Other model markets
            </h3>
            <div className="grid gap-2 sm:grid-cols-2">
              {detail.prediction.cover_prob_spread !== null && detail.prediction.cover_prob_spread !== undefined && detail.prediction.market_point != null && (
                <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                  <div className="text-sm text-[var(--color-net-faint)]">
                    {detail.home_team} {signedLine(detail.prediction.market_point)} · {pct(detail.prediction.cover_prob_spread)}
                  </div>
                  {detail.prediction.margin_sigma && (
                    <div className="text-sm text-[var(--color-net-dim)]">
                      σ = {detail.prediction.margin_sigma.toFixed(1)} pts
                    </div>
                  )}
                </div>
              )}
              {detail.prediction.cover_prob_total !== null && detail.prediction.cover_prob_total !== undefined && detail.prediction.market_total_point != null && (
                <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                  <div className="text-sm text-[var(--color-net-faint)]">
                    Over {detail.prediction.market_total_point.toFixed(1)} · {pct(detail.prediction.cover_prob_total)}
                  </div>
                  {detail.prediction.total_sigma && (
                    <div className="text-sm text-[var(--color-net-dim)]">
                      σ = {detail.prediction.total_sigma.toFixed(1)} pts
                    </div>
                  )}
                </div>
              )}
              <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                <div className="text-sm text-[var(--color-net-faint)]">Projected margin ±</div>
                <div className="text-lg font-pr-display font-semibold">
                  {detail.prediction.predicted_margin >= 0
                    ? `${detail.home_team} +${detail.prediction.predicted_margin.toFixed(1)}`
                    : `${detail.away_team} +${Math.abs(detail.prediction.predicted_margin).toFixed(1)}`}
                </div>
                {detail.prediction.margin_sigma && (
                  <div className="text-sm text-[var(--color-net-dim)]">
                    ±{detail.prediction.margin_sigma.toFixed(1)} pts (1 σ)
                  </div>
                )}
              </div>
              <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                <div className="text-sm text-[var(--color-net-faint)]">Projected total ±</div>
                <div className="text-lg font-pr-display font-semibold">
                  {detail.prediction.predicted_total.toFixed(1)}
                </div>
                {detail.prediction.total_sigma && (
                  <div className="text-sm text-[var(--color-net-dim)]">
                    ±{detail.prediction.total_sigma.toFixed(1)} pts (1 σ)
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Injury report line */}
        {detail && detail.injury_summary && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm">
            <div className="text-sm text-[var(--color-net-faint)]">{detail.injury_summary}</div>
          </div>
        )}

        {detail && detail.head_to_head.length > 0 && (
          <div className="mb-4">
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">Head to head</h3>
            <ul className="text-sm">
              {detail.head_to_head.map((meeting) => (
                <li key={meeting.game_id} className="flex justify-between border-b border-[var(--color-line)] py-1">
                  <span>{kickoff(meeting.game_date)}</span>
                  <span className="inline-flex items-center gap-1">
                    <TeamLogo team={meeting.away_team} size={20} />
                    {meeting.away_team} {meeting.away_pts} – {meeting.home_pts}{" "}
                    <TeamLogo team={meeting.home_team} size={20} />
                    {meeting.home_team}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Who is not playing, for these two teams. The names live here and not
            in the ranking block below, so a player is named once. */}
        {detail && (outForGame.length > 0 || doubtfulForGame.length > 0) ? (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5">
            <Availability out={outForGame} doubtful={doubtfulForGame} />
          </div>
        ) : null}

        {/* The signal rows (spec §4), ABOVE the projections -- the same position
            Sports' `GameDetailModal` and PL's `FixtureModal` put them, so a fixture
            page reads the same way in all three.

            On NBA this is the only place the out player's own projection appears at
            all: `/games/{id}/players` withholds his rows before serialising, so the
            mention under the lists below is all he gets today. This row is the first
            thing on the page that says what the model expected of him.

            **Gated on `drawnSignals`, NOT on `signals`.** `signals.length > 0` counts
            the RAW list, and `SignalRows` drops any row it cannot draw and returns
            `null` when none survive -- so the raw guard mounted a `<section>` with no
            heading and no rows inside it, which is the empty state spec §2 forbids
            ("no data, no row"). The filter is exported by the shared component
            precisely so a caller can gate on the same answer the renderer computes,
            rather than re-deriving it and getting it subtly wrong. Caught by
            CodeRabbit on #36.

            **The heading exists because `aria-labelledby` names it.** The first
            version carried `aria-labelledby={`${titleId}-signals`}` with no element
            of that id anywhere, so the reference dangled and the section was
            announced with no name at all -- worse than an unlabelled section, because
            it reads as labelled to anyone auditing the attribute. The sibling
            `-box` and `-calls` sections each pair the attribute with a real `<h3>`;
            this one now does too. Caught by CodeRabbit on #36. */}
        {drawnSignals.length > 0 && (
          <section aria-labelledby={`${titleId}-signals`} className="mt-4">
            <h3 id={`${titleId}-signals`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Signals
            </h3>
            <SignalRows signals={drawnSignals} />
          </section>
        )}


        {/* The ranked calls, as a block of their own rather than another column
            of the box score: a ranking that a reader could mistake for the full
            roster is the thing this must not be, and one list per category with
            at most three rows each cannot be read that way. Gated on the out
            feed having loaded -- see `outPlayers`. */}
        {/* The post-game review, above the player block: this game's counted
            prop lines, graded against the model's own error, in the match's
            context. It renders from the game's grading payload alone -- a
            game with no counted props has no section, whatever its box
            score holds. */}
        {detail?.completed && gameGrading && gameGrading.rows.length > 0 && (
          <section aria-labelledby={`${titleId}-review`} className="mb-5 border-b border-[var(--color-line)] pb-5">
            <h3 id={`${titleId}-review`} className="sr-only">
              Player review
            </h3>
            <PostGameReview grading={gameGrading} />
          </section>
        )}

        {players && players.length > 0 && outPlayers && (
          <section aria-labelledby={`${titleId}-calls`} className="mt-4">
            <h3 id={`${titleId}-calls`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Player projections
            </h3>
            {/* The out players are named once, in the Availability section
                above, so the block below does not name them a second time. */}
            <TopCalls props={players} out={outForGame} showOut={false} />
          </section>
        )}
        {players && players.length > 0 && !outPlayers && (
          // No "0 players out", and no empty ranking: see `outPlayers` for why
          // an unreadable feed is not an empty report.
          <p data-testid="top-calls-unavailable" className="mt-4 text-sm text-pr-text-dim">
            The model's top calls are hidden for this game: we could not read the availability report, so we can't
            confirm who is fit to be ranked.
          </p>
        )}

        {players && players.length > 0 && (
          <section aria-labelledby={`${titleId}-box`} className="mt-4">
            <h3 id={`${titleId}-box`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Projected box score
            </h3>
            {players.some((p) => p.rebuilt) && (
              <p className="mb-2 text-sm text-pr-text-dim">
                Rows marked Rebuilt were built after tip-off, by a later retrain. They are shown for reference and never
                judged.
              </p>
            )}
            <PlayerBoxScore
              playerProps={players}
              hubPlayers={hubPlayers}
              homeTeam={detail?.home_team ?? ""}
              awayTeam={detail?.away_team ?? ""}
              bands={detail?.prop_bands ?? null}
            />
          </section>
        )}

      </div>
    </div>
  );
}
