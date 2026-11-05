
import { createContextLoader, type Explanation, type Signal } from "../predictor-ui";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const EXPLAIN_BASE = `${API_BASE}/api/explain/nba`;

export interface Team {
  abbreviation: string;
  name: string;
  conference: "East" | "West";
  division: string;
}

export interface Prediction {
  home_win_probability: number;
  predicted_margin: number;
  predicted_total: number;
  // The market line the cover probability is priced against (None when no line).
  market_point?: number | null;
  market_total_point?: number | null;
  // Spread cover probability (None when no line available)
  cover_prob_spread?: number | null;
  // Total over probability (None when no line available)
  cover_prob_total?: number | null;
  // Residual sigma from the manifest, driving the above
  margin_sigma?: number | null;
  total_sigma?: number | null;
}

export interface HeadToHeadMeeting {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_pts: number | null;
  away_pts: number | null;
}

export interface Game {
  game_id: string;
  game_date: string;
  /** UTC start time; missing for games cached before it was recorded. */
  tip_off?: string | null;
  home_team: string;
  away_team: string;
  prediction: Prediction | null;
  /** The pick shown was made after tip-off: labelled, never counted. */
  rebuilt?: boolean;
  completed: boolean;
  home_pts: number | null;
  away_pts: number | null;
  /** Reported-Out players who normally play real minutes; upcoming games only. */
  key_out?: { team: string; player_name: string }[];
}

export interface MarketPrediction {
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  edge: number | null;
  bookmaker: string | null;
  american_odds: number | null;
  point: number | null;
  /** Priced after tip-off: shown, never judged. */
  rebuilt?: boolean;
}

/** `GET /api/signals/{game_id}`'s envelope. */
export interface SignalsResponse {
  sport: string;
  id: string;
  signals: Signal[];
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
  head_to_head: HeadToHeadMeeting[];
  home_recent_form: string[];
  away_recent_form: string[];
  /** The availability gate's own sentence. */
  injury_summary?: string | null;
  /** The props model's own error per stat: the band a projection has to miss
   *  by before it counts as wrong. Absent when no props manifest was read. */
  prop_bands?: Record<string, number> | null;
  /** These two teams' ratings and league ranks, read from the box-score
   *  window as of this game's own date. Absent when the league is not ranked
   *  yet -- no ratings is not a table of zeros. */
  ratings?: GameRatings | null;
}

/** One figure and its place in the league. `value` is null when nobody has the
 *  figure; `rank` is null with it, because a rank for a missing number is a
 *  position in nothing. */
export interface RatingFigure {
  value: number | null;
  rank: number | null;
}

/** One metric: what it is called, what it means, and whether more is better. */
export interface RatingMetric {
  key: string;
  label: string;
  unit: string;
  /** The metric in words, for the tooltip. Never jargon without one. */
  help: string;
  higher_is_better: boolean;
}

/** One game a team has played, for the row's detail. */
export interface TeamGame {
  game_id: string;
  game_date: string;
  opponent: string;
  home: boolean;
  /** "110–100", home score first. */
  score: string;
  result: "W" | "L" | "";
}

/** The team's next game, with what the model picked for it. */
export interface TeamNextGame {
  game_id: string;
  game_date: string;
  tip_off?: string | null;
  opponent: string;
  home: boolean;
  /** null when the model has not picked this game yet. */
  pick: string | null;
  pick_probability: number | null;
}

/** Everything the Teams page needs, in one payload. */
/** One result in a team's form: the win or the loss, when it was played, and
 *  whether it was played in the season on the page. Early in a season a team's
 *  last ten games are mostly last season's, and the dots must not pretend
 *  otherwise. */
export interface FormEntry {
  result: "W" | "L";
  game_date: string;
  this_season: boolean;
}

export interface TeamOverview {
  as_of: string;
  n_teams: number;
  caption: string;
  metrics: RatingMetric[];
  ratings: Record<string, Record<string, RatingFigure>>;
  /** The last ten results, oldest first. */
  form: Record<string, FormEntry[]>;
  last_games: Record<string, TeamGame[]>;
  next_game: Record<string, TeamNextGame | null>;
}

/** The ratings of the teams in one fixture, and the league they are ranked in. */
export interface GameRatings {
  as_of: string;
  n_teams: number;
  /** What the window was, in words. */
  caption: string;
  metrics: RatingMetric[];
  teams: Record<string, Record<string, RatingFigure>>;
}

export interface SeasonBounds {
  first_week_start: string | null;
}

export interface PlayerProp {
  player_id: string;
  player_name: string;
  /** His team now, for an upcoming game; absent for a finished one (the hub feed has it). */
  team?: string | null;
  stat: string;
  predicted_value: number;
  actual_value: number | null;
  /** Built after tip-off (a retrain backtest): shown, never judged. */
  rebuilt?: boolean;
  /**
   * In-sample mean absolute error for THIS stat -- the HEADLINE, over every
   * counted pick: one per (game, player, stat), the earliest recorded,
   * whenever it was made.
   *
   * `null` when nothing has been resolved for the stat yet, which is "never
   * measured" and not "measured at zero": the site says so in words rather than
   * drawing a ±0, which would claim the model has never missed by a tenth of a
   * point. It is a per-stat aggregate, so it must never be read as this one
   * player's error -- there is no per-player graded record for NBA props.
   *
   * The name is unchanged from before 2026-10-01, when it was the pre-tip-only
   * figure: same field, fuller record behind it.
   */
  mae?: number | null;
  /**
   * The same error estimate over the picks made BEFORE tip-off only: what the
   * model would have said on the night. Published beside `mae` because the two
   * differ, and a reader weighing the model against a book needs to know which
   * one they are reading. `null`, never 0, when no counted pick for this stat
   * was made in time.
   */
  mae_pre_tip?: number | null;
  /** The n behind `mae`. An error estimate with no count is unweighable. */
  mae_n?: number;
  /** The n behind `mae_pre_tip`, which is the figure most likely to be thin. */
  mae_n_pre_tip?: number;
  /**
   * The model's 80% conformal range for this projection, calibrated on the
   * counted record's residuals from game days strictly before this game's
   * day, with the width scaled by the size of the projection. Both null
   * when no calibration tier could serve the day: the projection then
   * stands without a range rather than beside an invented one.
   */
  interval_low?: number | null;
  interval_high?: number | null;
}

/** A player the availability gate removed from this game's ranking. Served by
 *  its own sibling route rather than wrapped onto the props response, because
 *  the props fetch is typed as `PlayerProp[]` and the two answer different
 *  questions: what the model projects, and who cannot be projected for. */
export interface OutPlayer {
  player_id: string;
  player_name: string;
  team: string;
  status: string;
  /** Where the out claim came from, in words. Never an id. */
  source: string;
  /** When it was known, in words. Never an epoch. */
  dated: string;
}

/** A player the gate flagged as doubtful. Same shape as an out player, and
 *  served by its own route because the two answer different questions: an out
 *  player leaves the ranking, a doubtful one stays in it. */
export type DoubtfulPlayer = OutPlayer;

export interface TrackRecordWeek {
  /** ISO date of the Monday the week starts on. */
  week_start: string;
  n: number;
  correct: number;
  /** null when the week graded nothing: an em dash, never a fabricated 0%. */
  hit_rate: number | null;
  tracked: boolean;
}

/** The secondary figure: the same tally over the picks made BEFORE tip-off. */
export interface TrackRecordTally {
  total_predictions: number;
  correct_predictions: number;
  /** null when nothing in the subset graded — a dash, never a 0% claim. */
  hit_rate: number | null;
  n_push?: number;
  weekly?: TrackRecordWeek[];
}

/** One recorded pick for one market, with when it was made. */
export interface TrackRecordPick {
  game_id: string;
  market: string;
  /** What the model backed, with the line it was priced at. */
  pick: string;
  /** What actually happened, in the same words. */
  actual: string;
  /** null for a push or a row with no line: nobody won it, so it is not a miss. */
  hit: boolean | null;
  /** Derived on every read from `created_at` vs tip-off, as UTC instants. */
  made_before_tip: boolean;
  created_at: string;
  /** False for a rerun that lost the earliest-pick contest. */
  counted: boolean;
  gameday: string | null;
  /** The line the pick was priced at. Present for spreads and totals, null
   *  for moneylines and markets without a line. Used to distinguish a push
   *  (hit=null but point present) from a row with no line (hit=null, no point).
   */
  point: number | null;
}

/** One counted player projection graded against the actual box score.
 *  `delta` is actual minus projected: positive means the player beat the
 *  projection. `grade` is the miss in units of the stat's holdout MAE, or
 *  null when the manifest publishes no MAE for the stat. `baseline` is the
 *  player's own rolling average (previous 10 recorded games) and
 *  `departure` how far the projection moved from it; both null when the
 *  stored record holds no prior game for him. */
export interface PropGradingRow {
  game_id: string;
  game_date: string;
  home_team: string | null;
  away_team: string | null;
  player_id: string;
  player_name: string | null;
  stat: string;
  predicted: number;
  actual: number;
  delta: number;
  grade: string | null;
  made_before_tip: boolean;
  baseline: number | null;
  departure: number | null;
  /** The model's 80% conformal range for the line, calibrated on counted
   *  game days strictly before this game's day, and whether the actual
   *  landed inside it. All three null when no calibration tier could
   *  serve the day. */
  interval_low?: number | null;
  interval_high?: number | null;
  inside_interval?: boolean | null;
}

export interface PropGradingGame {
  game_id: string;
  game_date: string;
  home_team: string | null;
  away_team: string | null;
  rows: PropGradingRow[];
}

export interface PropGradingDay {
  date: string;
  n_graded: number;
  n_pre_tip: number;
  games: PropGradingGame[];
}

/** One grade's share of a stat's banded rows in the tracking aggregates. */
export interface PropBandShare {
  grade: string;
  n: number;
  pct: number;
}

/** One stat's tracking over the whole counted record, pre-tip rows only:
 *  the average miss, the average signed miss (`bias` -- positive means the
 *  players beat the projections on average), the band shares, and the
 *  stat's holdout MAE as the reference the live figure tracks against. */
export interface PropStatAggregate {
  stat: string;
  n: number;
  mae: number;
  bias: number;
  holdout_mae: number | null;
  band_n: number;
  bands: PropBandShare[];
  /** Live coverage of the shipped 80% intervals: the percent of the
   *  stat's counted pre-tip lines whose actual landed inside the range
   *  that day's own calibration produced, over the `interval_n` lines
   *  that carried a range at all. Null when none did. */
  interval_coverage?: number | null;
  interval_n?: number;
}

/** The prop model's tracking: aggregates over the full counted record the
 *  store holds. `from_date`/`to_date`/`n_days` state the exact window;
 *  `daily` is the per-day, per-stat MAE the form trend reads. */
export interface PropAggregates {
  basis: string;
  from_date: string | null;
  to_date: string | null;
  n_days: number;
  total: number;
  per_stat: PropStatAggregate[];
  daily: { date: string; stats: Record<string, { n: number; mae: number }> }[];
  /** The shipped intervals as captioned on the record page: the method
   *  and level are constants of the build; `interval_calibration` is the
   *  backend's sentence for how the latest counted day was calibrated. */
  interval_method?: string | null;
  interval_level?: number | null;
  interval_calibration?: string | null;
}

/** The prop-grading payload for the most recent completed game day. The
 *  three note strings are the backend's wording, printed verbatim. */
export interface PropGrading {
  bands: Record<string, number> | null;
  counting: string;
  confidence_basis: string;
  grade_basis: string;
  latest_day: PropGradingDay | null;
  most_confident: PropGradingRow[];
  biggest_surprises: PropGradingRow[];
  n_with_baseline: number;
  aggregates?: PropAggregates | null;
  served_stale_seconds?: number | null;
}

/** One game's complete counted grading, served for its post-game review.
 *  Every counted row for the game (post-tip rows included and labelled),
 *  plus the game's biggest surprises. */
export interface PropGameGrading {
  game_id: string;
  game_date: string;
  home_team: string | null;
  away_team: string | null;
  n_graded: number;
  n_pre_tip: number;
  rows: PropGradingRow[];
  biggest_surprises: PropGradingRow[];
  bands: Record<string, number> | null;
  counting: string;
  grade_basis: string;
  /** How this game's 80% ranges were calibrated, in the backend's words;
   *  null when no range could be served for the game at all. */
  interval_method?: string | null;
  interval_level?: number | null;
  interval_basis?: string | null;
}

export interface TrackRecord {
  market: string;
  total_predictions: number;
  correct_predictions: number;
  /** null when nothing was graded — a dash, never a 0% claim. */
  hit_rate: number | null;
  /**
   * How stale this row is, in seconds, when a cached value is served because the
   * refresh failed. `null` means it was computed for this request — an ordinary
   * in-TTL cache hit is fresh, not stale, and saying otherwise would train a
   * reader to ignore the field.
   *
   * Set only on the cached `player_props` aggregate, and only on failure. No
   * surface renders player-props detail on this page yet, so the notice has no
   * visible home; it is here so the disclosure is not silently dropped.
   */
  served_stale_seconds?: number | null;
  /**
   * RENAMED IN MEANING, name kept: it used to count finals LEFT OUT of the
   * record. It is now the number of graded counted picks made at or after their
   * own tip-off, so `total_predictions === pre_tip.total_predictions +
   * n_rebuilt`. Subtract it to see how much of the headline is the rerun
   * rather than the night.
   */
  n_rebuilt?: number;
  /** Picks left out of the rate: the result landed on the line, or there was no line. */
  n_push?: number;
  /** Counted picks the schedule cannot date, so they appear in no week row. */
  n_unplaced?: number;
  /** The size of the pre-tip subset; equal to `pre_tip.total_predictions`. */
  n_pre_tip?: number;
  /** The pre-tip subset beside the headline. Absent only where `settled` is false. */
  pre_tip?: TrackRecordTally;
  /**
   * Every recorded pick for this market, counted or not. No longer inlined
   * in the summary response (it was ~14 MB); the rows are paged from
   * `getTrackRecordPicks`. Kept on the type for the picks page's own rows.
   */
  per_pick?: TrackRecordPick[];
  /** This market's latest counted pre-tip picks, newest first (bounded
   *  server-side): what the "Latest counted picks" block renders. */
  recent_picks?: TrackRecordPick[];
  /** False when the backend has no rule for judging this market: never render a rate. */
  settled?: boolean;
  /** Player props only: how far the projections were off, per stat (no hit rate exists for them). */
  per_stat?: { stat: string; n: number; mae: number; mean_signed_error: number }[];
  /** Every week since tracking began, through this week; gaps carried as tracked=false. */
  weekly?: TrackRecordWeek[];
}

/** One page of the every-recorded-pick disclosure, from /hub/track-record/picks. */
export interface TrackRecordPicksPage {
  /** The whole filtered set, before paging. */
  total: number;
  limit: number;
  offset: number;
  picks: TrackRecordPick[];
}

/** When the data the site serves was last written. `generated_at` is null
 *  when nothing has been written yet. */
export interface SnapshotMeta {
  generated_at: string | null;
  source: string;
}

export interface VsMarketWeek {
  week_start: string;
  /** Games actually compared with a price that week. */
  tracked: boolean;
  n: number;
  /** Model minus price, in percentage points; null when nothing was compared. */
  mean_edge_points: number | null;
  disagreement_n: number;
  disagreement_hit_rate: number | null;
}

export interface VsMarketScope {
  population: string;
  weekly_from: string | null;
  weekly_through: string | null;
  n_games_total: number;
  n_games_in_weekly: number;
  n_games_outside_weekly: number;
}

export interface VsMarket {
  market: string;
  n: number;
  mean_model_probability: number | null;
  mean_market_probability: number | null;
  mean_edge_points: number | null;
  disagreement_n: number;
  disagreement_hit_rate: number | null;
  disagreement_game_ids: string[];
  weekly: VsMarketWeek[];
  scope: VsMarketScope;
  /** Sentences the panel prints verbatim — see hub_service._VS_MARKET_METHOD. */
  method: Record<string, string | number>;
}

export interface TeamHubRow {
  abbreviation: string;
  conference: "East" | "West";
  division: string;
  wins: number;
  losses: number;
  points_per_game: number;
  opp_points_per_game: number;
  net_rating: number;
  pace: number;
  streak: number;
}

export interface PlayerHubRow {
  player_id: string;
  player_name: string;
  team: string;
  position: string;
  rating: number;
  live_form_rating: number;
  points_per_game: number;
  rebounds_per_game: number;
  assists_per_game: number;
  fg_pct: number;
  three_pt_pct: number;
  ft_pct: number;
  usage_rate: number;
  minutes_per_game: number;
  /** Threes made per game. Absent until the hub cache is rewritten with it. */
  threes_per_game?: number;
}

/** One game of a player's record: what happened, and what the model said. */
export interface PlayerGame {
  game_id: string;
  game_date: string;
  opponent: string;
  home: boolean;
  score: string;
  actual: Record<string, number | null>;
  projection: Record<string, number | null>;
}

/** A player's next game, with the model's projection for it. */
export interface PlayerNextGame {
  game_id: string;
  game_date: string;
  tip_off?: string | null;
  opponent: string;
  home: boolean;
  projections: Record<string, number | null>;
}

/** One player's detail: their last ten games and their status. */
export interface PlayerDetail {
  player_id: string;
  team: string | null;
  games: PlayerGame[];
  status: { status: string; source: string; dated: string } | null;
  next_game?: PlayerNextGame | null;
  averages?: Record<string, number | null>;
}

export interface PowerRankingRow {
  rank: number;
  abbreviation: string;
  power_rating: number;
  trend: "up" | "down" | "steady";
}

export interface StandingsRow {
  conference: "East" | "West";
  seed: number;
  abbreviation: string;
  wins: number;
  losses: number;
  win_pct: number;
  games_back: number;
  // null, not a status string, when the schedule says there is no race to be in
  // (off-season or pre-season). The backend drops the label rather than
  // inventing one, so the type has to admit its absence.
  playoff_status: "clinched" | "play-in" | "eliminated" | "in-hunt" | null;
  season_state: "offseason" | "preseason" | "regular" | "postseason";
}

export interface Manifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
  training?: {
    n_train_games?: number;
    n_holdout_games?: number;
    n_current_season_games?: number;
  };
}

export interface PlayerPropsManifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
  training?: {
    n_train_player_games?: number;
    in_sample_metrics?: boolean;
  };
}

export interface CalibrationBin {
  bin_start: number;
  bin_end: number;
  predicted_rate: number;
  actual_rate: number;
  count: number;
}

// A backend that accepts the connection but never answers must still end in
// the page's error state (with Try again), never an endless "Loading…".
export const REQUEST_TIMEOUT_MS = 15_000;

function fetchWithTimeout(url: string, init: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  return fetch(url, { ...init, signal: controller.signal }).finally(() => clearTimeout(timer));
}

// A redeploy or a restart shows up to a reader as a few seconds of refused connections or 502/503/504. A read is safe to
// repeat, so those (and only those) are tried again after a short wait instead of ending on the page's error state. A
// timeout is NOT retried: a backend that hangs should reach the error state in 15s, not 45s.
export const RETRY_DELAYS_MS = [1_000, 3_000];
const RETRYABLE_STATUS = new Set([502, 503, 504]);

async function fetchJson<T>(path: string): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    let response: Response;
    try {
      response = await fetchWithTimeout(`${API_BASE}${path}`);
    } catch (error) {
      // fetch rejects with a TypeError when the connection is refused or dropped; an abort is the timeout.
      if (error instanceof TypeError && attempt < RETRY_DELAYS_MS.length) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
        continue;
      }
      throw error;
    }
    if (!response.ok) {
      if (RETRYABLE_STATUS.has(response.status) && attempt < RETRY_DELAYS_MS.length) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
        continue;
      }
      throw new Error(`Request to ${path} failed with status ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}

export const api = {
  getTeams: () => fetchJson<Team[]>("/teams"),
  getTeam: (abbreviation: string) => fetchJson<Team>(`/teams/${abbreviation}`),
  getGames: (date: string) => fetchJson<Game[]>(`/games?date=${date}`),
  getGamesWeek: (start: string) => fetchJson<Game[]>(`/games/week?start=${start}`),
  getGameDetail: (gameId: string) => fetchJson<GameDetail>(`/games/${gameId}`),
  getGamePlayers: (gameId: string) => fetchJson<PlayerProp[]>(`/games/${gameId}/players`),
  /** Players the gate removed from this game's ranking: shown once, below the
   *  lists, attributed and dated. A sibling route, not a field on the props
   *  response -- see `OutPlayer`. */
  getGameOutPlayers: (gameId: string) => fetchJson<OutPlayer[]>(`/games/${gameId}/players/out`),
  /** Players the gate flagged as doubtful: still ranked, and said out loud.
   *  A sibling of the out feed, because an out player leaves the ranking and
   *  a doubtful one does not. */
  getGameDoubtfulPlayers: (gameId: string) => fetchJson<DoubtfulPlayer[]>(`/games/${gameId}/players/doubtful`),
  /** Spec §3's per-fixture signal payloads, rendered by the shared `SignalRows`.
   *  `[]` is a complete answer, never absent -- spec §2's "no data, no row". */
  getGameSignals: (gameId: string) => fetchJson<SignalsResponse>(`/api/signals/${encodeURIComponent(gameId)}`),
  /** The shared plain-English summary, via this API's explainer proxy. The id
   *  is the game's own id — the only value the route needs, and the only one
   *  the site has. */
  explainGame: (gameId: string) => fetchJson<Explanation>(`/api/explain/nba/${gameId}`),
  /** The Matchup section's data, `<explain base>/<id>/context`; same base as `explainGame`. */
  loadContext: createContextLoader(EXPLAIN_BASE),
  getSeasonFirstWeek: () => fetchJson<SeasonBounds>("/season/first-week"),
  getHubTeams: () => fetchJson<TeamHubRow[]>("/hub/teams"),
  getHubPlayers: () => fetchJson<PlayerHubRow[]>("/hub/players"),
  /** Every player's next game, with the model's projection for it. The table's
   *  last column. */
  getHubPlayersNextGames: () => fetchJson<Record<string, PlayerNextGame>>("/hub/players/next-games"),
  /** One player: their last ten games, their status, their next game. */
  getHubPlayer: (playerId: string) => fetchJson<PlayerDetail>(`/hub/players/${playerId}`),
  getHubRankings: () => fetchJson<PowerRankingRow[]>("/hub/rankings"),
  /** The season's standings, or the previous season's when `season` is
   *  "previous" (computed from the schedule cache's completed games of the
   *  previous season, preseason excluded). */
  getHubStandings: (season?: "current" | "previous") =>
    fetchJson<StandingsRow[]>(season ? `/hub/standings?season=${season}` : "/hub/standings"),
  /** The Teams page's one payload: the league's ratings, every team's form,
   *  its last five games and its next game with the model's pick. */
  getHubTeamOverview: () => fetchJson<TeamOverview>("/hub/team-overview"),
  getTrackRecord: () => fetchJson<TrackRecord[]>("/hub/track-record"),
  /** The every-recorded-pick disclosure, one page at a time. Fetched only
   *  when a reader opens the table (or the odds block asks for its games):
   *  the full set is tens of thousands of rows. */
  getTrackRecordPicks: (params: { market?: string; gameIds?: string[]; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.market) query.set("market", params.market);
    for (const id of params.gameIds ?? []) query.append("game_id", id);
    if (params.limit != null) query.set("limit", String(params.limit));
    if (params.offset != null) query.set("offset", String(params.offset));
    const qs = query.toString();
    return fetchJson<TrackRecordPicksPage>(`/hub/track-record/picks${qs ? `?${qs}` : ""}`);
  },
  getPropGrading: () => fetchJson<PropGrading>("/hub/prop-grading"),
  /** One game's complete counted grading, for its post-game review. Null
   *  on ANY failure -- a 404 (the game has no counted props), a network
   *  error and an unreadable payload are deliberately indistinguishable
   *  here: the review is an enhancement on the game page, its absence is
   *  the honest rendering for all three, and it must never become the
   *  page's error state. The same rule the signal rows follow. */
  getGamePropGrading: async (gameId: string): Promise<PropGameGrading | null> => {
    try {
      return await fetchJson<PropGameGrading>(`/hub/prop-grading/games/${encodeURIComponent(gameId)}`);
    } catch {
      return null;
    }
  },
  getVsMarket: () => fetchJson<VsMarket>("/hub/vs-market"),
  getSnapshotMeta: () => fetchJson<SnapshotMeta>("/snapshot-meta"),
  getManifest: () => fetchJson<Manifest>("/manifest"),
  getPlayerPropsManifest: () => fetchJson<PlayerPropsManifest>("/player-props-manifest"),
  getCalibration: () => fetchJson<CalibrationBin[]>("/calibration"),
};