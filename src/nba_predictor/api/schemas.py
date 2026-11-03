from pydantic import BaseModel


class TeamOut(BaseModel):
    abbreviation: str
    name: str
    conference: str
    division: str


class PredictionOut(BaseModel):
    home_win_probability: float
    predicted_margin: float
    predicted_total: float
    # The market line the cover probability is priced against (None when no line).
    market_point: float | None = None
    market_total_point: float | None = None
    # Spread cover probability (None when no line available)
    cover_prob_spread: float | None = None
    # Total over probability (None when no line available)
    cover_prob_total: float | None = None
    # Residual sigma from the manifest, driving the above
    margin_sigma: float | None = None
    total_sigma: float | None = None


class MarketPredictionOut(BaseModel):
    market: str
    selection: str
    model_probability: float
    market_probability: float | None = None
    edge: float | None = None
    bookmaker: str | None = None
    american_odds: int | None = None
    point: float | None = None
    # Priced after tip-off (e.g. a refresh during the game): shown, never judged.
    rebuilt: bool = False


class KeyOutOut(BaseModel):
    team: str
    player_name: str


class GameOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    tip_off: str | None = None
    prediction: PredictionOut | None = None
    # True when the pick shown was made at or after tip-off (a backtest). It
    # is labelled on the site and never counted.
    rebuilt: bool = False
    completed: bool = False
    home_pts: int | None = None
    away_pts: int | None = None
    # Reported-Out players who normally play real minutes (upcoming games only; empty when rosters are unknown).
    key_out: list[KeyOutOut] = []


class HeadToHeadMeetingOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_pts: int | None = None
    away_pts: int | None = None


class GameDetailOut(GameOut):
    markets: list[MarketPredictionOut] = []
    head_to_head: list[HeadToHeadMeetingOut] = []
    home_recent_form: list[str] = []
    away_recent_form: list[str] = []
    # The availability gate's own sentence, or the reason it could not be read.
    # "Checked against ESPN availability — 2 Out, 0 Day-to-Day" or the gate's
    # actual state (e.g. "Injury report not yet published for this game").
    injury_summary: str | None = None
    # The props model's own holdout error per stat (points, rebounds, assists,
    # threes): the band a projection has to miss by before it counts as wrong.
    # None when there is no props manifest -- no band is "not measured", never a
    # default number.
    prop_bands: dict[str, float] | None = None
    # These two teams' ratings and their league ranks, read from the box-score
    # window as of this game's own date. None when the league is not ranked
    # yet -- no ratings is not a section with zeros.
    ratings: dict | None = None


class PlayerPropOut(BaseModel):
    player_id: str
    player_name: str
    # The team he is on NOW for an upcoming game (ESPN roster); None for a finished game, where the hub feed has it.
    team: str | None = None
    stat: str
    # A projection, in the stat's own unit -- points, rebounds, assists, made
    # threes. Deliberately not a probability: models/player_props.py is an
    # XGBRegressor returning a raw point total, and this repo has no calibrated
    # probability for it to serve. predict_double_double_probability exists and
    # has zero callers; shipping it is not this change. Do not add a
    # probability field here without calibration evidence.
    predicted_value: float
    actual_value: float | None = None
    # Built after tip-off (the retrain backtest): shown, never judged.
    rebuilt: bool = False
    # In-sample mean absolute error for this stat (see
    # models.player_props.in_sample_mae_by_stat). None when nothing has been
    # resolved for the stat yet -- never 0.0, which would claim the model never
    # missed by a tenth of a point. The site renders this as "+/- N" next to
    # the projection, or says it has no error estimate yet.
    #
    # This is the HEADLINE: over every COUNTED pick for this stat -- one per
    # (game, player, stat), the earliest recorded, whenever it was made. Before
    # 2026-10-01 (predictor-hub #66) it was the pre-tip-only figure and a game
    # whose only pick came from the retrain backtest contributed nothing, which
    # is how a projection could sit on the page for a season with no error
    # estimate at all. The name is unchanged, so every reader of it now sees
    # the fuller record.
    mae: float | None = None
    # The same error estimate over the picks made BEFORE tip-off only: what the
    # model would have said on the night. Published beside `mae` because the
    # two differ, and a reader comparing a model against a book needs to know
    # which one they are reading. None when no counted pick for this stat was
    # made in time -- never 0.0, for the same reason as `mae`.
    mae_pre_tip: float | None = None
    # The n behind each figure. An error estimate with no count beside it is a
    # number nobody can weigh, and the pre-tip figure is the one most likely to
    # be thin -- so both travel with it rather than being reconstructed.
    mae_n: int = 0
    mae_n_pre_tip: int = 0


class OutPlayerOut(BaseModel):
    """A player the availability gate removed from the ranking.

    Removed, not flagged in place: no list, no bar, no rank position. Shown
    once, below the lists, attributed and dated.
    """

    player_id: str
    player_name: str
    team: str
    status: str
    source: str
    dated: str


class DoubtfulPlayerOut(BaseModel):
    """A player the availability gate flagged as doubtful but kept ranked.

    Flagged in place, NOT removed: the player is still a legitimate call and
    still appears in /games/{id}/players. This is the note that makes the
    ranking honest -- before it, a day-to-day player reached no feed at all,
    which reads as "checked and clear".

    Deliberately the same six fields as OutPlayerOut and no more. ``status`` is
    the feed's own word ("Day-To-Day"), reported verbatim rather than reworded
    into a judgement; there is NO probability and NO downgrade coefficient
    field, because a status is not a number and this repo has no calibrated
    quantity to pair it with. Nothing here is a recommendation against the
    player: it states availability, nothing more.
    """

    player_id: str
    player_name: str
    team: str
    status: str
    source: str
    dated: str


class TrackRecordTallyOut(BaseModel):
    """The secondary figure: the same tally over the picks made BEFORE tip-off.

    A whole sub-record rather than three loose numbers, so the pre-tip figure
    carries its own n, its own rate and its own week table exactly as the
    headline does. Its `n` is the size of that subset by construction -- it is
    the same summariser over the same counted rows, filtered by the derivation
    in `tracking.timing.made_before_tip`, not a figure reconciled by hand.
    """
    total_predictions: int
    correct_predictions: int
    # None when nothing in the subset graded: 0.0 would claim every one missed.
    hit_rate: float | None = None
    n_push: int = 0
    # The same window as the headline's, filled by compute_track_record.
    weekly: list["TrackRecordWeekOut"] = []


class TrackRecordPickOut(BaseModel):
    """One recorded pick for one market, with when it was made.

    Disclosure is per pick, not only in aggregate: `made_before_tip` is derived
    on every read from this row's own `created_at` against the game's tip-off,
    compared as UTC instants and failing closed to False, and `created_at` is
    published next to it so a reader can check the derivation rather than take
    it on trust. A pick made after tip-off is never presented as one made
    before.

    `counted` says whether this row is the one the headline scored. A rerun
    that lost the earliest-pick contest is still here -- recorded stays
    recorded -- and is marked `counted: false` so the rows a reader tallies are
    exactly the rows that produced the number above them.
    """
    game_id: str
    market: str
    # What the model backed, in words, with the line it was priced at.
    pick: str
    # What actually happened, in the same words.
    actual: str
    # None for a push or a row with no line: nobody won it, so it is not a miss.
    hit: bool | None = None
    made_before_tip: bool
    created_at: str
    counted: bool = True
    gameday: str | None = None
    # The line the pick was priced at. Present for spreads and totals, null
    # for moneylines and markets without a line. Used by the frontend to
    # distinguish a push (hit=null but point present) from a row with no line
    # (hit=null and no point).
    point: float | None = None


class ConfidenceBucketOut(BaseModel):
    bucket: str
    total_predictions: int
    correct_predictions: int
    hit_rate: float | None = None




class TrackRecordWeekOut(BaseModel):
    week_start: str  # ISO date of the Monday the week starts on
    n: int
    correct: int
    # None for a week with no graded picks -- see TrackRecordOut.hit_rate.
    hit_rate: float | None = None
    tracked: bool


class VsMarketWeekOut(BaseModel):
    week_start: str  # ISO Monday date
    tracked: bool  # n > 0: games actually compared with a price
    n: int
    # Percentage points; None when nothing was compared that week.
    mean_edge_points: float | None = None
    disagreement_n: int = 0
    disagreement_hit_rate: float | None = None


class VsMarketScopeOut(BaseModel):
    population: str
    weekly_from: str | None = None
    weekly_through: str | None = None
    n_games_total: int
    n_games_in_weekly: int
    n_games_outside_weekly: int


class VsMarketOut(BaseModel):
    """The model's moneyline pick beside the price it was measured against.

    NBA stores the market's own de-vigged probability per side (see
    pipeline/refresh_odds), so unlike the NFL/CFB version this needs no
    implied-probability conversion of a line: the comparison reads the same
    row the pick was priced with.
    """

    market: str = "h2h"
    n: int = 0
    mean_model_probability: float | None = None
    mean_market_probability: float | None = None
    # Percentage points, signed toward the model; None when n == 0.
    mean_edge_points: float | None = None
    # Games where the model backed the side the price did not favour.
    disagreement_n: int = 0
    disagreement_hit_rate: float | None = None
    disagreement_game_ids: list[str] = []
    weekly: list[VsMarketWeekOut] = []
    scope: VsMarketScopeOut
    # Sentences the page prints verbatim; see hub_service._VS_MARKET_METHOD.
    method: dict[str, str | float] = {}


class PropStatOut(BaseModel):
    stat: str
    n: int
    mae: float
    mean_signed_error: float


class PropGradingRowOut(BaseModel):
    """One counted player projection graded against the actual box score.

    `delta` is actual minus projected: positive means the player beat the
    projection. `grade` is the miss in units of the stat's holdout MAE, or
    None when the manifest publishes no MAE for the stat -- unjudged, never
    judged against an invented band. `baseline`/`departure` are None when
    the stored record holds no prior game for the player, so the row cannot
    be ranked by how far the model moved from his usual line.
    """

    game_id: str
    game_date: str
    home_team: str | None = None
    away_team: str | None = None
    player_id: str
    player_name: str | None = None
    stat: str
    predicted: float
    actual: float
    delta: float
    grade: str | None = None
    made_before_tip: bool = False
    baseline: float | None = None
    departure: float | None = None


class PropGradingGameOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str | None = None
    away_team: str | None = None
    rows: list[PropGradingRowOut] = []


class PropGradingDayOut(BaseModel):
    date: str
    # Every graded counted pick that day, and how many of them carry a
    # timestamp that proves they were made before tip-off. The two numbers
    # sit together so the headline can never quietly become the subset.
    n_graded: int
    n_pre_tip: int
    games: list[PropGradingGameOut] = []


class PropBandShareOut(BaseModel):
    """One grade's share of a stat's banded rows, as a count and a percent."""

    grade: str
    n: int
    pct: float


class PropStatAggregateOut(BaseModel):
    """One stat's tracking over the whole counted record, pre-tip rows only.

    `bias` is the average signed delta (actual minus projected): positive
    means the players beat the projections on average -- the projections run
    low, in the page's words. `holdout_mae` is the manifest's own error for
    the stat, the reference the live `mae` is tracking against; None when
    the manifest publishes none (no row of the stat is then banded either).
    """

    stat: str
    n: int
    mae: float
    bias: float
    holdout_mae: float | None = None
    # How many of the stat's rows carried a grade at all; the band shares
    # are shares of these, never of a larger set they were not judged in.
    band_n: int = 0
    bands: list[PropBandShareOut] = []


class PropDailyStatOut(BaseModel):
    n: int
    mae: float


class PropDailyMaeOut(BaseModel):
    date: str
    stats: dict[str, PropDailyStatOut] = {}


class PropAggregatesOut(BaseModel):
    """The prop model's tracking: aggregates over the full counted record
    the store holds, counted pre-tip rows only. `from_date`/`to_date`/
    `n_days` state the exact window the figures cover."""

    basis: str
    from_date: str | None = None
    to_date: str | None = None
    n_days: int = 0
    total: int = 0
    per_stat: list[PropStatAggregateOut] = []
    daily: list[PropDailyMaeOut] = []


class PropGameGradingOut(BaseModel):
    """One game's complete counted grading, for its post-game review."""

    game_id: str
    game_date: str
    home_team: str | None = None
    away_team: str | None = None
    # Every graded counted row for the game, and how many of them prove
    # they were made before tip-off; the pair sits together, as on the day.
    n_graded: int
    n_pre_tip: int
    rows: list[PropGradingRowOut] = []
    biggest_surprises: list[PropGradingRowOut] = []
    bands: dict[str, float] | None = None
    counting: str
    grade_basis: str


class PropGradingOut(BaseModel):
    # The holdout MAE per stat the grades were judged against; None when the
    # manifest publishes none (every grade is then None too).
    bands: dict[str, float] | None = None
    # Sentences the page prints verbatim; the backend owns the wording.
    counting: str
    confidence_basis: str
    grade_basis: str
    latest_day: PropGradingDayOut | None = None
    most_confident: list[PropGradingRowOut] = []
    biggest_surprises: list[PropGradingRowOut] = []
    # How many of the day's graded rows carried a provable rolling-average
    # baseline and so could be ranked in most_confident at all.
    n_with_baseline: int = 0
    # The model's tracking over the whole counted record (pre-tip rows
    # only), computed from the same graded population as the day above.
    aggregates: PropAggregatesOut | None = None
    # Set only when a cached value is served because a refresh failed; see
    # TrackRecordOut.served_stale_seconds for the rule.
    served_stale_seconds: float | None = None


class TrackRecordOut(BaseModel):
    market: str
    total_predictions: int
    correct_predictions: int
    # None when nothing was graded: 0.0 would claim every graded pick missed,
    # which is a different statement from "never measured".
    hit_rate: float | None = None
    # How stale this row is, in seconds, when a cached value is being served
    # because a refresh failed. None means the row was computed for this
    # request. Set only on the cached player-props aggregate, and only when the
    # last refresh attempt failed -- an ordinary in-TTL cache hit is fresh, not
    # stale, and saying otherwise would train readers to ignore the field.
    served_stale_seconds: float | None = None
    # The number of graded counted picks made at or after their own tip-off:
    # the reconciliation between the headline and the pre-tip subset.
    n_rebuilt: int = 0
    # The size of the pre-tip subset. Equal to `pre_tip.total_predictions`.
    n_pre_tip: int = 0
    # Picks left out of the rate because there was nothing to grade them
    # against (a push, or a row with no line). Counted, never scored as a miss.
    n_push: int = 0
    # Counted picks the schedule cannot date: in the headline and in
    # `per_pick` but in no week row.
    n_unplaced: int = 0
    # False when there is no rule for judging the market (or no results).
    settled: bool = True
    # The pre-tip subset beside the headline. None only where `settled` is false.
    pre_tip: TrackRecordTallyOut | None = None
    # Every recorded pick for this market, counted or not. Only the service's
    # full builder fills this; the summary route serves it empty and the rows
    # are paged from GET /hub/track-record/picks instead (see
    # TrackRecordPicksOut) -- inlined here they made the summary ~14 MB.
    per_pick: list[TrackRecordPickOut] = []
    # This market's latest counted pre-tip picks, newest first, bounded by
    # hub_service.RECENT_PICKS_PER_MARKET. The page's "Latest counted picks"
    # block renders from these, so the summary carries the handful of rows it
    # actually shows rather than the whole disclosure.
    recent_picks: list[TrackRecordPickOut] = []
    # Weekly table; the headline's n adds up to `total_predictions`.
    weekly: list["TrackRecordWeekOut"] = []
    # Confidence buckets over the counted graded picks (50-60/60-70/70%+).
    confidence_buckets: list[ConfidenceBucketOut] | None = None
    # Per-stat MAE + signed error for player props.
    per_stat: list[PropStatOut] | None = None
    # Per-position MAE for player props.
    per_position_mae: dict[str, float] | None = None




class TrackRecordPicksOut(BaseModel):
    """One page of the every-recorded-pick disclosure.

    The rows are the same `per_pick` rows `compute_track_record` builds --
    one builder, so the table can never disagree with the aggregates above
    it -- paged here instead of inlined into the summary (the full list is
    tens of thousands of rows). `total` counts the whole filtered set, so a
    reader paging through knows when they have seen all of it. Rows are
    newest-stamp-first, the order the table reads in.
    """

    total: int
    limit: int
    offset: int
    picks: list[TrackRecordPickOut] = []


class SeasonBoundsOut(BaseModel):
    first_week_start: str | None = None
