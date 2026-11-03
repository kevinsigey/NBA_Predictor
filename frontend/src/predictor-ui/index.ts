// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
export * from "./fmt";
export { contrast, inkFor, luminance } from "./contrast";
export { AppFrame, type SiteLink, type Sport, type Tab } from "./components/AppFrame";
export { GlobalNav, type NavEntry } from "./components/GlobalNav";
export { SportTabs, type SportTab } from "./components/SportTabs";
export { MatchCard, type Side } from "./components/MatchCard";
export { ProbabilityBar, type PickRef, type Segment } from "./components/ProbabilityBar";
export { RoundNavigator } from "./components/RoundNavigator";
export { StatTile } from "./components/StatTile";
export { EmptyState, ErrorState, Skeleton } from "./components/States";
export { ExplainerPanel, type Explanation, type Verdict, type Common } from "./components/ExplainerPanel";
export { BandChip, PanelHeading, type Band } from "./components/ExplainerVerdict";
export { KeyNumberTile, type MarketTile } from "./components/KeyNumberTile";
export { FactorList, type Factor } from "./components/FactorList";
export { MatchupBrief, createContextLoader, type MatchupRow, type FormRow, type MatchupContext } from "./components/MatchupBrief";
export { MatchupTable } from "./components/MatchupTable";
export { pivotMatchups } from "./lib/pivotMatchups";
export { rankTier } from "./lib/rankTier";
export { RankDuel, rankFill, RankOutOfRangeError, type RankDuelProps } from "./components/RankDuel";
export { WeatherChip, weatherSentence, WINDY_MPH, WET_ENOUGH_PCT, type Conditions, type WeatherKind } from "./components/WeatherChip";
export { RecordStrip } from "./components/RecordStrip";
export { FixtureFlow, type FixtureFlowProps, type FlowState } from "./components/FixtureFlow";
export { SummaryButton, type SummaryButtonProps } from "./components/SummaryButton";
export { FixtureExplainer, type FixtureExplainerProps, type FixtureExtras, type Summary } from "./components/FixtureExplainer";
export { StatTable, type Column, type Separator } from "./components/StatTable";
export { StatusBadge, statusWords, type Moment, type Status } from "./components/StatusBadge";
export { TeamChip } from "./components/TeamChip";
export { BoxScore, type BoxScoreRow, type BoxScoreColumn, type BoxScoreGroup, type BoxScoreTotal, type BoxScoreProps } from "./components/BoxScore";
export { PicksList, MAX_ROWS_PER_CATEGORY, DEFAULT_DETAIL_LABEL, detailAddsToHeading, rowShowsDetail, OutPlayerInRankingError, RowKindMismatchError, type PickRow, type OutPlayer, type PicksListProps } from "./components/PicksList";
export { SignalRows, SPEC_MIN_N, MAX_HEADLINE_WORDS, DEFAULT_SIGNAL_TITLE, clipHeadline, rateIsDrawable, signalIsDrawn, signalFigure, headlineStatesFigure, figureRange, assertFigureIsStated, UndrawableSignalVisualError, SignalFigureError, HeadlineFigureMismatchError, type Signal, type SignalKind, type SignalVisual, type SignalRowsProps } from "./components/SignalRows";
export * from "./lib/bundleFacts";
export { InstantBlock, type InstantBlockProps, AI_PROMISE } from "./components/InstantBlock";
export { StickyStatTable, type StickyColumn, type StickyRow, type StickySection, type StickyStatTableProps } from "./components/StickyStatTable";
export { TeamSwitch, TeamToggle, type TeamOption } from "./components/TeamSwitch";
// The shared Track Record kit: one vocabulary, one view model, five sports.
export { RIGHT, WRONG, PUSH, MARKET_LABELS, marketLabel, marketRank, SPORT_MOMENT, momentFor } from "./trackRecord/vocabulary";
export { wilsonInterval } from "./trackRecord/wilson";
export type {
  RecordModel,
  MarketTally,
  Tally,
  HeadlineMode,
  WeekPoint,
  CalibrationSeries,
  CalibrationBucket,
  RecentPick,
} from "./trackRecord/types";
export { RecordBlock } from "./trackRecord/RecordBlock";
export { RecordHeadline, EARLY_SAMPLE } from "./trackRecord/RecordHeadline";
export { WeekStrip } from "./trackRecord/WeekStrip";
export { CalibrationChart } from "./trackRecord/CalibrationChart";
export { CalibrationBlock } from "./trackRecord/CalibrationBlock";
export { RecentPicks } from "./trackRecord/RecentPicks";
export { HowCounted } from "./trackRecord/HowCounted";
export { HubTabs, type HubTab } from "./trackRecord/HubTabs";

// The one family switcher list, shared by every site.
export { SPORT_SITES, siteLinks } from "./lib/sites";
