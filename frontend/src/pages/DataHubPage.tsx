import TeamsPanel from "../components/TeamsPanel";
import PlayerHubPanel from "../components/PlayerHubPanel";
import TrackRecordPanel from "../components/TrackRecordPanel";

const PANELS = {
  "team-hub": TeamsPanel,
  "player-hub": PlayerHubPanel,
  "track-record": TrackRecordPanel,
} as const;

/**
 * One Data Hub panel per route. The page tabs in AppFrame are the only navigation:
 * the panel is chosen by the route (`initialTab`), never by a second tab row here.
 */
export default function DataHubPage({ initialTab = "team-hub" }: { initialTab?: keyof typeof PANELS }) {
  const Panel = PANELS[initialTab];
  return <Panel />;
}
