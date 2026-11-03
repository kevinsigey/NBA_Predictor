import { useEffect, useRef } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";
import GamesPage from "./pages/GamesPage";
import DataHubPage from "./pages/DataHubPage";
import ModelSummaryPage from "./pages/ModelSummaryPage";
import CalibrationPage from "./pages/CalibrationPage";

declare global {
  interface Window {
    goatcounter?: {
      count: (vars?: { path?: string; title?: string; event?: boolean }) => void;
    };
  }
}

// GoatCounter counts the initial page load itself (the count.js onload); this
// tracker counts every client-side navigation after it. No-ops until count.js
// has loaded, and in tests where it never does.
function GoatCounterTracker() {
  const { pathname } = useLocation();
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    window.goatcounter?.count({ path: pathname });
  }, [pathname]);
  return null;
}

// Page tabs map onto the existing routes, so deep links and refreshes keep working.
// The aggregate "Data Hub" tab is dropped; its panels each have their own route.
// Teams lives at /hub (the old Data Hub URL): /teams is an API route, so a refresh or shared
// link to it would return JSON instead of this app.
//
// ONE Teams tab. The standings and the power rankings were two more tabs saying two more
// things about the same thirty teams, so both now land on the Teams page -- the power
// rankings' old link keeps its intent with ?sort=power.
const TABS = [
  { id: "/", label: "Games" },
  { id: "/hub", label: "Teams" },
  { id: "/players", label: "Players" },
  { id: "/track-record", label: "Track record" },
  { id: "/calibration-report", label: "Calibration" },
  { id: "/model", label: "Model" },
];

export default function App() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const active = TABS.find((tab) => tab.id !== "/" && pathname.startsWith(tab.id))?.id ?? "/";

  return (
    <>
      <GoatCounterTracker />
    <AppFrame
      sport="nba"
      sportName="NBA"
      sites={SITES}
      tabs={TABS}
      activeTab={active}
      paperTradingHref="https://40-160-91-131.sslip.io/paper.html"
      onTab={(id) => navigate(id)}
    >
      <Routes>
        <Route path="/" element={<GamesPage />} />
        <Route path="/hub" element={<DataHubPage initialTab="team-hub" />} />
        <Route path="/players" element={<DataHubPage initialTab="player-hub" />} />
        <Route path="/track-record" element={<DataHubPage initialTab="track-record" />} />
        {/* Prop grading has no page of its own any more: the graded lines
            live inside each game's post-game review, and the model's
            tracking aggregates live on the track record. An old link lands
            on the record, where the aggregates are. */}
        <Route path="/prop-grading" element={<Navigate to="/track-record" replace />} />
        {/* The two old team tabs land on the Teams page. The power rankings' old
            link keeps its intent with ?sort=power. */}
        <Route path="/standings" element={<Navigate to="/hub" replace />} />
        <Route path="/power-rankings" element={<Navigate to="/hub?sort=power" replace />} />
        <Route path="/model" element={<ModelSummaryPage />} />
        <Route path="/calibration-report" element={<CalibrationPage />} />
      </Routes>
    </AppFrame>
    </>
  );
}
