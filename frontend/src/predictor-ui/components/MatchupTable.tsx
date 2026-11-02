// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { rankTier } from "../lib/rankTier";

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

function RankBox({ rank, teamName, label, n, tier }: {
  rank: number | null;
  teamName: string;
  label: string;
  n: number;
  tier: "good" | "mid" | "bad";
}) {
  if (rank === null) {
    return (
      <td className="pr-matchup-rank pr-matchup-missing" data-testid="rank-box">
        —
      </td>
    );
  }
  return (
    <td className="pr-matchup-rank">
      <div 
        className={`pr-rank-box pr-rank-${tier}`}
        data-testid="rank-box"
        aria-label={`${teamName}: ${ordinal(rank)} of ${n} for ${label}`}
      >
        {rank}
      </div>
    </td>
  );
}

export function MatchupTable({ pivot }: { pivot: {
  home: string;
  away: string;
  rows: { key: string; label: string; homeRank: number | null; awayRank: number | null; n: number }[];
} | null }) {
  if (!pivot) return null;

  return (
    <div className="pr-matchup-table-wrapper">
      <table className="pr-matchup-table">
        <caption className="sr-only" data-testid="matchup-caption">
          Matchup: {pivot.home} vs {pivot.away}
        </caption>
        <thead>
          <tr>
            <th scope="col" className="pr-matchup-team">{pivot.home}</th>
            <th scope="col" className="pr-matchup-label-header">Stat</th>
            <th scope="col" className="pr-matchup-team">{pivot.away}</th>
          </tr>
        </thead>
        <tbody>
          {pivot.rows.map((row) => (
            <tr key={row.key}>
              <RankBox
                rank={row.homeRank}
                teamName={pivot.home}
                label={row.label}
                n={row.n}
                tier={row.homeRank ? rankTier(row.homeRank, row.n) : "bad"}
              />
              <td className="pr-matchup-label">{row.label}</td>
              <RankBox
                rank={row.awayRank}
                teamName={pivot.away}
                label={row.label}
                n={row.n}
                tier={row.awayRank ? rankTier(row.awayRank, row.n) : "bad"}
              />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}