// Synced from predictor-ui@5b41a7e2d962. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useEffect, useRef, useState } from "react";
import type { CalibrationBucket, CalibrationSeries } from "./types";
import type { MarketTally } from "./types";

/** "50-60%" -> 0.55, "70%+" -> 0.8: where the bucket sits on the stated-confidence axis. */
function bucketMid(label: string): number | null {
  const m = /(\d+)\s*-\s*(\d+)/.exec(label);
  if (m) return (Number(m[1]) + Number(m[2])) / 200;
  const plus = /(\d+)\s*%?\s*\+/.exec(label);
  return plus ? (Number(plus[1]) + 10) / 100 : null;
}

/** Below this many settled picks for a market the chart does not draw a dot at all. */
const TOO_EARLY = 3;
/** Below this many picks in a bin the dot is faint. */
const THIN = 5;

/**
 * Stated confidence against what actually happened. On the diagonal =
 * trustworthy. Ported from NBA, with `n`/`correct` in place of the API's
 * `total_predictions`/`correct_predictions`.
 *
 * The chart is the track-record hero: full width under the tabs, at least
 * 280px tall (360px at 1280). One pill per market sits above it, each showing
 * how many settled picks that market holds; a market with fewer than 3 settled
 * picks says "Too early" instead of a dot.
 */
/**
 * The chart measures its own width and draws 1:1 in it. A fixed viewBox scaled to the container shrinks the axis text
 * with it (a 960-unit chart at 343px made 10px labels 3.6px tall); measured, the text is the same size on every screen.
 */
export function CalibrationChart(props: { series: CalibrationSeries[]; markets?: MarketTally[] }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const read = () => setWidth(Math.round(el.getBoundingClientRect().width));
    read();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(read);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={wrap} className="w-full">
      <ChartBody {...props} width={width} />
    </div>
  );
}

function ChartBody({
  series,
  markets,
  width,
}: {
  series: CalibrationSeries[];
  /** Each market's counted tally, so the pills show settled-pick counts. */
  markets?: MarketTally[];
  width: number;
}) {
  const usable = series.filter((s) => s.buckets.some((b) => b.n > 0));
  const settledByMarket = new Map(
    usable.map((s) => [s.key, s.buckets.reduce((a, b) => a + b.n, 0)]),
  );
  const countFor = (key: string) => {
    const m = markets?.find((m) => m.key === key);
    return m != null ? m.counted.n : (settledByMarket.get(key) ?? 0);
  };
  // Open on a market that has enough settled picks to draw, not on the first one listed.
  const [pick, setPick] = useState(usable.find((s) => countFor(s.key) >= TOO_EARLY)?.key ?? usable[0]?.key);
  if (usable.length === 0) return <p className="text-sm text-pr-text-dim">No confidence buckets graded yet.</p>;
  const active = usable.find((s) => s.key === pick) ?? usable[0];
  const settled = countFor(active.key);
  const W = Math.max(300, Math.min(width || 960, 1200));
  const H = Math.max(280, Math.round(W * 0.4));
  const L = 34;
  const B = 26;
  const T = 8;
  const R = 8;
  const x = (c: number) => L + ((c - 0.45) / 0.45) * (W - L - R);
  const y = (v: number) => H - B - v * (H - B - T);
  const pills = (
    <div role="group" aria-label="Market" className="mb-3 flex flex-wrap gap-2">
      {usable.map((s) => {
        const n = countFor(s.key);
        const pressed = s.key === active.key;
        return (
          <button
            key={s.key}
            type="button"
            aria-pressed={pressed}
            onClick={() => setPick(s.key)}
            disabled={n < TOO_EARLY}
            className={`inline-flex items-center gap-1.5 rounded-pr border px-3 py-1.5 text-sm font-semibold transition-colors ${
              pressed
                ? "border-pr-accent bg-pr-accent text-pr-accent-ink"
                : n < TOO_EARLY
                  ? "border-transparent bg-pr-panel-2 text-pr-text-faint opacity-50"
                  : "border-pr-rule bg-pr-panel-2 text-pr-text hover:bg-pr-panel"
            }`}
          >
            {s.label}
            {n > 0 && <span className={`tabular-nums ${pressed ? "opacity-80" : "text-pr-text-faint"}`}>· {n} settled</span>}
          </button>
        );
      })}
    </div>
  );
  if (settled < TOO_EARLY) {
    return (
      <div className="w-full min-h-[280px] lg:min-h-[360px] flex flex-col items-center justify-center gap-2">
        {pills}
        <p className="text-sm text-pr-text-faint">
          Too early to calibrate {active.label} — fewer than {TOO_EARLY} settled picks.
        </p>
      </div>
    );
  }
  const pts = active.buckets
    .map((b) => ({ b, mid: bucketMid(b.bucket), rate: b.n > 0 ? b.correct / b.n : null }))
    .filter((p): p is { b: CalibrationBucket; mid: number; rate: number } => p.mid != null && p.b.n > 0 && p.rate != null);
  return (
    <div className="w-full min-h-[280px] lg:min-h-[360px]">
      {usable.length > 1 && pills}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="block max-w-full"
        role="img"
        aria-label={`Calibration for ${active.label}`}
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="stroke-pr-rule" strokeWidth={1} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" className="fill-pr-text-faint text-[11px]">{Math.round(v * 100)}%</text>
          </g>
        ))}
        {/* The diagonal: stated confidence = actual hit rate. */}
        <line x1={x(0.45)} y1={y(0.45)} x2={x(0.9)} y2={y(0.9)} className="stroke-pr-text-faint" strokeDasharray="4 4" strokeWidth={1} />
        {[0.5, 0.6, 0.7, 0.8].map((c) => (
          <text key={c} x={x(c)} y={H - 8} textAnchor="middle" className="fill-pr-text-faint text-[11px]">{Math.round(c * 100)}%</text>
        ))}
        {pts.map(({ b, mid, rate }) => {
          const thin = b.n < THIN;
          return (
            <circle
              key={b.bucket}
              data-testid="calibration-point"
              cx={x(mid)}
              cy={y(rate)}
              r={5 + Math.min(10, b.n)}
              className={thin ? "fill-pr-accent/25 stroke-pr-accent" : "fill-pr-accent/60 stroke-pr-accent"}
              strokeWidth={1.5}
            >
              <title>{`${b.bucket} confidence: ${b.correct} of ${b.n} right`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="mt-1 text-xs text-pr-text-faint">
        Across: how sure the model said it was. Up: how often it was right. On the dashed line is honest; bigger dots
        rest on more picks; faint dots rest on fewer than {THIN}.
      </p>
    </div>
  );
}
