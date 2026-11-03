// Synced from predictor-ui@213533e42ec1. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { RecordBlock } from "./RecordBlock";
import { CalibrationChart } from "./CalibrationChart";
import type { CalibrationSeries, MarketTally } from "./types";

/**
 * The calibration block: the "Does 70% mean 70%?" hero block, titled and
 * wrapped like every other record block. A site passes its markets and series;
 * the kit renders the chart, the market pills, and the small-sample states.
 *
 * This is the composition helper B1 asks for: each site only passes its
 * markets, never the markup.
 */
export function CalibrationBlock({
  series,
  markets,
}: {
  series: CalibrationSeries[];
  /** Each market's counted tally, so the pills show settled-pick counts. */
  markets?: MarketTally[];
}) {
  return (
    <RecordBlock id="tr-calibration" title="Does 70% mean 70%?">
      <CalibrationChart series={series} markets={markets} />
    </RecordBlock>
  );
}
