"""Serving replay: does scoring the availability columns from the pre-tip injury report beat the previous-game proxy?

Injury reports are stored only from 2026-10-20 (data/injuries/), so there is nothing to train on. This replays the
COMMITTED game models (trained 2026-10-19, before every game scored here) over settled games with a stored report,
twice: baseline = today's previous-game availability, candidate = `report_by_game` from the latest snapshot seen before
tip-off (and at most 30h before it, the props lead window). Everything else -- features, models, sigma, total drift --
is the served path. Scored with the repo's paired bootstrap (games resampled) and `ship_decision`.

Fetches public ESPN data into the gitignored data/cache (no key). Writes per-game raw predictions beside the doc.

Run:
    uv run python tools/replay_injury_report.py
"""

import csv
import gzip
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.availability import AVAILABILITY_COLUMNS  # noqa: E402
from nba_predictor.features.build import build_feature_frame  # noqa: E402
from nba_predictor.models.artifact_gate import calibration_gap  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap, paired_gap_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.ship_rule import ship_decision  # noqa: E402
from nba_predictor.models.probability import win_prob  # noqa: E402
from nba_predictor.pipeline.history import merge_history  # noqa: E402
from nba_predictor.pipeline.ingest import (  # noqa: E402
    enrich_with_boxscores, fetch_player_boxscores, fetch_schedule_range, to_player_training_frame, to_scoring_frame,
)
from nba_predictor.pipeline.retrain import manifest_total_drift  # noqa: E402

START, FIRST_REPORT = "2025-10-01", "2026-10-20"
MAX_LEAD = timedelta(hours=30)
N_RES, SEED = 2000, 20261117
OUT_CSV = Path("docs/nba-injury-report-availability-2026-11.csv")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_snapshots(folder=Path("data/injuries")) -> list[tuple[datetime, dict]]:
    """[(seen_at, {player_id: status})] in time order, one per collector run."""
    snaps: dict[str, dict] = {}
    for f in sorted(folder.glob("nba_injuries_*.jsonl.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line)
            if r.get("player_id"):
                snaps.setdefault(r["seen_at"], {})[r["player_id"]] = r.get("status", "")
    return sorted((_ts(k), v) for k, v in snaps.items())


def pre_tip_report(snaps, tip_off: str) -> dict | None:
    tip = _ts(tip_off)
    before = [(t, rep) for t, rep in snaps if tip - MAX_LEAD <= t < tip]
    return before[-1][1] if before else None


def main() -> int:
    end = (date.today() - timedelta(days=1)).isoformat()
    games = enrich_with_boxscores(fetch_schedule_range(START, end))
    players = to_player_training_frame(games, fetch_player_boxscores(games))
    all_games, players = merge_history(games, players, START)

    snaps = load_snapshots()
    scored = [g for g in games if g.get("completed") and g.get("home_pts") is not None and g["game_date"] >= FIRST_REPORT
              and g.get("season_type") != 1 and g.get("tip_off")]
    reports = {g["game_id"]: r for g in scored if (r := pre_tip_report(snaps, g["tip_off"])) is not None}

    sdf = to_scoring_frame(all_games)
    base, cols = build_feature_frame(sdf, players=players)
    cand, _ = build_feature_frame(sdf, players=players, report_by_game=reports)
    base, cand = (f.set_index("game_id").loc[list(reports)] for f in (base, cand))

    models = Path("models")  # the committed artifacts production serves (trusted repo files)
    manifest = json.loads((models / "manifest.json").read_text())
    sigma = manifest["metrics"]["margin"]["residual_sigma"]
    margin_m, total_m = joblib.load(models / "margin_model.pkl"), joblib.load(models / "total_model.pkl")
    drift = manifest_total_drift(manifest)

    def predict(f):
        m = margin_m.predict(f[cols])
        return m, np.array([win_prob(float(x), sigma) for x in m]), total_m.predict(f[cols]) + drift

    (bm, bp, bt), (cm, cp, ct) = predict(base), predict(cand)
    margin = (base.home_pts - base.away_pts).to_numpy(float)
    total = (base.home_pts + base.away_pts).to_numpy(float)
    y = (margin > 0).astype(int)
    changed = (base[cols].to_numpy() != cand[cols].to_numpy()).any(axis=1)

    boot = {
        "win log_loss": paired_bootstrap(y, bp, cp, "log_loss", n_resamples=N_RES, seed=SEED),
        "win brier": paired_bootstrap(y, bp, cp, "brier", n_resamples=N_RES, seed=SEED),
        "win auc": paired_bootstrap(y, bp, cp, "auc", n_resamples=N_RES, seed=SEED),
        "margin mae": paired_bootstrap(margin, bm, cm, "mae", n_resamples=N_RES, seed=SEED),
        "total mae": paired_bootstrap(total, bt, ct, "mae", n_resamples=N_RES, seed=SEED),
    }
    gb, gc = calibration_gap(y, bp, 5), calibration_gap(y, cp, 5)
    widened = paired_gap_bootstrap(y, bp, cp, n_resamples=500, seed=SEED)["widened"]
    dec = ship_decision(boot, gap=gc, baseline_gap=gb, gap_widened=widened)

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "game_date", "home_team", "away_team", "home_margin", "total", "report_changed_features",
                    "base_margin", "cand_margin", "base_win_prob", "cand_win_prob", "base_total", "cand_total",
                    *[f"base_{c}" for c in AVAILABILITY_COLUMNS], *[f"cand_{c}" for c in AVAILABILITY_COLUMNS]])
        for i, gid in enumerate(base.index):
            w.writerow([gid, base.game_date.iloc[i], base.home_team.iloc[i], base.away_team.iloc[i], margin[i], total[i],
                        int(changed[i]), round(bm[i], 4), round(cm[i], 4), round(bp[i], 5), round(cp[i], 5),
                        round(bt[i], 4), round(ct[i], 4),
                        *[round(float(base[c].iloc[i]), 5) for c in AVAILABILITY_COLUMNS],
                        *[round(float(cand[c].iloc[i]), 5) for c in AVAILABILITY_COLUMNS]])

    print(f"settled games since {FIRST_REPORT}: {len(scored)}; with a pre-tip report (<=30h): {len(reports)}; "
          f"features changed by the report: {int(changed.sum())}  ({min(base.game_date)} .. {max(base.game_date)})")
    for k, o in boot.items():
        print(f"  {k:<13}{o['baseline']:>9.4f}{o['candidate']:>9.4f}{o['difference']:>+9.4f}  "
              f"[{o['ci_low']:+.4f},{o['ci_high']:+.4f}] {'IMPROVED' if o['improved'] else 'n.s.'}")
    print(f"  cal gap {gb:.4f} -> {gc:.4f} (real widening: {widened}); improved={dec['improved']} "
          f"worse={dec['worse']} auc_ok={dec['auc_ok']} => {'SHIPS' if dec['ships'] else 'does NOT ship'}")
    print(f"  raw per-game predictions -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
