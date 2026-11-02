from datetime import datetime
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.candidate_race import run_candidate_race
from nba_predictor.models.evaluate.walk_forward import chronological_split
from nba_predictor.models.evaluate.walk_forward_eval import (
    walk_forward_metrics,
    walk_forward_regression,
)
from nba_predictor.models.game_outcome import (
    predict_win_probability,
    train_margin_model,
    train_total_model,
    train_win_probability_model,
)
from nba_predictor.models.manifest import append_manifest_history, build_manifest, write_manifest
from nba_predictor.models.probability import MAE_TO_SIGMA, win_prob


logger = logging.getLogger(__name__)

#: Games the league-wide scoring drift is read over, and the most it may move a total. League scoring rose every season
#: since 2020 and a model fitted on all earlier seasons lags it: its totals were low by +2.6 pts in each of 2020-2025
#: (nba-eval/calib_audit.py, race8_totals.py). Adding the model's own recent league-wide residual to each total removed
#: the bias (+2.59 -> +0.08 out of sample; MAE 15.07 -> 14.99, interval excluding zero).
TOTAL_DRIFT_WINDOW = 300
TOTAL_DRIFT_CAP = 4.0


def compute_total_drift(train_df: pd.DataFrame, total_model, feature_cols: list[str]) -> float:
    """Mean (actual - predicted) total over the league's most recent `TOTAL_DRIFT_WINDOW` games, capped.

    Read in-sample on the games the model was just fitted on: the recent ones still carry the drift because the fit
    averages over all of history. 0.0 when there is nothing to read; never raises.
    """
    try:
        recent = train_df.sort_values("game_date").tail(TOTAL_DRIFT_WINDOW)
        if len(recent) < 30:
            return 0.0
        resid = (recent["home_pts"] + recent["away_pts"]) - total_model.predict(recent[feature_cols])
        drift = float(np.mean(resid))
    except Exception:  # noqa: BLE001 - a drift that cannot be read is no correction, not a failed retrain
        logger.exception("total drift not computed")
        return 0.0
    if not np.isfinite(drift):
        return 0.0
    return float(np.clip(drift, -TOTAL_DRIFT_CAP, TOTAL_DRIFT_CAP))


def manifest_total_drift(manifest: dict) -> float:
    """The drift a scorer adds to every predicted total: the manifest's, or 0.0 when it has none (an older manifest)."""
    drift = (manifest.get("metrics", {}).get("total", {}) or {}).get("drift")
    return float(drift) if isinstance(drift, (int, float)) and np.isfinite(drift) else 0.0


def _publish_sigma(block: dict, pooled_mae: float) -> None:
    """Write `residual_sigma` only when it is a usable number.

    A zero or non-finite MAE means the fit found no spread to measure, so
    sigma = mae * MAE_TO_SIGMA would publish 0.0 and every served probability
    would divide by it. Omitting the key is the honest state: the scorer
    refuses to serve, and says the sigma is missing, rather than reporting a
    confident 50% from a broken fit.
    """
    sigma = float(pooled_mae) * MAE_TO_SIGMA
    if np.isfinite(sigma) and sigma > 0.0:
        block["residual_sigma"] = sigma


def score_served_win_from_margin(pooled: dict) -> dict:
    """log-loss / Brier / AUC of the probability we actually serve.

    `pipeline/ingest.py` writes `home_win_prob = win_prob(predicted_margin,
    margin_sigma)`, so this is the number the manifest's `win_probability` key
    has to carry. Scoring the logistic classifier instead described a model
    nothing serves, which is how 0.6904 / 0.2481 / 0.5587 came to describe a
    served Phi(margin/sigma) it had nothing to do with.

    Graded on the same out-of-fold games as the margin walk-forward that
    produced `pooled`, against each game's real result: a positive held-out
    margin IS a home win, so `y > 0` is the outcome, not a second label column
    that could disagree with it.
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    preds = np.asarray(pooled["preds"], dtype=float)
    outcomes = (np.asarray(pooled["y"], dtype=float) > 0).astype(int)
    sigma = float(pooled["mae"]) * MAE_TO_SIGMA
    if preds.size == 0 or sigma <= 0 or len(set(outcomes.tolist())) < 2:
        # One class in the holdout makes AUC undefined, and log_loss with a
        # single label is a division by zero. None means "not measured",
        # which the frontend already renders as a dash.
        return {"log_loss": None, "brier": None, "auc": None}
    probabilities = np.array([win_prob(float(m), sigma) for m in preds])
    return {
        "log_loss": float(log_loss(outcomes, probabilities)),
        "brier": float(brier_score_loss(outcomes, probabilities)),
        "auc": float(roc_auc_score(outcomes, probabilities)),
        "sigma": sigma,
        "n": int(preds.size),
    }


def _nba_season_start_year(game_date: str) -> int:
    """NBA season starting year: Oct-Dec belongs to the season starting that
    calendar year, Jan-Sep to the season that started the prior year."""
    dt = datetime.fromisoformat(str(game_date).replace("Z", "+00:00"))
    return dt.year if dt.month >= 10 else dt.year - 1


def run_retrain_pipeline(
    games: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str, players: pd.DataFrame | None = None,
) -> dict:
    """`players`: per-player box scores behind the availability features (None = documented no-history default)."""
    models_dir.mkdir(parents=True, exist_ok=True)

    train_games, holdout_games = chronological_split(games, date_col="game_date", holdout_fraction=0.2)

    latest_season = _nba_season_start_year(games["game_date"].max())
    # Counted on the FULL frame, not the train split: the chronological
    # holdout holds the newest games by construction, so a train-split count
    # reads 0 for the current season until it is nearly over (it read 0 all
    # through Oct-Nov 2026 while ~100 current-season games sat in holdout).
    # The field means "games of the latest season present in the data".
    n_current_season_games = int(
        (games["game_date"].apply(_nba_season_start_year) == latest_season).sum()
    )

    train_df, feature_cols = build_training_frame(train_games, players=players)
    holdout_df, _ = build_training_frame(pd.concat([train_games, holdout_games], ignore_index=True), players=players)
    holdout_df = holdout_df[holdout_df["game_id"].isin(holdout_games["game_id"])]

    logger = logging.getLogger(__name__)

    # Candidate race on the training frame to pick the winners per target
    # No fallback: if the race fails or any target has no winner, we stop.
    # Publishing a manifest from a failed race or a target with no winner would
    # serve models that did not win a valid race.
    race = run_candidate_race(train_df, feature_cols=feature_cols, windows=4)
    win_candidate = race["winner"].get("win")
    margin_candidate = race["winner"].get("margin")
    total_candidate = race["winner"].get("total")
    if win_candidate is None or margin_candidate is None or total_candidate is None:
        raise RuntimeError(
            f"candidate race produced no winner for one or more targets: "
            f"win={win_candidate}, margin={margin_candidate}, total={total_candidate}; "
            "retrain aborted, no manifest published"
        )

    win_model = train_win_probability_model(train_df[feature_cols], train_df["home_win"], candidate=win_candidate)
    margin_model = train_margin_model(train_df[feature_cols], train_df["home_pts"] - train_df["away_pts"], candidate=margin_candidate)
    total_model = train_total_model(train_df[feature_cols], train_df["home_pts"] + train_df["away_pts"], candidate=total_candidate)

    metrics = {}
    # Compute walk-forward metrics as required
    wf_win = None
    wf_margin = None
    wf_total = None
    try:
        wf_win = walk_forward_metrics(
            train_df,
            model_factory=lambda tr: (
                lambda X: predict_win_probability(
                    train_win_probability_model(tr[feature_cols], tr["home_win"], candidate=win_candidate), X[feature_cols]
                )
            ),
            windows=4,
        )
    except Exception:
        logger.exception("walk_forward win failed")
        wf_win = None
    try:
        wf_margin = walk_forward_regression(
            train_df.assign(home_margin=train_df["home_pts"] - train_df["away_pts"]),
            model_factory=lambda tr: (
                lambda X: train_margin_model(tr[feature_cols], tr["home_margin"], candidate=margin_candidate).predict(X[feature_cols])
            ),
            target="home_margin",
            windows=4,
        )
    except Exception:
        logger.exception("walk_forward margin failed")
        wf_margin = None
    try:
        total_target = train_df["home_pts"] + train_df["away_pts"]
        wf_total = walk_forward_regression(
            train_df.assign(home_total=total_target),
            model_factory=lambda tr: (
                lambda X: train_total_model(tr[feature_cols], tr["home_total"], candidate=total_candidate).predict(X[feature_cols])
            ),
            target="home_total",
            windows=4,
        )
    except Exception:
        logger.exception("walk_forward total failed")
        wf_total = None

    if len(holdout_df) > 0:
        win_probs = predict_win_probability(win_model, holdout_df[feature_cols])
        win_preds = (win_probs >= 0.5).astype(int)
        metrics["win_probability"] = {"accuracy": float((win_preds == holdout_df["home_win"]).mean())}

        margin_preds = margin_model.predict(holdout_df[feature_cols])
        actual_margin = holdout_df["home_pts"] - holdout_df["away_pts"]
        metrics["margin"] = {"mae": float(np.mean(np.abs(margin_preds - actual_margin)))}

        total_preds = total_model.predict(holdout_df[feature_cols])
        actual_total = holdout_df["home_pts"] + holdout_df["away_pts"]
        metrics["total"] = {"mae": float(np.mean(np.abs(total_preds - actual_total)))}
    else:
        metrics = {"win_probability": {"accuracy": None}, "margin": {"mae": None}, "total": {"mae": None}}

    metrics.setdefault("total", {})["drift"] = compute_total_drift(train_df, total_model, feature_cols)

    joblib.dump(win_model, models_dir / "win_probability_model.pkl")
    joblib.dump(margin_model, models_dir / "margin_model.pkl")
    joblib.dump(total_model, models_dir / "total_model.pkl")

    # Place probability metrics under metrics.win_probability (frontend reads per-model metrics)
    # win_probability: unprefixed so labels light up (G10); keep accuracy as-is
    #
    # These must describe the SERVED probability. `pipeline/ingest.py` serves
    # win = norm.cdf(predicted_margin / margin_sigma), so publishing the
    # logistic's own scores here described a model nothing serves, and a reader
    # had no way to tell which number they were looking at. The logistic's
    # scores move under `win_classifier` so the comparison stays visible rather
    # than deleted.
    if wf_win:
        pooled_win = wf_win["pooled"]
        metrics.setdefault("win_classifier", {})["log_loss"] = pooled_win.get("log_loss")
        metrics.setdefault("win_classifier", {})["brier"] = pooled_win.get("brier")
        metrics.setdefault("win_classifier", {})["auc"] = pooled_win.get("auc")

    # The served win probability, scored on the SAME out-of-fold games the
    # margin walk-forward was scored on, graded against each game's real
    # result (`y` is the held-out margin; a positive margin IS a home win).
    if wf_margin and wf_margin["pooled"].get("preds"):
        metrics.setdefault("win_probability", {}).update(
            score_served_win_from_margin(wf_margin["pooled"])
        )
    if wf_margin:
        metrics.setdefault("margin", {})["wf_mae"] = wf_margin["pooled"]["mae"]
        if "naive_mae_fixed" in wf_margin["pooled"]:
            metrics.setdefault("margin", {})["wf_naive_mae_fixed"] = wf_margin["pooled"]["naive_mae_fixed"]
        # The residual sigma that drives every served probability
        # (models/probability.py). `pooled["mae"]` is already the mean absolute
        # residual, so this is exactly `fit_residual_sigma(residuals)` — one
        # path, not a try/except whose fallback recomputed the same number.
        #
        # Published only when usable. `score_upcoming_games` refuses to serve
        # without a positive finite sigma, so writing a 0.0 here would ship a
        # manifest that advertises a model the scorer will reject. Absent means
        # "not fitted", which the scorer reports by name.
        _publish_sigma(metrics["margin"], wf_margin["pooled"]["mae"])
    if wf_total:
        metrics.setdefault("total", {})["wf_mae"] = wf_total["pooled"]["mae"]
        _publish_sigma(metrics["total"], wf_total["pooled"]["mae"])

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
        training={
            "n_train_games": int(len(train_games)),
            "n_holdout_games": int(len(holdout_games)),
            "n_current_season_games": n_current_season_games,
            "win_candidate": win_candidate,
            "margin_candidate": margin_candidate,
            "total_candidate": total_candidate,
        },
    )
    write_manifest(manifest, models_dir / "manifest.json")
    append_manifest_history(manifest, models_dir / "manifest_history.jsonl")

    return manifest
