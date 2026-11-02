import numpy as np
import pandas as pd
import pytest


def _synthetic_games(n=60, seed=2):
    rng = np.random.default_rng(seed)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-10-21", periods=n).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(rng.random() > 0.4),
            }
        )
    return pd.DataFrame(rows)


def test_run_retrain_pipeline_produces_models_and_manifest(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    manifest = run_retrain_pipeline(games, models_dir, model_version="v-test", trained_at="2026-11-01T00:00:00")

    assert manifest["model_version"] == "v-test"
    assert set(manifest["models"]) == {"win_probability", "margin", "total"}
    assert (models_dir / "win_probability_model.pkl").exists()
    assert (models_dir / "margin_model.pkl").exists()
    assert (models_dir / "total_model.pkl").exists()
    assert (models_dir / "manifest.json").exists()
    assert (models_dir / "manifest_history.jsonl").exists()
    assert "accuracy" in manifest["metrics"]["win_probability"]


def test_run_retrain_pipeline_appends_history_across_calls(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    run_retrain_pipeline(games, models_dir, model_version="v1", trained_at="2026-11-01T00:00:00")
    run_retrain_pipeline(games, models_dir, model_version="v2", trained_at="2026-11-02T00:00:00")

    lines = (models_dir / "manifest_history.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_retrain_manifest_has_training_counts(tmp_path):
    import json
    from nba_predictor.pipeline.retrain import run_retrain_pipeline
    games = _synthetic_games()  # 60 games starting 2026-10-21: one NBA season
    models_dir = tmp_path / "models"
    manifest = run_retrain_pipeline(games, models_dir, "v9", "2026-09-23T00:00:00+00:00")
    training = manifest["training"]
    assert training["n_train_games"] > 0
    assert training["n_holdout_games"] > 0
    assert training["n_train_games"] + training["n_holdout_games"] == len(games)
    # Full-frame count: this fixture is a single season, so every game is a
    # current-season game (the old train-split count under-reported by the
    # holdout's share and read 0 in production early in a season).
    assert training["n_current_season_games"] == len(games)
    on_disk = json.loads((models_dir / "manifest.json").read_text())
    assert on_disk["training"] == training


def test_retrain_uses_candidate_race_winner(tmp_path):
    """Retrain pipeline should run candidate race and use winner per target.
    
    The manifest should record which candidate was used for each model.
    This test will fail until we wire the candidate race into retrain.
    """
    import json
    from nba_predictor.pipeline.retrain import run_retrain_pipeline
    
    games = _synthetic_games(n=100)  # Need more games for race
    models_dir = tmp_path / "models"
    manifest = run_retrain_pipeline(games, models_dir, "v-test", "2026-11-01T00:00:00")
    
    # The manifest should record the candidate used for each model
    training = manifest["training"]
    assert "win_candidate" in training, "manifest should record win_candidate"
    assert "margin_candidate" in training, "manifest should record margin_candidate"
    assert "total_candidate" in training, "manifest should record total_candidate"
    
    # The candidates should be valid choices
    valid_candidates = {"xgboost", "logistic", "ridge"}
    assert training["win_candidate"] in valid_candidates
    assert training["margin_candidate"] in valid_candidates
    assert training["total_candidate"] in valid_candidates


def test_a_failed_candidate_race_aborts_the_retrain_and_publishes_no_manifest(tmp_path, monkeypatch):
    """A race that raises must not fall back to xgboost: the retrain stops and writes nothing.

    xgboost is the model Phase A measured as worse than naive. Shipping it silently when the race
    breaks is the failure this test exists to forbid.
    """
    import pytest
    from nba_predictor.pipeline import retrain

    def boom(*args, **kwargs):
        raise RuntimeError("race exploded")

    monkeypatch.setattr(retrain, "run_candidate_race", boom)
    models_dir = tmp_path / "models"
    with pytest.raises(RuntimeError):
        retrain.run_retrain_pipeline(_synthetic_games(n=100), models_dir, "v-fail", "2026-11-01T00:00:00")
    assert not (models_dir / "manifest.json").exists(), "no manifest may be published"
    assert not (models_dir / "win_probability_model.pkl").exists()


def test_a_race_with_no_winner_for_a_target_aborts_the_retrain(tmp_path, monkeypatch):
    import pytest
    from nba_predictor.pipeline import retrain

    monkeypatch.setattr(
        retrain, "run_candidate_race",
        lambda *a, **k: {"winner": {"win": "logistic", "margin": None, "total": "ridge"}},
    )
    models_dir = tmp_path / "models"
    with pytest.raises(RuntimeError, match="no winner"):
        retrain.run_retrain_pipeline(_synthetic_games(n=100), models_dir, "v-nowin", "2026-11-01T00:00:00")
    assert not (models_dir / "manifest.json").exists()


def test_manifest_win_metrics_describe_the_served_probability(tmp_path):
    """The manifest's `win_probability` numbers must be the served
    Phi(margin/sigma) score, not the logistic classifier's.

    The defect this pins: the manifest published the classifier's log-loss /
    Brier / AUC under `win_probability` while `pipeline/ingest.py` served
    win-from-margin, so the published figures described a model nothing serves.
    The classifier's own scores now live under `win_classifier`, so the
    comparison is visible rather than deleted.
    """
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    # Real margins must vary in sign. `_synthetic_games` scores every game
    # 110-105, so its held-out margins are all +5 and the outcomes are one
    # class -- AUC and log_loss are undefined there, which is the guard
    # working, not a score to assert on.
    rng = np.random.default_rng(11)
    games = _synthetic_games(n=100)
    margins = rng.integers(-20, 21, size=len(games))
    games["home_pts"] = 110 + margins // 2
    games["away_pts"] = 110 - (margins - margins // 2)
    games["home_win"] = (margins > 0).astype(int)
    assert games["home_win"].nunique() == 2

    manifest = run_retrain_pipeline(
        games, tmp_path / "models",
        model_version="v-test", trained_at="2026-11-01T00:00:00",
    )

    served = manifest["metrics"]["win_probability"]
    assert served["log_loss"] is not None, "served win metrics were not measured"
    assert served["brier"] is not None
    assert served["auc"] is not None
    assert served["sigma"] == pytest.approx(
        manifest["metrics"]["margin"]["residual_sigma"]
    ), "the served score was graded on a different sigma than the one served"

    # The classifier's scores are kept, but under their own name.
    assert "log_loss" in manifest["metrics"]["win_classifier"]
    assert manifest["metrics"]["win_classifier"]["log_loss"] != served["log_loss"], (
        "win_probability still carries the classifier's numbers"
    )


def test_a_zero_spread_fit_publishes_no_sigma():
    """A fit with no residual spread must not publish residual_sigma: 0.0.

    `score_upcoming_games` refuses to serve without a positive finite sigma, so
    publishing 0.0 would ship a manifest advertising a model the scorer then
    rejects. Absent is the honest state, and it is what the scorer names.
    """
    from nba_predictor.pipeline.retrain import _publish_sigma

    block = {}
    _publish_sigma(block, 0.0)
    assert "residual_sigma" not in block, "published an unusable sigma"

    block = {}
    _publish_sigma(block, float("nan"))
    assert "residual_sigma" not in block

    block = {}
    _publish_sigma(block, 12.0)
    assert block["residual_sigma"] > 0
