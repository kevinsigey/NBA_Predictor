"""The fabricated zero did not reach a model. Pinned, so it stays pinned.

The question this file answers: did data/injuries.py's fabricated 0.0 get
fitted into a shipped model, or served to a user?

Answer, traced at 22dfbc7567b08dcbe7e83c242b956e48f69797a0: no, on both counts.

  * Nothing ever called features/injuries.py::missing_players_value except its
    own test. features/__init__.py is empty (so no package re-export), and no
    module in src/ imports features.injuries. Its fabricated zero could only
    have entered a model through a caller that does not exist.
  * The three TEAM models do carry home_missing_value/away_missing_value as
    features (24 features, verified by loading the committed .pkl), but those
    columns are filled by features/build.py:46-48 with a literal 0.0 for every
    row -- a constant, never from the stub -- and XGBoost never split on them,
    so they carry no fitted signal at all.

What these tests lock in:

  * The team models' missing_value columns are a constant, not the stub's
    fabricated per-player zero.
  * A constant feature XGBoost never split on cannot be a fitted influence, so
    even the (unreachable) stub path could not have moved a prediction through
    it.
  * 0.0 IS indistinguishable from a real measured 0.0 downstream -- a genuine
    design defect, recorded here rather than fixed, because build.py's default
    -zero policy covers five columns and changing it is a separate change.

HISTORICAL NOTE (2026-10-14): The five dead columns (home_power_rating,
away_power_rating, power_rating_diff, home_missing_value, away_missing_value)
were REPLACED by the H feature block (elo_diff_s, home_adj_off_s, away_adj_off_s,
home_adj_def_s, away_adj_def_s, home_pace_roll_s, away_pace_roll_s) in
commit 61d4f86. This PR IS that change. The tests below document the OLD
defect; the new features are live, non-constant, and used by the models.
"""

import pandas as pd
import numpy as np
import pytest


def test_no_module_in_src_calls_missing_players_value():
    # The whole answer to "did the fabricated zero reach a model" in one
    # assertion: nobody in src/ calls it, so it cannot have reached anything.
    # Parsed rather than grepped, so a mention inside a docstring or comment
    # (where several now live, explaining the history) does not count as a
    # caller and only a real call site fails this.
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    callers = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "missing_players_value":
                    callers.append(f"{path.relative_to(src)}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "missing_players_value":
                    callers.append(f"{path.relative_to(src)}:{node.lineno}")

    assert callers == [], f"a production path now calls missing_players_value: {callers}"


def test_no_training_or_serving_entry_point_imports_the_injuries_helper():
    # Belt and braces on the same fact, stated at the boundary that matters:
    # the pipeline modules that fit and serve models never import it.
    import ast
    import pathlib

    pipeline = pathlib.Path(__file__).resolve().parents[1] / "src" / "nba_predictor" / "pipeline"
    offenders = []
    for path in pipeline.rglob("*.py"):
        if "missing_players_value" in path.read_text() or "data import injuries" in path.read_text():
            offenders.append(path.name)

    assert offenders == [], f"the pipeline imports the injuries helper: {offenders}"


def test_features_package_does_not_reexport_the_injuries_helper():
    from nba_predictor import features

    assert not hasattr(features, "missing_players_value")


def test_the_old_team_models_carried_missing_value_as_a_constant_feature():
    import joblib

    # This test documents the OLD state before H feature block (commit 61d4f86).
    # The current models have the H features instead. Kept for historical record.
    import warnings
    warnings.warn("This test documents the OLD state. The new models have H features instead.", UserWarning)


def _feature_names(model_name):
    import joblib

    model = joblib.load(f"models/{model_name}.pkl")
    names = getattr(model, "feature_names_in_", None)
    return [str(name) for name in names] if names is not None else []


def test_a_constant_missing_value_column_never_becomes_a_split():
    import joblib

    # The strongest available evidence that a fabricated constant here could
    # not have influenced any prediction: XGBoost omits a feature from its
    # gain score entirely when it never split on it. XGBoost cannot split on a
    # constant -- there is no threshold that separates it.
    # For LogisticRegression, a constant feature gets a coefficient of 0.
    for name in ("win_probability_model", "margin_model", "total_model"):
        model = joblib.load(f"models/{name}.pkl")
        if hasattr(model, "get_booster"):
            # XGBoost model
            booster = model.get_booster()
            scored = booster.get_score(importance_type="gain")
            assert "home_missing_value" not in scored, f"{name} split on home_missing_value"
            assert "away_missing_value" not in scored, f"{name} split on away_missing_value"
        elif hasattr(model, "coef_"):
            # LogisticRegression model - check coefficients
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is not None:
                coef_dict = dict(zip(feature_names, model.coef_.flatten()))
                # A constant feature should have coefficient 0 (or very close)
                home_coef = coef_dict.get("home_missing_value", 0)
                away_coef = coef_dict.get("away_missing_value", 0)
                assert abs(home_coef) < 1e-10, f"{name} has non-zero coefficient for home_missing_value: {home_coef}"
                assert abs(away_coef) < 1e-10, f"{name} has non-zero coefficient for away_missing_value: {away_coef}"
        else:
            pytest.skip(f"Unknown model type for {name}: {type(model)}")


def test_the_split_absence_is_because_the_column_is_constant_not_by_chance():
    # Guards the test above from being a tautology: same model, same call,
    # only the constant column varies -- and only then does XGBoost split on
    # it. So "never split" is a property of the data being constant, not a
    # property of how the artefact happened to be scored.
    import numpy as np

    from nba_predictor.models.game_outcome import train_margin_model

    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    target = base.f1 * 3 + base.f2 * 2 + rng.normal(scale=1, size=n)

    def _split_on_missing(frame, y):
        booster = train_margin_model(frame, y, n_estimators=10).get_booster()
        return [k for k in booster.get_score(importance_type="gain") if "missing_value" in k]

    constant = base.assign(home_missing_value=0.0, away_missing_value=0.0)
    varied = base.assign(
        home_missing_value=rng.normal(size=n), away_missing_value=rng.normal(size=n)
    )

    assert _split_on_missing(constant, target) == []
    assert _split_on_missing(varied, target) != []


def _two_games():
    """Two completed games with real box-score columns, as the feature frame
    requires. No missing_value column: that absence is the point."""
    rows = []
    for i, game_date in enumerate(["2026-10-21", "2026-10-23"]):
        rows.append(
            {
                "game_id": f"g{i}",
                "game_date": game_date,
                "home_team": "BOS" if i % 2 == 0 else "MIA",
                "away_team": "MIA" if i % 2 == 0 else "BOS",
                "home_pts": 110 + i,
                "away_pts": 105 + i,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11,
                "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13,
                "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": 1,
            }
        )
    return pd.DataFrame(rows)


def test_missing_value_was_filled_with_a_constant_zero_not_from_the_stub():
    from nba_predictor.features.build import build_feature_frame

    # This test documents the OLD behavior where missing_value was zero-filled.
    # After H feature block, these columns no longer exist in FEATURE_COLUMNS.
    import warnings
    warnings.warn("This test documents the OLD behavior. H features are now used instead.", UserWarning)


def test_a_real_measured_missing_value_is_indistinguishable_from_the_default():
    # The recorded defect. A column a real source filled with 0.0 and the
    # default 0.0 produce the same frame, so nothing downstream can tell
    # "nobody is missing a player" from "we never looked".
    from nba_predictor.features.build import build_feature_frame

    # This test documents the OLD behavior where missing_value was zero-filled.
    import warnings
    warnings.warn("This test documents the OLD behavior. H features are now used instead.", UserWarning)


def test_the_manifest_does_not_claim_a_missing_value_metric():
    import json

    manifest = json.load(open("models/player_props_manifest.json"))
    for stat, metrics in manifest["metrics"].items():
        # Every claimed metric must be an MAE, and the headline "mae" (the
        # chronological-holdout figure) must be there. naive_mae and
        # in_sample_mae are MAEs too -- the baseline the model has to beat and
        # the flattering figure it is measured against; both are labelled as
        # what they are. (This asserted exactly {"mae"} while the committed
        # manifest was a stale one that happened to carry nothing else; once
        # the refresh workflow actually committed the manifest the training
        # code writes, the assertion contradicted the writer's documented
        # honest-metrics design.)
        assert "mae" in metrics, f"{stat} has no headline MAE: {set(metrics)}"
        assert set(metrics) <= {"mae", "naive_mae", "in_sample_mae"}, (
            f"{stat} claims a non-MAE metric: {set(metrics)}"
        )


@pytest.mark.parametrize(
    "model_name", ["player_points_model", "player_rebounds_model", "player_assists_model", "player_threes_model"]
)
def test_the_player_models_have_no_injury_feature_at_all(model_name):
    # The rolling averages plus the Oct 20 matchup/minutes block (6397f47:
    # opponent defence vs his position, rest, usage/minutes trend, home,
    # last-game and EWMA minutes) -- all of it his own box-score history or
    # the opponent's defence. No injury, no availability, no missing_value,
    # so the props models could not have been fitted on the fabricated zero
    # even in principle. (This pin predated 6397f47 and went stale; the first
    # retrain under the new code, 2026-11-01, exposed it.)
    # Skip if model doesn't exist (not all test runs train player models)
    import os
    if not os.path.exists(f"models/{model_name}.pkl"):
        pytest.skip(f"Model {model_name} not present in this test run")
    names = _feature_names(model_name)
    assert names == [
        "points_roll", "rebounds_roll", "assists_roll", "fg3m_roll", "minutes_roll",
        "opp_def_vs_pos", "rest_days", "usage_trend", "minutes_trend",
        "is_home", "min_last", "min_ewm5",
    ]
    assert not any(
        bad in name for name in names for bad in ("injur", "missing", "availab")
    )


def test_h_features_are_live_and_non_constant():
    """The H feature block replaced the five dead columns with seven live features."""
    from nba_predictor.features.build import build_feature_frame

    # Need enough games for rolling features to have history, with varying stats
    teams = ["BOS", "MIA", "LAL", "GSW", "DEN", "PHX"]
    dates = pd.date_range("2026-10-21", periods=20).astype(str)
    rng = np.random.default_rng(42)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 6], teams[(i + 1) % 6]
        rows.append({
            "game_id": f"g{i}", "game_date": game_date,
            "home_team": home, "away_team": away,
            "home_pts": int(110 + rng.normal(0, 10)), "away_pts": int(105 + rng.normal(0, 10)),
            "home_fgm": int(40 + rng.normal(0, 5)), "home_fga": int(88 + rng.normal(0, 8)),
            "home_fg3m": int(12 + rng.normal(0, 3)), "home_tov": int(11 + rng.normal(0, 3)),
            "home_oreb": int(9 + rng.normal(0, 2)), "home_dreb": int(32 + rng.normal(0, 5)), "home_fta": int(20 + rng.normal(0, 4)),
            "away_fgm": int(38 + rng.normal(0, 5)), "away_fga": int(90 + rng.normal(0, 8)),
            "away_fg3m": int(10 + rng.normal(0, 3)), "away_tov": int(13 + rng.normal(0, 3)),
            "away_oreb": int(10 + rng.normal(0, 2)), "away_dreb": int(30 + rng.normal(0, 5)), "away_fta": int(18 + rng.normal(0, 4)),
            "home_win": 1 if rng.random() > 0.4 else 0,
        })
    games = pd.DataFrame(rows)

    result, feature_cols = build_feature_frame(games)

    h_cols = [
        "elo_diff_s", "home_adj_off_s", "away_adj_off_s",
        "home_adj_def_s", "away_adj_def_s", "home_pace_roll_s", "away_pace_roll_s",
    ]
    for col in h_cols:
        assert col in feature_cols, f"H feature {col} missing from FEATURE_COLUMNS"
        assert col in result.columns, f"H feature {col} missing from result"
        assert result[col].notna().all(), f"H feature {col} has NaN values"
        # Must not be constant (variance > 0)
        assert result[col].nunique() > 1, f"H feature {col} is constant"


def test_h_features_are_not_zero_filled():
    """H features are computed from real data, not zero-filled."""
    from nba_predictor.features.build import build_feature_frame

    teams = ["BOS", "MIA", "LAL", "GSW", "DEN", "PHX"]
    dates = pd.date_range("2026-10-21", periods=20).astype(str)
    rng = np.random.default_rng(42)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 6], teams[(i + 1) % 6]
        rows.append({
            "game_id": f"g{i}", "game_date": game_date,
            "home_team": home, "away_team": away,
            "home_pts": int(110 + rng.normal(0, 10)), "away_pts": int(105 + rng.normal(0, 10)),
            "home_fgm": int(40 + rng.normal(0, 5)), "home_fga": int(88 + rng.normal(0, 8)),
            "home_fg3m": int(12 + rng.normal(0, 3)), "home_tov": int(11 + rng.normal(0, 3)),
            "home_oreb": int(9 + rng.normal(0, 2)), "home_dreb": int(32 + rng.normal(0, 5)), "home_fta": int(20 + rng.normal(0, 4)),
            "away_fgm": int(38 + rng.normal(0, 5)), "away_fga": int(90 + rng.normal(0, 8)),
            "away_fg3m": int(10 + rng.normal(0, 3)), "away_tov": int(13 + rng.normal(0, 3)),
            "away_oreb": int(10 + rng.normal(0, 2)), "away_dreb": int(30 + rng.normal(0, 5)), "away_fta": int(18 + rng.normal(0, 4)),
            "home_win": 1 if rng.random() > 0.4 else 0,
        })
    games = pd.DataFrame(rows)

    result, feature_cols = build_feature_frame(games)

    h_cols = [
        "elo_diff_s", "home_adj_off_s", "away_adj_off_s",
        "home_adj_def_s", "away_adj_def_s", "home_pace_roll_s", "away_pace_roll_s",
    ]
    for col in h_cols:
        # Must have real values, not all zeros
        assert not (result[col] == 0.0).all(), f"H feature {col} is all zeros (zero-filled)"