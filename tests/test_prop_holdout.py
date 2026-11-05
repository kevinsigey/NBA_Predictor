"""Chronological holdout for the prop regressors (spec section 7, gap G4).

The prop models shipped with in-sample metrics only — `ingest.py` says so in the
manifest (`training={"in_sample_metrics": True}`). A model scored on the rows it
was fitted against cannot fail its own evaluation, so those numbers have never
been evidence of anything.

The fixture below carries real signal on purpose. An earlier version of this file
measured pure noise (`rng.uniform` targets unrelated to any feature), which
produced a holdout MAE *better* than in-sample and a set of numbers that looked
like findings while measuring nothing. A holdout needs a model worth holding out.

Pinned here, each of which can fail:
* the holdout is strictly later than the train window;
* every player-game row lands in exactly one of train/holdout, none dropped;
* holdout features carry training history (a holdout player's rolling means are
  not truncated at the split boundary);
* one split serves all four markets, so their numbers are comparable;
* a missing target column raises rather than reporting `inf`.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.prop_holdout import MARKETS, prop_holdout_metrics


def make_player_games(n_games: int = 120, n_players: int = 24, seed: int = 7) -> pd.DataFrame:
    """Player-games with a learnable relationship, shaped like
    `ingest.to_player_training_frame` (player_id, game_id, game_date, stats)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-10-01", periods=n_games).astype(str)
    for g, day in enumerate(dates):
        for p in range(n_players):
            ability = 12 + (p % 8) * 1.5
            rows.append({
                "player_id": f"p{p:02d}",
                "player_name": f"Player {p}",
                "team": "BOS" if p % 2 else "MIA",
                "game_id": f"g{g}",
                "game_date": day,
                "minutes": 30.0,
                # Target depends on the player's ability, which the rolling
                # features can recover -- so the holdout has something real to
                # generalise to rather than measuring noise.
                "points": float(np.clip(rng.normal(ability, 4), 2, 45)),
                "rebounds": float(np.clip(rng.normal(4 + p % 4, 1.5), 0, 15)),
                "assists": float(np.clip(rng.normal(2 + p % 5, 1.2), 0, 15)),
                "fg3m": float(max(0, rng.poisson(1 + p % 3))),
            })
    return pd.DataFrame(rows)


def dummy_prop_factories():
    from nba_predictor.models.player_props import train_player_stat_model

    def factory(X, y):
        return train_player_stat_model(X, y, n_estimators=20, max_depth=3)

    return {market: factory for market in MARKETS}


def test_holdout_is_strictly_later_than_train():
    res = prop_holdout_metrics(make_player_games(), dummy_prop_factories())
    for market in MARKETS:
        assert res[market]["train_max_date"] < res[market]["holdout_min_date"]


def test_holdout_publishes_the_conformal_cold_start():
    """The cold start the served intervals calibrate from before the live
    counted record builds up: measured on THIS holdout's residuals, never
    invented. Sigma scales must be non-decreasing in the projection (the
    isotonic fit), the quantile a real order statistic of the normalized
    scores, and n the holdout's own size."""
    res = prop_holdout_metrics(make_player_games(), dummy_prop_factories())
    for market in MARKETS:
        block = res[market]["conformal"]
        assert block is not None
        assert block["n_holdout"] == res[market]["n_holdout"]
        assert block["half_width_80"] > 0
        assert block["q80"] > 0
        scales = block["sigma"]["scales"]
        assert scales == sorted(scales)
        assert len(scales) == len(block["sigma"]["edges"]) + 1
        assert block["sigma"]["floor"] > 0


def test_all_four_markets_are_reported_and_nothing_is_dropped():
    df = make_player_games()
    res = prop_holdout_metrics(df, dummy_prop_factories())

    assert set(res) == set(MARKETS)
    for market in MARKETS:
        block = res[market]
        assert block["n_train"] > 0 and block["n_holdout"] > 0
        # Every row that survived feature assembly is in exactly one side. A
        # dropped row would silently shrink the denominator of every MAE.
        assert block["n_train"] + block["n_holdout"] <= len(df)
    assert (
        len({res[m]["train_max_date"] for m in MARKETS}) == 1
        and len({res[m]["holdout_min_date"] for m in MARKETS}) == 1
    ), "the four markets were scored on different splits and are not comparable"


def test_maes_are_finite_positive_and_report_their_gap():
    res = prop_holdout_metrics(make_player_games(), dummy_prop_factories())
    for market in MARKETS:
        block = res[market]
        for key in ("holdout_mae", "in_sample_mae"):
            assert np.isfinite(block[key]), f"{market}.{key} is not finite"
            assert block[key] > 0
        assert block["generalisation_gap"] == pytest.approx(
            block["holdout_mae"] - block["in_sample_mae"]
        )


def test_holdout_features_carry_training_history():
    """A holdout player's rolling features must include its *training* games.

    Built from the holdout slice alone they would be truncated at the split, so
    the first holdout game for every player would have no history -- which no
    serving scenario has, and which would flatter or punish the model at random.
    """
    df = make_player_games()
    from nba_predictor.features.player_stats import build_player_feature_frame
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=0.2)

    combined_frame, cols = build_player_feature_frame(
        pd.concat([train, holdout], ignore_index=True)
    )
    holdout_ids = set(holdout["game_id"])
    holdout_frame = combined_frame[combined_frame["game_id"].isin(holdout_ids)]

    first_holdout_row = holdout_frame.sort_values(["player_id", "game_date"]).iloc[0]
    assert not np.isnan(first_holdout_row[cols[0]]), (
        "a holdout row has no feature history -- the split truncated it"
    )

    # And the naive construction really does lose that history, so this test
    # has teeth. It loses it by *dropping* rows: build_player_feature_frame
    # dropna()s anything without a rolling value, so building from the holdout
    # slice alone discards each player's first holdout game outright.
    naive, _ = build_player_feature_frame(holdout.copy())
    assert len(naive) < len(holdout_frame), (
        f"the naive construction lost {len(holdout) - len(naive)} rows to a truncated "
        "history, so this test would pass against a broken implementation"
    )
    assert set(naive["game_id"]) <= holdout_ids


def test_a_missing_target_column_raises_rather_than_reporting_inf():
    """`inf` reads in a report as a real, terrible number. The honest fact is
    'this market had no target column', which is a different statement."""
    df = make_player_games().drop(columns=["assists"])
    with pytest.raises(KeyError, match="assists"):
        prop_holdout_metrics(df, dummy_prop_factories())


def test_a_missing_factory_raises():
    df = make_player_games()
    factories = dummy_prop_factories()
    del factories["rebounds"]
    with pytest.raises(KeyError, match="rebounds"):
        prop_holdout_metrics(df, factories)


def test_threes_maps_to_the_box_score_column():
    """The market is called "threes"; the box score column is `fg3m`. Getting
    this wrong scores a different column than the site publishes."""
    res = prop_holdout_metrics(make_player_games(), dummy_prop_factories())
    assert "threes" in res
    assert np.isfinite(res["threes"]["holdout_mae"])
    assert res["threes"]["holdout_mae"] != res["assists"]["holdout_mae"]


def test_a_frame_too_small_to_split_raises():
    with pytest.raises(ValueError):
        prop_holdout_metrics(pd.DataFrame({"game_date": ["2024-10-01"]}), dummy_prop_factories())