"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from era_translator.features import (
    build_position_oh,
    build_value_target,
    compute_age,
    decade_of,
    engineer_features,
    feature_columns,
    make_splits,
)


def test_decade_of():
    assert decade_of("1946-47") == "1940s"
    assert decade_of("2002-03") == "2000s"
    assert decade_of("2025-26") == "2020s"


def test_compute_age_known_birth():
    df = pd.DataFrame(
        {"season_year": ["2002-03", "1990-91", "2002-03"],
         "birth_date": ["1984-12-30", "1963-02-17", None]}
    )
    ages = compute_age(df)
    assert ages.iloc[0] == 18  # 2003 - 1985
    assert ages.iloc[1] == 27  # 1991 - 1964
    assert ages.iloc[2] == pytest.approx(ages.iloc[0])  # decade mean fallback


def test_build_value_target_per_season_zscore():
    df = pd.DataFrame(
        {
            "season_year": ["2020-21"] * 3 + ["2010-11"] * 3,
            "avg_pts": [10.0, 20.0, 30.0, 15.0, 15.0, 15.0],
            "avg_reb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "avg_ast": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "avg_stl": [None] * 6,
            "avg_blk": [None] * 6,
            "avg_tov": [None] * 6,
        }
    )
    t = build_value_target(df)
    s = np.std([10.0, 20.0, 30.0])  # ddof=0
    assert t.iloc[0] == pytest.approx((10.0 - 20.0) / s, abs=1e-6)
    assert t.iloc[1] == pytest.approx(0.0, abs=1e-6)
    assert t.iloc[2] == pytest.approx((30.0 - 20.0) / s, abs=1e-6)
    assert t.iloc[3:].abs().max() == pytest.approx(0.0, abs=1e-6)


def test_engineer_features_no_nan():
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season_year": ["1960-61", "2000-01", "2020-21"],
            "gp": [70, 82, 75],
            "avg_min": [30.0, 32.0, 34.0],
            "avg_pts": [20.0, 22.0, 25.0],
            "avg_reb": [10.0, 8.0, 7.0],
            "avg_ast": [3.0, 5.0, 7.0],
            "avg_stl": [None, 1.5, 1.2],
            "avg_blk": [None, 0.5, 0.6],
            "avg_tov": [None, 2.5, 2.8],
            "fg_pct": [0.45, 0.46, 0.48],
            "fg3_pct": [None, 0.35, 0.38],
            "ft_pct": [0.75, 0.78, 0.80],
            "avg_net_rating": [None, 3.0, 4.0],
            "avg_ts_pct": [0.50, 0.55, 0.58],
            "avg_usg_pct": [None, 25.0, 27.0],
            "birth_date": ["1940-01-01", "1980-01-01", "1998-01-01"],
            "position": ["C", "F", "G"],
        }
    )
    out = engineer_features(df)
    cols = feature_columns(out)
    assert out[cols].isna().sum().sum() == 0
    assert set(cols) >= {"avg_min", "gp", "fg_pct", "age", "pos_G", "pos_F", "pos_C", "pos_Unknown"}


def test_make_splits_stratified_disjoint():
    df = pd.DataFrame(
        {"era": np.repeat(["1940s", "2000s", "2020s"], 50), "x": np.arange(150)}
    )
    idx_tr, idx_va, idx_te = make_splits(df)
    assert len(set(idx_tr) & set(idx_va)) == 0
    assert len(set(idx_tr) & set(idx_te)) == 0
    assert len(set(idx_va) & set(idx_te)) == 0
    assert len(idx_tr) + len(idx_va) + len(idx_te) == 150
    for idx in (idx_tr, idx_va, idx_te):
        assert set(df.iloc[idx]["era"]) == {"1940s", "2000s", "2020s"}


def test_build_position_oh():
    df = pd.DataFrame({"position": ["G", "F", "C", None, "PG"]})
    oh = build_position_oh(df)
    assert list(oh.columns) == ["pos_G", "pos_F", "pos_C", "pos_Unknown"]
    assert oh.iloc[0].tolist() == [1, 0, 0, 0]
    assert oh.iloc[3].tolist() == [0, 0, 0, 1]  # None -> Unknown
    assert oh.iloc[4].tolist() == [0, 0, 0, 1]  # unknown position -> Unknown
