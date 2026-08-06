"""Feature engineering: era labels, age, imputation, scaling, target, splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    NUMERIC_FEATURES,
    SEED,
    TEST_FRAC,
    VAL_FRAC,
    VALUE_ALWAYS,
    VALUE_CONDITIONAL,
)

POSITIONS = ["G", "F", "C", "Unknown"]


def decade_of(season_year: str) -> str:
    """'2002-03' -> '2000s'."""
    start = int(str(season_year)[:4])
    return f"{start // 10 * 10}s"


def compute_age(df: pd.DataFrame) -> pd.Series:
    """Age at season start = season_end_year - birth_year; decade-mean fallback."""
    season_end = df["season_year"].str[:4].astype(int)
    birth = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
    age = season_end - birth
    df_tmp = pd.DataFrame({"age": age, "decade": df["season_year"].map(decade_of)})
    dec_mean = df_tmp.groupby("decade")["age"].transform("mean")
    overall_mean = age.mean(skipna=True)
    return age.fillna(dec_mean).fillna(overall_mean).fillna(0.0)


def build_value_target(df: pd.DataFrame) -> pd.Series:
    """Per-season z-score of a box-score composite.

    raw = pts + reb + ast + (stl + blk - tov if recorded else 0)
    Then z-scored within each season_year so the target is era-normalized.
    """
    raw = pd.Series(0.0, index=df.index, dtype=np.float64)
    for col in VALUE_ALWAYS:
        raw = raw + df[col].fillna(0.0).astype(np.float64)
    for col, w in VALUE_CONDITIONAL.items():
        raw = raw + df[col].fillna(0.0).astype(np.float64) * w

    def zscore(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std > 0 else s * 0.0

    return raw.groupby(df["season_year"]).transform(zscore)


def build_position_oh(df: pd.DataFrame) -> pd.DataFrame:
    pos = df["position"].fillna("Unknown").where(
        df["position"].fillna("Unknown").isin(POSITIONS), "Unknown"
    )
    return pd.get_dummies(pos, prefix="pos").reindex(
        columns=[f"pos_{p}" for p in POSITIONS], fill_value=0
    ).astype(np.float32)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds era, age, position one-hot; returns a copy with FEATURES + TARGET + ERA columns."""
    out = df.copy()
    out["era"] = out["season_year"].map(decade_of)
    out["age"] = compute_age(out)

    # Impute numeric features with decade mean (historical reality: stl/blk/3pt
    # not recorded early; advanced metrics absent pre-1970s). If an entire era
    # lacks a stat (fg3_pct pre-1979, stl/blk pre-1973, tov/net_rating/usg pre-1977),
    # the decade mean is NaN -> fall back to the global column mean.
    for col in NUMERIC_FEATURES:
        dec_mean = out.groupby("era")[col].transform("mean")
        out[col] = out[col].fillna(dec_mean).fillna(out[col].mean())

    pos_oh = build_position_oh(out)
    out = pd.concat([out, pos_oh], axis=1)

    out["value_target"] = build_value_target(out)
    return out


def make_splits(df: pd.DataFrame):
    """Stratified train/val/test split by era. Returns (train, val, test) indices."""
    era_series = df["era"].astype("category").cat.codes
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, stratify=era_series, random_state=SEED
    )
    train_era = era_series.iloc[idx_train]
    idx_train, idx_val = train_test_split(
        idx_train, test_size=VAL_FRAC, stratify=train_era, random_state=SEED
    )
    return idx_train, idx_val, idx_test


def feature_columns(df: pd.DataFrame) -> list:
    cols = list(NUMERIC_FEATURES) + ["age"] + [f"pos_{p}" for p in POSITIONS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return cols


def fit_scaler(df: pd.DataFrame, idx_train, cols):
    scaler = StandardScaler()
    scaler.fit(df.loc[idx_train, cols])
    return scaler


def transform_features(df: pd.DataFrame, scaler, cols) -> np.ndarray:
    return scaler.transform(df[cols]).astype(np.float32)
