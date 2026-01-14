from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .utils import remove_outliers_zscore, select_columns_by_keywords, smart_interpolation


def load_long_csv(path: Path) -> pd.DataFrame:
    """Load v1-style long CSV: columns [name, timestamp, value] with header row."""
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)
    # tolerate either headerless or headered
    if set(df.columns) >= {"name", "timestamp", "value"}:
        pass
    else:
        # fallback to v1 behavior
        df = pd.read_csv(path, names=["name", "timestamp", "value"], skiprows=1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def pivot_minute(df_long: pd.DataFrame, floor_freq: str = "min") -> pd.DataFrame:
    df = df_long.copy()
    df["timestamp"] = df["timestamp"].dt.floor(floor_freq)

    wide = (
        df.pivot_table(index="timestamp", columns="name", values="value", aggfunc="first")
        .sort_index()
        .copy()
    )

    # Ensure datetime index
    wide.index = pd.to_datetime(wide.index)
    return wide


def preprocess_wide(df_wide: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    df = df_wide.copy()
    df = smart_interpolation(df)
    df = remove_outliers_zscore(df, threshold=cfg.preprocessing.outlier_z_threshold)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")
    return df


def prepare_and_save(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads raw CSV, preprocesses, selects forecasting/classification features and writes CSVs."""
    paths = cfg.paths
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    df_long = load_long_csv(paths.raw_csv_path)
    df_wide = pivot_minute(df_long, floor_freq=cfg.preprocessing.resample_freq)
    df = preprocess_wide(df_wide, cfg)

    forecasting_cols = select_columns_by_keywords(
        df.columns,
        include_keywords=cfg.preprocessing.forecasting_include_keywords,
        exclude_keywords=cfg.preprocessing.forecasting_exclude_keywords,
    )
    classification_cols = select_columns_by_keywords(
        df.columns,
        include_keywords=cfg.preprocessing.classification_include_keywords,
        exclude_keywords=cfg.preprocessing.classification_exclude_keywords,
    )

    df_forecasting = df[forecasting_cols].copy()
    df_classification = df[classification_cols].copy()

    # Ensure numeric where possible
    df_forecasting = df_forecasting.apply(pd.to_numeric, errors="coerce")
    df_classification = df_classification.apply(pd.to_numeric, errors="coerce")

    df_forecasting.to_csv(paths.processed_forecasting_csv, index=True)
    df_classification.to_csv(paths.processed_classification_csv, index=True)

    print(f"Saved forecasting features: {paths.processed_forecasting_csv}")
    print(f"Saved classification features: {paths.processed_classification_csv}")

    return df_forecasting, df_classification
