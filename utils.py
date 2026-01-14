from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def smart_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate numeric columns, forward-fill low-cardinality 'status-like' columns."""
    df = ensure_datetime_index(df.copy())

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_count = non_null.nunique()
        total_count = len(non_null)

        # status-like: low cardinality, mostly repeats
        if unique_count <= 10 and (unique_count / max(total_count, 1)) < 0.1:
            df[col] = series.ffill()
            continue

        # numeric interpolation
        if pd.api.types.is_numeric_dtype(series):
            df[col] = series.interpolate(method="time")

    return df


def remove_outliers_zscore(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Clip outliers by replacing extreme z-scores with the column mean."""
    if df is None:
        raise ValueError("remove_outliers_zscore got None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"remove_outliers_zscore expected DataFrame, got {type(df)}")

    df_clean = df.copy()
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        mean = df_clean[col].mean()
        std = df_clean[col].std()
        if std == 0 or np.isnan(std):
            continue

        z = (df_clean[col] - mean) / std
        df_clean.loc[z.abs() > threshold, col] = mean

    return df_clean


def select_columns_by_keywords(
    columns: Iterable[str],
    include_keywords: Iterable[str],
    exclude_keywords: Iterable[str] = (),
) -> list[str]:
    include = [k.lower() for k in include_keywords]
    exclude = [k.lower() for k in exclude_keywords]

    selected: list[str] = []
    for col in columns:
        col_l = col.lower()
        if include and not any(k in col_l for k in include):
            continue
        if exclude and any(k in col_l for k in exclude):
            continue
        selected.append(col)
    return selected


def create_windows(X: np.ndarray, y: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - window_size + 1):
        Xs.append(X[i : i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(Xs), np.array(ys)


def safe_print_exception(prefix: str, exc: Exception) -> None:
    print(f"{prefix}: {type(exc).__name__}: {exc}")
