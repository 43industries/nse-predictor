"""Time-based and walk-forward validation helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from nse_predictor.eval.metrics import compare_to_always_up, direction_metrics


def time_split(
    df: pd.DataFrame,
    train_end: str = "2019-12-31",
    valid_end: str = "2022-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    train = d[d["trade_date"] <= train_end]
    valid = d[(d["trade_date"] > train_end) & (d["trade_date"] <= valid_end)]
    test = d[d["trade_date"] > valid_end]
    return train, valid, test


def walk_forward_splits(
    df: pd.DataFrame,
    start_year: int = 2015,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Expanding-window yearly folds: train through year Y-1, test year Y."""
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    for year in range(start_year, end_year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        train = d[d["trade_date"] <= train_end]
        test = d[(d["trade_date"] >= test_start) & (d["trade_date"] <= test_end)]
        if train["trade_date"].dt.year.nunique() < min_train_years:
            continue
        if test.empty:
            continue
        yield f"wf_{year}", train, test


def evaluate_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    return compare_to_always_up(direction_metrics(y_true, y_pred, y_prob))


def stress_slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    if period == "covid_2020":
        return d[(d["trade_date"] >= "2020-03-01") & (d["trade_date"] <= "2020-12-31")]
    if period == "fx_2023":
        return d[(d["trade_date"] >= "2023-01-01") & (d["trade_date"] <= "2023-12-31")]
    raise ValueError(period)
