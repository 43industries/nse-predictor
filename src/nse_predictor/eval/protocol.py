"""Time-split protocol with label-horizon embargo / purge."""

from __future__ import annotations

import pandas as pd

from nse_predictor.eval.walk_forward import time_split

LABEL_HORIZON_DAYS = 5


def _with_dates(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return d


def apply_embargo(
    df: pd.DataFrame,
    split_end: str | pd.Timestamp,
    horizon_days: int = LABEL_HORIZON_DAYS,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Keep rows whose trade_date is at least ``horizon_days`` before ``split_end``."""
    d = _with_dates(df, date_col)
    if d is None or d.empty:
        return d
    cutoff = pd.Timestamp(split_end) - pd.Timedelta(days=int(horizon_days))
    return d[d[date_col] <= cutoff].copy()


def purge_after(
    df: pd.DataFrame,
    split_start: str | pd.Timestamp,
    horizon_days: int = LABEL_HORIZON_DAYS,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Drop the first ``horizon_days`` after a split start (purge gap)."""
    d = _with_dates(df, date_col)
    if d is None or d.empty:
        return d
    cutoff = pd.Timestamp(split_start) + pd.Timedelta(days=int(horizon_days))
    return d[d[date_col] >= cutoff].copy()


def time_split_embargoed(
    df: pd.DataFrame,
    train_end: str = "2019-12-31",
    valid_end: str = "2022-12-31",
    horizon_days: int = LABEL_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Calendar split then embargo train/valid for fit; purge early test rows.

    Returns (train_fit, valid_fit, test_eval, protocol_meta).
    """
    train_raw, valid_raw, test = time_split(df, train_end=train_end, valid_end=valid_end)
    train = apply_embargo(train_raw, train_end, horizon_days=horizon_days)
    valid = apply_embargo(
        purge_after(valid_raw, train_end, horizon_days=horizon_days),
        valid_end,
        horizon_days=horizon_days,
    )
    test_eval = purge_after(test, valid_end, horizon_days=horizon_days)
    meta = {
        "train_end": train_end,
        "valid_end": valid_end,
        "horizon_days": int(horizon_days),
        "embargo": True,
        "n_train_raw": int(len(train_raw)),
        "n_train_fit": int(len(train)),
        "n_valid_raw": int(len(valid_raw)),
        "n_valid_fit": int(len(valid)),
        "n_test_raw": int(len(test)),
        "n_test_eval": int(len(test_eval)),
    }
    return train, valid, test_eval, meta


def embargo_train_tail(
    train: pd.DataFrame,
    horizon_days: int = LABEL_HORIZON_DAYS,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Embargo relative to the last date in ``train`` (walk-forward folds)."""
    if train is None or train.empty:
        return train
    end = pd.to_datetime(train[date_col]).max()
    return apply_embargo(train, end, horizon_days=horizon_days, date_col=date_col)
