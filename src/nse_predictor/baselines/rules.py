"""Simple baselines every model must beat on unseen data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def always_up(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["trade_date", "stock_code"]].copy()
    out["pred_label"] = "up"
    out["prob_up"] = 1.0
    out["model_name"] = "baseline_always_up"
    return out


def tomorrow_equals_today(df: pd.DataFrame) -> pd.DataFrame:
    """Price baseline: predict next close = today's close => expected return 0."""
    out = df[["trade_date", "stock_code", "price"]].copy()
    out["pred_value"] = out["price"]
    out["pred_ret"] = 0.0
    out["model_name"] = "baseline_tomorrow_eq_today"
    return out


def moving_average_direction(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """Direction baseline: up if short MA > long MA (both known as of t)."""
    g = df.sort_values(["stock_code", "trade_date"]).groupby("stock_code", group_keys=False)
    tmp = df.copy()
    tmp["ma_s"] = g["price"].transform(lambda s: s.rolling(short).mean())
    tmp["ma_l"] = g["price"].transform(lambda s: s.rolling(long).mean())
    out = tmp[["trade_date", "stock_code"]].copy()
    signal = (tmp["ma_s"] > tmp["ma_l"]).astype(float)
    out["pred_label"] = np.where(signal == 1, "up", "down")
    out["prob_up"] = signal
    out["model_name"] = f"baseline_ma_{short}_{long}"
    return out
