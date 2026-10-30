"""Per-stock forecast error history helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_error_history(
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Join labelled panel rows with direction predictions."""
    cols = ["trade_date", "stock_code", "y_dir_5d"]
    base = eval_df[cols].copy()
    pred = predictions[["trade_date", "stock_code", "prob_up", "pred_label"]].copy()
    out = base.merge(pred, on=["trade_date", "stock_code"], how="inner")
    out["y_dir_5d"] = out["y_dir_5d"].astype(float)
    label = out["pred_label"].astype(str).str.lower()
    out["pred_up"] = (label == "up").astype(int)
    # Flat abstentions excluded from hit-rate (correct = NA)
    out["correct"] = np.where(
        label == "flat",
        pd.NA,
        out["pred_up"] == out["y_dir_5d"].astype(int),
    )
    out["horizon_days"] = horizon_days
    return out.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def summarize_error_history(
    df: pd.DataFrame,
    stock_code: str,
    lookback_n: int = 60,
) -> dict:
    """Hit rate over the last `lookback_n` labelled forecasts for one ticker."""
    if df is None or df.empty:
        return {
            "stock_code": stock_code.upper(),
            "lookback_n": lookback_n,
            "n": 0,
            "hit_rate": None,
            "recent_streak": 0,
            "note": "No error history — run train_evaluate.py",
        }
    hit = df[df["stock_code"].astype(str).str.upper() == stock_code.upper()].copy()
    hit = hit.dropna(subset=["correct"]).sort_values("trade_date")
    if hit.empty:
        return {
            "stock_code": stock_code.upper(),
            "lookback_n": lookback_n,
            "n": 0,
            "hit_rate": None,
            "recent_streak": 0,
            "note": f"No labelled forecasts for {stock_code.upper()}",
        }
    tail = hit.tail(lookback_n)
    correct = tail["correct"].astype(bool)
    streak = 0
    for val in correct.iloc[::-1]:
        if bool(val):
            streak += 1
        else:
            break
    return {
        "stock_code": stock_code.upper(),
        "lookback_n": lookback_n,
        "n": int(len(tail)),
        "hit_rate": float(correct.mean()),
        "recent_streak": int(streak),
        "as_of_min": str(pd.to_datetime(tail["trade_date"].min()).date()),
        "as_of_max": str(pd.to_datetime(tail["trade_date"].max()).date()),
        "note": None,
    }
