"""Simple signal backtest with NSE-like costs (research illustration only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_long_flat(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = 50.0,
    lot_size: int = 100,
) -> dict:
    """Long when pred_label==up else flat. One stock panel.

    cost_bps applied on each position change (round-turn approximation / 2 per side).
    """
    df = prices.merge(signals, on=["trade_date", "stock_code"], how="inner")
    df = df.sort_values("trade_date")
    if df.empty:
        return {"ok": False, "error": "empty join"}

    pos = (df["pred_label"].astype(str).str.lower() == "up").astype(int).values
    ret = df["price"].pct_change().fillna(0).values
    # Position held from prior signal
    held = np.roll(pos, 1)
    held[0] = 0
    switch = np.abs(np.diff(held, prepend=held[0]))
    costs = switch * (cost_bps / 10000.0)
    strat = held * ret - costs
    equity = np.cumprod(1 + strat)
    bh = np.cumprod(1 + ret)
    dd = equity / np.maximum.accumulate(equity) - 1
    wins = strat[held == 1]
    return {
        "ok": True,
        "total_return": float(equity[-1] - 1),
        "buy_hold_return": float(bh[-1] - 1),
        "max_drawdown": float(dd.min()),
        "win_rate": float((wins > 0).mean()) if len(wins) else float("nan"),
        "n_trades": int(switch.sum()),
        "lot_size_note": f"Signals sized in multiples of {lot_size} shares conceptually; returns are fractional.",
        "cost_bps": cost_bps,
    }
