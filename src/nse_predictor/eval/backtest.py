"""Simple signal backtest with NSE-like costs (research illustration only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DISCLAIMER = "research_only_not_for_selection"


def _strategy_arrays(
    joined: pd.DataFrame,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return strat_rets, buyhold_rets, held, switch counts path."""
    pos = (joined["pred_label"].astype(str).str.lower() == "up").astype(int).values
    ret = joined["price"].pct_change().fillna(0).values
    held = np.roll(pos, 1)
    held[0] = 0
    switch = np.abs(np.diff(held, prepend=held[0]))
    strat = held * ret - switch * (cost_bps / 10000.0)
    return strat, ret, held, switch


def _join_one(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    return (
        prices.merge(signals, on=["trade_date", "stock_code"], how="inner")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def simulate_long_flat(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = 50.0,
    lot_size: int = 100,
) -> dict:
    """Long when pred_label==up else flat. One stock panel."""
    df = _join_one(prices, signals)
    if df.empty:
        return {"ok": False, "error": "empty join"}
    strat, ret, held, switch = _strategy_arrays(df, cost_bps)
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
        "lot_size_note": (
            f"Signals sized in multiples of {lot_size} shares conceptually; returns are fractional."
        ),
        "cost_bps": cost_bps,
    }


def simulate_book_long_flat(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = 50.0,
) -> dict:
    """Equal-weight long/flat book across names (flat → flat). Research-only."""
    need = {"trade_date", "stock_code", "price"}
    if not need.issubset(prices.columns):
        return {"ok": False, "error": "prices missing columns", "disclaimer": DISCLAIMER}
    if "pred_label" not in signals.columns:
        return {"ok": False, "error": "signals missing pred_label", "disclaimer": DISCLAIMER}

    px = prices[["trade_date", "stock_code", "price"]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    sig = signals[["trade_date", "stock_code", "pred_label"]].copy()
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])

    per_name: list[dict] = []
    daily_rets: list[pd.Series] = []
    bh_parts: list[pd.Series] = []

    for code, g_sig in sig.groupby("stock_code"):
        g_px = px[px["stock_code"] == code]
        df = _join_one(g_px, g_sig)
        if df.empty:
            continue
        strat, ret, _held, switch = _strategy_arrays(df, cost_bps)
        equity = np.cumprod(1 + strat)
        per_name.append(
            {
                "stock_code": str(code),
                "total_return": float(equity[-1] - 1),
                "n_trades": int(switch.sum()),
            }
        )
        idx = pd.to_datetime(df["trade_date"])
        daily_rets.append(pd.Series(strat, index=idx))
        bh_parts.append(pd.Series(ret, index=idx))

    if not daily_rets:
        return {"ok": False, "error": "no scorable names", "disclaimer": DISCLAIMER}

    book = pd.concat(daily_rets, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    equity = np.cumprod(1 + book.values)
    dd = equity / np.maximum.accumulate(equity) - 1
    bh = pd.concat(bh_parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    bh_eq = np.cumprod(1 + bh.reindex(book.index).fillna(0).values)

    return {
        "ok": True,
        "disclaimer": DISCLAIMER,
        "n_names": int(len(per_name)),
        "total_return": float(equity[-1] - 1),
        "buy_hold_return": float(bh_eq[-1] - 1),
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "n_trades": int(sum(p["n_trades"] for p in per_name)),
        "cost_bps": cost_bps,
        "mean_name_total_return": float(np.mean([p["total_return"] for p in per_name])),
    }
