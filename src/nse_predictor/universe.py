"""Liquid universe selection shared by train and live scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nse_predictor.config import ROOT, settings


def pick_universe(prices: pd.DataFrame, n: int = 15) -> list[str]:
    code = prices["stock_code"].astype(str)
    exclude = code.str.contains(
        r"-R$|^\^|NASI|N20I|NSE20|NSE 20|NSE-20", case=False, na=False, regex=True
    )
    eq = prices.loc[~exclude].copy()
    stats = eq.groupby("stock_code").agg(
        avg_volume=("volume", "mean"),
        n_days=("trade_date", "count"),
        last=("trade_date", "max"),
    )
    stats = stats[(stats["n_days"] >= 1500) & (stats["last"] >= pd.Timestamp("2023-01-01"))]
    liq = stats.sort_values("avg_volume", ascending=False).head(n).index.tolist()
    if len(liq) < 8:
        stats = (
            eq.groupby("stock_code")["volume"].mean().sort_values(ascending=False).head(n)
        )
        liq = stats.index.tolist()
    present = set(code)
    for idx in ["^NASI", "^N20I", "NASI", "NSE20"]:
        if idx in present and idx not in liq:
            liq.append(idx)
    return liq


def load_universe(prices: pd.DataFrame | None = None) -> list[str]:
    """Prefer universe from last evaluation; fall back to liquidity screen."""
    ev_path = ROOT / "reports" / "evaluation_results.json"
    if ev_path.exists():
        try:
            data = json.loads(ev_path.read_text(encoding="utf-8"))
            uni = data.get("universe") or []
            if uni:
                return list(uni)
        except Exception:  # noqa: BLE001
            pass
    if prices is None:
        path = settings.data_processed_dir / "prices.parquet"
        prices = pd.read_parquet(path)
    return pick_universe(prices)
