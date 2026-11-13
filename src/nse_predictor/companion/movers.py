"""Session gainers / losers / movers from processed prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nse_predictor.config import settings


def latest_equity_moves(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """One row per stock: latest session move for gainers / losers / movers boards."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    df = prices.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df = df[~df["stock_code"].str.contains(r"-R$|^\^", regex=True)]
    if universe:
        uni = {str(c).upper() for c in universe}
        df = df[df["stock_code"].isin(uni)]
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "price"])
    latest = df.sort_values("trade_date").groupby("stock_code", as_index=False).tail(1)
    if "ret_1d" not in latest.columns or latest["ret_1d"].isna().all():
        if "change_pct" in latest.columns:
            latest["ret_1d"] = pd.to_numeric(latest["change_pct"], errors="coerce") / 100.0
        else:
            latest["ret_1d"] = pd.NA
    latest["ret_1d"] = pd.to_numeric(latest["ret_1d"], errors="coerce")
    latest["volume"] = pd.to_numeric(latest.get("volume", 0), errors="coerce").fillna(0)
    latest["abs_ret"] = latest["ret_1d"].abs()
    return latest.dropna(subset=["ret_1d"]).reset_index(drop=True)


def _row(r: Any) -> dict[str, Any]:
    return {
        "stock_code": str(r.stock_code),
        "price": float(r.price) if pd.notna(r.price) else None,
        "ret_1d": float(r.ret_1d),
        "volume": float(r.volume) if pd.notna(getattr(r, "volume", None)) else 0.0,
        "trade_date": str(pd.to_datetime(r.trade_date).date())
        if pd.notna(r.trade_date)
        else None,
    }


def build_movers(
    *,
    top_n: int = 5,
    processed: Path | None = None,
    universe: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(processed or settings.data_processed_dir) / "prices.parquet"
    if not path.exists():
        return {
            "as_of": None,
            "gainers": [],
            "losers": [],
            "movers": [],
            "disclaimer": settings.disclaimer,
            "note": "prices.parquet missing",
        }
    prices = pd.read_parquet(path)
    moves = latest_equity_moves(prices, universe=universe)
    if moves.empty:
        return {
            "as_of": None,
            "gainers": [],
            "losers": [],
            "movers": [],
            "disclaimer": settings.disclaimer,
        }
    as_of = pd.to_datetime(moves["trade_date"]).max()
    as_of_s = str(as_of.date()) if pd.notna(as_of) else None
    n = max(1, min(int(top_n), 20))
    gainers = moves.sort_values("ret_1d", ascending=False).head(n)
    losers = moves.sort_values("ret_1d", ascending=True).head(n)
    movers = moves.sort_values(["abs_ret", "volume"], ascending=[False, False]).head(n)
    return {
        "as_of": as_of_s,
        "gainers": [_row(r) for r in gainers.itertuples()],
        "losers": [_row(r) for r in losers.itertuples()],
        "movers": [_row(r) for r in movers.itertuples()],
        "disclaimer": settings.disclaimer,
        "note": "Price moves — not model signals.",
    }
