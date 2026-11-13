"""Replay paper fills into a daily NAV series (compute-on-read)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.portfolio.store import default_book, load_book, load_fills


def _prices_panel(processed: Path | None = None) -> pd.DataFrame:
    path = Path(processed or settings.data_processed_dir) / "prices.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "stock_code", "price"])
    df = pd.read_parquet(path, columns=["trade_date", "stock_code", "price"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.upper()
    df = df.dropna(subset=["trade_date", "price"])
    return df.sort_values(["trade_date", "stock_code"])


def build_nav_history(
    *,
    days: int = 90,
    processed: Path | None = None,
) -> dict[str, Any]:
    """Mark-to-market cash+positions after each fill date and through latest prices."""
    fills = load_fills(processed)
    book0 = load_book(processed)
    starting_cash = float(book0.get("cash_kes") or settings.portfolio_starting_cash)
    # Reconstruct starting cash before fills by reversing fills if positions/fills exist
    # Simpler approach: start from default cash and replay all fills from empty book.
    cash = float(settings.portfolio_starting_cash)
    # Prefer explicit starting cash stored when book was reset: use book cash + cost of open
    # positions at avg_cost + fees approximation from fills chronologically from empty.
    cash = float(default_book()["cash_kes"])
    if book0.get("cash_kes") is not None and not fills:
        cash = float(book0["cash_kes"])

    cost_bps = float(book0.get("cost_bps") or settings.portfolio_cost_bps)
    positions: dict[str, dict[str, float | int]] = {}

    # If we have fills, start from portfolio_starting_cash (or last reset cash if no fills
    # consumed it). Use starting cash = current cash + sum(buy notionals+fees) - sum(sell
    # notionals-fees) so replay lands on current book.
    if fills:
        cash = float(settings.portfolio_starting_cash)
        # Infer starting cash from current book + reverse fills
        rev_cash = float(book0.get("cash_kes") or 0.0)
        for f in reversed(fills):
            side = str(f.get("side") or "").lower()
            notional = float(f.get("notional_kes") or (float(f["price"]) * int(f["shares"])))
            fees = float(f.get("fees_kes") or notional * (cost_bps / 10000.0))
            if side == "buy":
                rev_cash += notional + fees
            elif side == "sell":
                rev_cash -= notional - fees
        cash = rev_cash
        positions = {}

    prices = _prices_panel(processed)
    if prices.empty and not fills:
        return {
            "points": [],
            "days": int(days),
            "disclaimer": settings.disclaimer,
            "mode": "paper",
        }

    # Parse fills with dates
    fill_rows: list[dict[str, Any]] = []
    for f in fills:
        ts = pd.to_datetime(f.get("ts"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        # Use EAT calendar date approx via date of timestamp
        d = ts.tz_convert("Africa/Nairobi").date() if ts.tzinfo else ts.date()
        fill_rows.append({**f, "_date": pd.Timestamp(d)})
    fill_rows.sort(key=lambda x: (x["_date"], str(x.get("ts") or "")))

    # Trading dates to mark: fill dates + available price dates after first event
    if fill_rows:
        start = fill_rows[0]["_date"]
    elif not prices.empty:
        start = prices["trade_date"].max() - pd.Timedelta(days=max(1, int(days)))
    else:
        start = pd.Timestamp(datetime.now(timezone.utc).date())

    if not prices.empty:
        trade_dates = sorted(
            {
                pd.Timestamp(d).normalize()
                for d in prices.loc[prices["trade_date"] >= start, "trade_date"].unique()
            }
        )
    else:
        trade_dates = sorted({pd.Timestamp(f["_date"]).normalize() for f in fill_rows})

    # Always include today if we have a book
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if not trade_dates or trade_dates[-1] < today:
        trade_dates.append(today)

    # Limit to last N calendar days of points
    if days and days > 0 and trade_dates:
        cutoff = trade_dates[-1] - pd.Timedelta(days=int(days))
        trade_dates = [d for d in trade_dates if d >= cutoff]

    # Index prices by date for MTM
    px_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    if not prices.empty:
        for d, g in prices.groupby(prices["trade_date"].dt.normalize()):
            px_by_date[pd.Timestamp(d)] = {
                str(r.stock_code): float(r.price) for r in g.itertuples()
            }

    last_px: dict[str, float] = {}
    fill_i = 0
    points: list[dict[str, Any]] = []

    for d in trade_dates:
        # Apply fills on this date
        while fill_i < len(fill_rows) and fill_rows[fill_i]["_date"].normalize() <= d:
            f = fill_rows[fill_i]
            fill_i += 1
            code = str(f["stock_code"]).upper()
            side = str(f["side"]).lower()
            shares = int(f["shares"])
            price = float(f["price"])
            notional = float(f.get("notional_kes") or price * shares)
            fees = float(f.get("fees_kes") or notional * (cost_bps / 10000.0))
            if side == "buy":
                cash -= notional + fees
                if code in positions:
                    prev_sh = int(positions[code]["shares"])
                    prev_avg = float(positions[code]["avg_cost"])
                    new_sh = prev_sh + shares
                    positions[code]["avg_cost"] = (prev_avg * prev_sh + notional) / new_sh
                    positions[code]["shares"] = new_sh
                else:
                    positions[code] = {"shares": shares, "avg_cost": price}
            elif side == "sell":
                cash += notional - fees
                if code in positions:
                    left = int(positions[code]["shares"]) - shares
                    if left <= 0:
                        del positions[code]
                    else:
                        positions[code]["shares"] = left
            last_px[code] = price

        # Refresh last prices from panel
        day_px = px_by_date.get(d) or {}
        last_px.update(day_px)

        equity = 0.0
        cost_basis = 0.0
        for code, pos in positions.items():
            sh = int(pos["shares"])
            avg = float(pos["avg_cost"])
            px = last_px.get(code)
            if px is None:
                px = avg
            equity += px * sh
            cost_basis += avg * sh
        nav = cash + equity
        points.append(
            {
                "date": str(d.date()),
                "nav_kes": round(nav, 2),
                "cash_kes": round(cash, 2),
                "equity_kes": round(equity, 2),
                "unrealized_pnl_kes": round(equity - cost_basis, 2),
            }
        )

    return {
        "points": points,
        "days": int(days),
        "n_points": len(points),
        "disclaimer": settings.disclaimer,
        "mode": "paper",
    }
