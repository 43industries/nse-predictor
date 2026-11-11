#!/usr/bin/env python3
"""Write a Wanjawa-format full-board CSV using last known closes (carry-forward).

Not real market prints — research / pipeline testing only.

Usage:
  python scripts/make_demo_daily_csv.py           # next business day after max price date
  python scripts/make_demo_daily_csv.py --to-today  # last business day ≤ today (clears stale monitor)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--to-today",
        action="store_true",
        help="Date the board as the latest business day on/before today",
    )
    args = ap.parse_args()

    prices_path = settings.data_processed_dir / "prices.parquet"
    if not prices_path.exists():
        print("Run ingest first (prices.parquet missing)", file=sys.stderr)
        return 1
    px = pd.read_parquet(prices_path)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    max_d = px["trade_date"].max()
    if args.to_today:
        today = pd.Timestamp.today().normalize()
        next_d = today if today.day_of_week < 5 else today - pd.offsets.BDay(1)
        if next_d <= max_d:
            next_d = max_d + pd.offsets.BDay(1)
    else:
        next_d = max_d + pd.offsets.BDay(1)

    latest = px.sort_values("trade_date").groupby("stock_code", as_index=False).tail(1)
    rows = []
    for _, r in latest.iterrows():
        price = float(r["price"])
        rows.append(
            {
                "Date": next_d.strftime("%Y-%m-%d"),
                "Code": r["stock_code"],
                "Name": r.get("stock_name") or r["stock_code"],
                "12m Low": r.get("low_12m") if pd.notna(r.get("low_12m")) else price * 0.8,
                "12m High": r.get("high_12m") if pd.notna(r.get("high_12m")) else price * 1.2,
                "Day Low": price,
                "Day High": price,
                "Day Price": price,
                "Previous": price,
                "Change": 0.0,
                "Change%": 0.0,
                "Volume": int(r["volume"]) if pd.notna(r.get("volume")) else 0,
                "Adjusted Price": price,
            }
        )
    out_dir = settings.data_raw_dir / "incoming"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"demo_carryforward_{next_d.strftime('%Y%m%d')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        {
            "ok": True,
            "path": str(out),
            "as_of": str(next_d.date()),
            "n_rows": len(rows),
            "note": "carry_forward_demo_not_real_prints",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
