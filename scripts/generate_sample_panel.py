#!/usr/bin/env python3
"""Generate a realistic synthetic NSE-like daily panel (2007–2025) for offline development.

Used when Mendeley downloads are unavailable. Schema matches Wanjawa CSVs so the
ingest/feature/train pipeline runs end-to-end. Replace with real downloads for research.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TICKERS = [
    ("SCOM", "Safaricom PLC", "Telecommunications"),
    ("EQTY", "Equity Group Holdings", "Banking"),
    ("KCB", "KCB Group", "Banking"),
    ("COOP", "Co-operative Bank", "Banking"),
    ("ABSA", "Absa Bank Kenya", "Banking"),
    ("SCBK", "Stanbic Holdings", "Banking"),
    ("EABL", "East African Breweries", "Manufacturing"),
    ("BAT", "British American Tobacco Kenya", "Manufacturing"),
    ("BAMB", "Bamburi Cement", "Manufacturing"),
    ("KQ", "Kenya Airways", "Aviation"),
    ("NMG", "Nation Media Group", "Media"),
    ("JHL", "Jubilee Holdings", "Insurance"),
    ("SBIC", "Stanbic IBTC", "Banking"),
    ("NCBA", "NCBA Group", "Banking"),
    ("NASI", "NSE All Share Index", "Index"),
    ("NSE20", "NSE 20 Share Index", "Index"),
]


def simulate_ticker(code: str, name: str, start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    # Geometric Brownian-ish with occasional thin-trade zeros
    mu = 0.00015
    sigma = 0.018 if code not in {"NASI", "NSE20"} else 0.01
    shocks = rng.normal(mu, sigma, n)
    # COVID-ish stress
    covid = (dates >= "2020-03-01") & (dates <= "2020-06-30")
    shocks = np.where(covid, shocks - 0.004, shocks)
    price0 = {"NASI": 140.0, "NSE20": 2000.0}.get(code, float(rng.uniform(5, 80)))
    price = price0 * np.exp(np.cumsum(shocks))
    # Sparse zero-volume days for non-index
    vol = rng.integers(10_000, 2_000_000, n).astype(float)
    if code not in {"NASI", "NSE20", "SCOM", "EQTY", "KCB"}:
        thin = rng.random(n) < 0.08
        vol[thin] = 0
    prev = np.roll(price, 1)
    prev[0] = price0
    high = price * (1 + rng.uniform(0, 0.015, n))
    low = price * (1 - rng.uniform(0, 0.015, n))
    # trailing 12m
    s = pd.Series(price)
    low12 = s.rolling(252, min_periods=1).min().values
    high12 = s.rolling(252, min_periods=1).max().values
    return pd.DataFrame(
        {
            "Date": dates,
            "Code": code,
            "Name": name,
            "12-month Low": low12,
            "12-month High": high12,
            "Day's Low": low,
            "Day's High": high,
            "Day's Final Price": price,
            "Previous traded price": prev,
            "Change in price by value": price - prev,
            "Change in price by %": (price / prev - 1) * 100,
            "Volume traded": vol,
            "Adjusted price": price,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2007-01-02")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "raw" / "nse" / "synthetic"),
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, (code, name, sector) in enumerate(TICKERS):
        frames.append(simulate_ticker(code, name, args.start, args.end, seed=100 + i))
    all_df = pd.concat(frames, ignore_index=True)

    # Write yearly files to mimic Wanjawa layout
    all_df["year"] = pd.to_datetime(all_df["Date"]).dt.year
    for year, part in all_df.groupby("year"):
        path = out / f"NSE_data_all_stocks_{year}.csv"
        part.drop(columns=["year"]).to_csv(path, index=False)
        print(f"wrote {path} rows={len(part)}")

    sectors = pd.DataFrame(
        [{"Market sector": s, "Stock Code": c, "Stock Name": n} for c, n, s in TICKERS]
    )
    sectors.to_csv(out / "NSE_data_stock_market_sectors_2025.csv", index=False)
    print(f"wrote sectors -> {out}")
    print(
        "NOTE: Synthetic data for pipeline testing only. "
        "Run scripts/download_wanjawa.py for real Mendeley CSVs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
