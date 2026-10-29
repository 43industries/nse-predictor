#!/usr/bin/env python3
"""Week-1 EDA summary: liquidity ranking and coverage -> reports/eda_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    prices = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet")
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    code = prices["stock_code"].astype(str)
    ordinary = prices.loc[
        ~code.str.contains(r"-R$|^\^|NASI|N20I|NSE20", case=False, na=False, regex=True)
    ]
    liq = (
        ordinary.groupby("stock_code")
        .agg(
            avg_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            zero_volume_days=("zero_volume", "sum"),
            n_days=("trade_date", "count"),
            first=("trade_date", "min"),
            last=("trade_date", "max"),
            name=("stock_name", "last"),
        )
        .sort_values("avg_volume", ascending=False)
    )
    liq["first"] = liq["first"].astype(str)
    liq["last"] = liq["last"].astype(str)
    top = liq.head(20).reset_index()
    out = {
        "n_rows": int(len(prices)),
        "n_tickers": int(prices["stock_code"].nunique()),
        "date_min": str(prices["trade_date"].min().date()),
        "date_max": str(prices["trade_date"].max().date()),
        "top_liquid": top.to_dict(orient="records"),
        "proposed_universe": top.head(15)["stock_code"].tolist(),
    }
    path = ROOT / "reports" / "eda_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Also update docs snippet
    md = ROOT / "docs" / "liquid_universe.md"
    lines = [
        "# Proposed liquid universe (v1)",
        "",
        "Ranked by average daily volume after ingest of Wanjawa 2007–2025 panel:",
        "",
        "| Rank | Code | Avg volume | Zero-vol days | Span |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for i, r in enumerate(top.head(15).itertuples(), 1):
        lines.append(
            f"| {i} | {r.stock_code} | {r.avg_volume:,.0f} | {int(r.zero_volume_days)} | {r.first[:10]} → {r.last[:10]} |"
        )
    lines.extend(["", "Indices retained for context features when present: NASI, NSE20.", ""])
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["n_rows", "n_tickers", "date_min", "date_max", "proposed_universe"]}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
