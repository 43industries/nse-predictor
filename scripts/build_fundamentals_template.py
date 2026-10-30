#!/usr/bin/env python3
"""Write empty fundamentals template for current investable universe (no invented EPS)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.universe import investable_equities, load_universe

OUT = settings.data_external_dir / "fundamentals_template.csv"
DOCS = settings.data_external_dir / "fundamentals.README.txt"
COLS = [
    "stock_code",
    "report_date",
    "publish_date",
    "eps",
    "dividend",
    "book_value",
    "period",
]


def main() -> int:
    settings.data_external_dir.mkdir(parents=True, exist_ok=True)
    prices_path = settings.data_processed_dir / "prices.parquet"
    investable: list[str] = []
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        investable = investable_equities(load_universe(prices))
    else:
        uni_path = ROOT / "reports" / "universe_investable.json"
        if uni_path.exists():
            data = json.loads(uni_path.read_text(encoding="utf-8"))
            investable = list(data.get("investable") or [])

    rows = [{"stock_code": c, **{k: "" for k in COLS if k != "stock_code"}} for c in investable]
    # One blank stub row per name — user duplicates for FY/H1 history
    pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
    DOCS.write_text(
        "\n".join(
            [
                "fundamentals.csv — point-in-time company fundamentals",
                "",
                "Copy fundamentals_template.csv → fundamentals.csv and fill rows.",
                "Add multiple rows per ticker (FY / H1) with real publish_date.",
                "",
                "Columns:",
                "  stock_code     NSE ticker",
                "  report_date    Period end (YYYY-MM-DD)",
                "  publish_date   When the print was knowable (preferred as-of)",
                "  eps            Earnings per share",
                "  dividend       DPS for the period",
                "  book_value     Book value per share",
                "  period         e.g. FY2023, H1-2024",
                "",
                "Leakage: models join on publish_date (fallback report_date) backward to trade_date.",
                "Features included only if coverage ≥ 30% of train rows.",
                "Do not invent EPS — leave blank until official filings are entered.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"path": str(OUT), "docs": str(DOCS), "n_tickers": len(investable)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
