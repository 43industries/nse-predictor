"""End-to-end ingest: raw CSVs -> cleaned parquet + database tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.db.session import get_engine, init_db
from nse_predictor.ingest.macro import load_macro_csv, make_synthetic_macro
from nse_predictor.ingest.prices import clean_prices, load_raw_prices, load_sectors


def run_ingest(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    macro_path: Path | None = None,
    use_db: bool = True,
) -> dict:
    raw_dir = Path(raw_dir or settings.data_raw_dir)
    processed_dir = Path(processed_dir or settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_prices(raw_dir)
    prices = clean_prices(raw)
    sectors = load_sectors(raw_dir)

    macro_file = Path(macro_path or (settings.data_external_dir / "macro_cbk.csv"))
    if macro_file.exists() and macro_file.stat().st_size > 0:
        macro = load_macro_csv(macro_file)
        if macro.empty:
            start = str(prices["trade_date"].min().date())
            end = str(prices["trade_date"].max().date())
            macro = make_synthetic_macro(start, end)
            macro_source = "synthetic_placeholder"
        else:
            macro_source = "file"
    else:
        start = str(prices["trade_date"].min().date())
        end = str(prices["trade_date"].max().date())
        macro = make_synthetic_macro(start, end)
        macro_source = "synthetic_placeholder"
        macro.to_csv(settings.data_external_dir / "macro_cbk_synthetic.csv", index=False)

    prices_path = processed_dir / "prices.parquet"
    sectors_path = processed_dir / "sectors.parquet"
    macro_path_out = processed_dir / "macro.parquet"
    prices.to_parquet(prices_path, index=False)
    sectors.to_parquet(sectors_path, index=False)
    macro.to_parquet(macro_path_out, index=False)

    if use_db:
        engine = init_db()
        prices_db = prices.rename(columns={})[
            [
                "trade_date",
                "stock_code",
                "stock_name",
                "low_12m",
                "high_12m",
                "day_low",
                "day_high",
                "day_close",
                "prev_price",
                "change_value",
                "change_pct",
                "volume",
                "adj_price",
                "source_file",
            ]
        ].copy()
        prices_db["trade_date"] = pd.to_datetime(prices_db["trade_date"]).dt.date
        with engine.begin() as conn:
            prices_db.to_sql("prices", conn, if_exists="replace", index=False)
            sectors.to_sql("sectors", conn, if_exists="replace", index=False)
            macro_db = macro.copy()
            macro_db["observe_date"] = pd.to_datetime(macro_db["observe_date"]).dt.date
            macro_db.to_sql("macro", conn, if_exists="replace", index=False)

    return {
        "n_price_rows": len(prices),
        "n_tickers": prices["stock_code"].nunique(),
        "date_min": str(prices["trade_date"].min().date()),
        "date_max": str(prices["trade_date"].max().date()),
        "macro_source": macro_source,
        "prices_path": str(prices_path),
        "sectors_path": str(sectors_path),
        "macro_path": str(macro_path_out),
    }
