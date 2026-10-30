"""End-to-end ingest: raw CSVs -> cleaned parquet + database tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.db.session import init_db
from nse_predictor.ingest.macro import resolve_macro
from nse_predictor.ingest.prices import clean_prices, load_raw_prices, load_sectors


def run_ingest(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    macro_path: Path | None = None,
    use_db: bool = True,
    run_qa: bool = False,
) -> dict:
    raw_dir = Path(raw_dir or settings.data_raw_dir)
    processed_dir = Path(processed_dir or settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    settings.data_external_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_prices(raw_dir)
    prices = clean_prices(raw)
    sectors = load_sectors(raw_dir)

    macro_file = Path(macro_path or (settings.data_external_dir / "macro_cbk.csv"))
    start = str(prices["trade_date"].min().date())
    end = str(prices["trade_date"].max().date())
    macro, macro_source = resolve_macro(macro_file, start, end)
    if macro_source.startswith("synthetic") and not macro_file.exists():
        macro.to_csv(settings.data_external_dir / "macro_cbk_synthetic.csv", index=False)

    prices_path = processed_dir / "prices.parquet"
    sectors_path = processed_dir / "sectors.parquet"
    macro_path_out = processed_dir / "macro.parquet"
    prices.to_parquet(prices_path, index=False)
    sectors.to_parquet(sectors_path, index=False)
    macro.to_parquet(macro_path_out, index=False)

    meta = {
        "macro_source": macro_source,
        "n_price_rows": int(len(prices)),
        "n_tickers": int(prices["stock_code"].nunique()),
        "date_min": start,
        "date_max": end,
        "n_sectors": int(len(sectors)),
    }
    (processed_dir / "ingest_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if use_db:
        engine = init_db()
        prices_db = prices[
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

    summary = {
        **meta,
        "prices_path": str(prices_path),
        "sectors_path": str(sectors_path),
        "macro_path": str(macro_path_out),
    }

    if run_qa:
        from nse_predictor.ingest.qa import run_price_qa, write_qa_reports

        qa = run_price_qa(prices=prices, macro=macro, sectors=sectors, macro_source=macro_source)
        write_qa_reports(qa)
        summary["qa"] = {
            "duplicates": qa["duplicates"]["duplicate_rows"],
            "gaps": qa["calendar_gaps"]["n_gap_events"],
            "jumps": qa["jumps"]["n_jumps"],
            "sector_coverage": qa["sectors"].get("coverage"),
            "macro_source": macro_source,
        }

    return summary
