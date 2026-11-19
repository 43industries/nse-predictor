"""Sync processed parquet / external CSVs into DATABASE_URL."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.db.session import init_db


def _read_parquet(name: str) -> pd.DataFrame | None:
    path = settings.data_processed_dir / name
    if not path.exists():
        return None
    return pd.read_parquet(path)


def sync_processed_to_db() -> dict:
    engine = init_db()
    dialect = engine.dialect.name
    summary: dict = {"database_url_dialect": dialect, "tables": {}}

    prices = _read_parquet("prices.parquet")
    if prices is not None and not prices.empty:
        cols = [
            c
            for c in [
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
            if c in prices.columns
        ]
        db = prices[cols].copy()
        db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.date
        with engine.begin() as conn:
            db.to_sql("prices", conn, if_exists="replace", index=False)
        summary["tables"]["prices"] = int(len(db))

    sectors = _read_parquet("sectors.parquet")
    if sectors is not None and not sectors.empty:
        with engine.begin() as conn:
            sectors.to_sql("sectors", conn, if_exists="replace", index=False)
        summary["tables"]["sectors"] = int(len(sectors))

    macro = _read_parquet("macro.parquet")
    if macro is not None and not macro.empty:
        m = macro.copy()
        if "observe_date" in m.columns:
            m["observe_date"] = pd.to_datetime(m["observe_date"]).dt.date
        with engine.begin() as conn:
            m.to_sql("macro", conn, if_exists="replace", index=False)
        summary["tables"]["macro"] = int(len(m))

    fund_path = settings.data_external_dir / "fundamentals.csv"
    if fund_path.exists():
        fund = pd.read_csv(fund_path)
        if not fund.empty:
            fund.columns = [str(c).strip().lower() for c in fund.columns]
            rename = {
                "code": "stock_code",
                "ticker": "stock_code",
                "as_of": "report_date",
                "report_as_of": "report_date",
                "div": "dividend",
                "dps": "dividend",
            }
            fund = fund.rename(columns={k: v for k, v in rename.items() if k in fund.columns})
            keep = [
                c
                for c in ["stock_code", "report_date", "eps", "dividend", "book_value", "period"]
                if c in fund.columns
            ]
            if "stock_code" in keep and "report_date" in keep:
                fdb = fund[keep].copy()
                fdb["report_date"] = pd.to_datetime(fdb["report_date"], errors="coerce").dt.date
                fdb = fdb.dropna(subset=["stock_code", "report_date"])
                with engine.begin() as conn:
                    fdb.to_sql("fundamentals", conn, if_exists="replace", index=False)
                summary["tables"]["fundamentals"] = int(len(fdb))

    forecasts = _read_parquet("latest_forecasts.parquet")
    if forecasts is not None and not forecasts.empty:
        fc = forecasts.copy()
        if "trade_date" in fc.columns and "as_of_date" not in fc.columns:
            fc["as_of_date"] = pd.to_datetime(fc["trade_date"]).dt.date
        elif "as_of_date" in fc.columns:
            fc["as_of_date"] = pd.to_datetime(fc["as_of_date"]).dt.date
        fc["horizon_days"] = 5
        fc["target_type"] = "direction_5d_xs"
        if "model_name" not in fc.columns:
            fc["model_name"] = "direction"
        if "pred_value" not in fc.columns:
            fc["pred_value"] = fc["prob_up"] if "prob_up" in fc.columns else None
        lo = fc["ret5d_xs_lo"] if "ret5d_xs_lo" in fc.columns else fc.get("ret5d_lo")
        hi = fc["ret5d_xs_hi"] if "ret5d_xs_hi" in fc.columns else fc.get("ret5d_hi")
        fc["lower"] = lo
        fc["upper"] = hi
        keep = [
            c
            for c in [
                "as_of_date",
                "stock_code",
                "horizon_days",
                "target_type",
                "model_name",
                "pred_label",
                "prob_up",
                "pred_value",
                "lower",
                "upper",
            ]
            if c in fc.columns
        ]
        fcdb = fc[keep].copy()
        with engine.begin() as conn:
            fcdb.to_sql("forecasts", conn, if_exists="replace", index=False)
        summary["tables"]["forecasts"] = int(len(fcdb))

    out = Path(settings.data_processed_dir) / "db_sync_status.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["status_path"] = str(out)
    return summary
