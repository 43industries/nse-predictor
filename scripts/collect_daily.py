#!/usr/bin/env python3
"""Drop-folder daily NSE price collector (no website scrape).

Place Wanjawa-compatible CSVs in data/raw/nse/incoming/, then run this script.
Rows append to data/raw/nse/daily/NSE_data_all_stocks_YYYY.csv (deduped).
Processed files move to data/raw/nse/incoming/archive/.

Status written to data/processed/collection_status.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.ingest.prices import _normalize_columns, _to_numeric

INCOMING = settings.data_raw_dir / "incoming"
ARCHIVE = INCOMING / "archive"
DAILY_DIR = settings.data_raw_dir / "daily"
STATUS_PATH = settings.data_processed_dir / "collection_status.json"

WANJAWA_OUT_COLS = [
    "Date",
    "Code",
    "Name",
    "12m Low",
    "12m High",
    "Day Low",
    "Day High",
    "Day Price",
    "Previous",
    "Change",
    "Change%",
    "Volume",
    "Adjusted Price",
]


def _parse_incoming(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(raw)
    if "trade_date" not in df.columns or "stock_code" not in df.columns:
        raise ValueError(f"{path.name}: need Date/Code columns (got {list(df.columns)})")

    parsed = pd.to_datetime(df["trade_date"], errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            df.loc[missing, "trade_date"], errors="coerce", dayfirst=False
        )
    df["trade_date"] = parsed
    df = df.dropna(subset=["trade_date", "stock_code"])
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    if "stock_name" not in df.columns:
        df["stock_name"] = df["stock_code"]
    for col in [
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
    ]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
        else:
            df[col] = pd.NA
    if "day_close" not in df.columns or df["day_close"].isna().all():
        raise ValueError(f"{path.name}: missing Day Price / close")
    return df


def _to_wanjawa_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d"),
            "Code": df["stock_code"],
            "Name": df["stock_name"],
            "12m Low": df["low_12m"],
            "12m High": df["high_12m"],
            "Day Low": df["day_low"],
            "Day High": df["day_high"],
            "Day Price": df["day_close"],
            "Previous": df["prev_price"],
            "Change": df["change_value"],
            "Change%": df["change_pct"],
            "Volume": df["volume"],
            "Adjusted Price": df["adj_price"],
        }
    )
    return out[WANJAWA_OUT_COLS]


def _append_year_file(new_rows: pd.DataFrame) -> int:
    """Append rows into yearly daily CSVs; return count of newly added rows."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    wanjawa = _to_wanjawa_csv(new_rows)
    wanjawa["_year"] = pd.to_datetime(wanjawa["Date"]).dt.year
    for year, part in wanjawa.groupby("_year"):
        path = DAILY_DIR / f"NSE_data_all_stocks_{int(year)}.csv"
        part = part.drop(columns=["_year"])
        if path.exists():
            existing = pd.read_csv(path)
            existing["_key"] = (
                existing["Date"].astype(str) + "|" + existing["Code"].astype(str).str.upper()
            )
            part = part.copy()
            part["_key"] = part["Date"].astype(str) + "|" + part["Code"].astype(str).str.upper()
            fresh = part[~part["_key"].isin(set(existing["_key"]))].drop(columns=["_key"])
            if fresh.empty:
                continue
            combined = pd.concat([existing.drop(columns=["_key"], errors="ignore"), fresh], ignore_index=True)
            combined.to_csv(path, index=False)
            added += len(fresh)
        else:
            part.to_csv(path, index=False)
            added += len(part)
    return added


def collect_daily() -> dict:
    INCOMING.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    files = sorted(INCOMING.glob("*.csv"))
    rows_added = 0
    max_date = None
    processed: list[str] = []

    if not files:
        status = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "rows_added": 0,
            "max_date": None,
            "n_tickers": 0,
            "missing_vs_universe": [],
            "coverage_pct": None,
            "files_processed": [],
            "errors": [],
            "note": "No CSV files in data/raw/nse/incoming/",
        }
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    frames = []
    for path in files:
        try:
            df = _parse_incoming(path)
            frames.append(df)
            processed.append(path.name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")

    batch = None
    if frames:
        batch = pd.concat(frames, ignore_index=True)
        batch = batch.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
        rows_added = _append_year_file(batch)
        max_date = str(pd.to_datetime(batch["trade_date"]).max().date())

    for name in processed:
        src = INCOMING / name
        if src.exists():
            dest = ARCHIVE / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{name}"
            shutil.move(str(src), str(dest))

    n_tickers = 0
    missing_vs_universe: list[str] = []
    coverage_pct = None
    if batch is not None and not batch.empty:
        from nse_predictor.universe import investable_equities

        batch_codes = set(batch["stock_code"].astype(str).str.upper())
        n_tickers = len(batch_codes)
        uni_path = ROOT / "reports" / "universe_investable.json"
        if uni_path.exists():
            try:
                uni = json.loads(uni_path.read_text(encoding="utf-8"))
                investable = list(uni.get("investable") or [])
                if not investable and uni.get("universe"):
                    investable = investable_equities(uni["universe"])
                missing_vs_universe = sorted(c for c in investable if c not in batch_codes)
                if investable:
                    coverage_pct = round(
                        100.0 * (len(investable) - len(missing_vs_universe)) / len(investable),
                        1,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"universe coverage: {exc}")

    status = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "rows_added": int(rows_added),
        "max_date": max_date,
        "n_tickers": int(n_tickers),
        "missing_vs_universe": missing_vs_universe,
        "coverage_pct": coverage_pct,
        "files_processed": processed,
        "errors": errors,
        "note": None if processed else "No valid CSVs processed",
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> int:
    status = collect_daily()
    print(json.dumps(status, indent=2))
    return 1 if status.get("errors") and not status.get("files_processed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
