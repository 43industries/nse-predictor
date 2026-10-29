"""Load and clean Wanjawa-style NSE daily price CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "date": "trade_date",
    "code": "stock_code",
    "stock code": "stock_code",
    "name": "stock_name",
    "stock name": "stock_name",
    # 12-month range (Wanjawa variants)
    "12-month low": "low_12m",
    "12-month low price": "low_12m",
    "12month lowprice": "low_12m",
    "12m low": "low_12m",
    "12-month high": "high_12m",
    "12-month high price": "high_12m",
    "12month highprice": "high_12m",
    "12m high": "high_12m",
    # Session OHLC
    "day's low": "day_low",
    "day's low price": "day_low",
    "day lowprice": "day_low",
    "day low": "day_low",
    "day's high": "day_high",
    "day's high price": "day_high",
    "day highprice": "day_high",
    "day high": "day_high",
    "day's final price": "day_close",
    "day finalprice": "day_close",
    "day final price": "day_close",
    "day price": "day_close",
    "final price": "day_close",
    "close": "day_close",
    # Previous / change / volume / adjust
    "previous traded price": "prev_price",
    "previous": "prev_price",
    "change in price by value": "change_value",
    "change in price value": "change_value",
    "change": "change_value",
    "change in price by %": "change_pct",
    "change in price %": "change_pct",
    "change in price%": "change_pct",
    "change%": "change_pct",
    "% change": "change_pct",
    "volume traded": "volume",
    "volume": "volume",
    "adjusted price": "adj_price",
    "adjusted price, if any": "adj_price",
    "adjusted": "adj_price",
    "adjust": "adj_price",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for c in df.columns:
        key = str(c).strip().lower().replace("_", " ")
        key = " ".join(key.split())
        renamed[c] = COLUMN_MAP.get(key, key.replace(" ", "_"))
    out = df.rename(columns=renamed)
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "iuf":
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw_prices(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).rglob("*.csv"))
    skip_parts = {"incoming", "archive"}
    price_files = [
        f
        for f in files
        if "sector" not in f.name.lower()
        and not any(part.lower() in skip_parts for part in f.parts)
    ]
    if not price_files:
        raise FileNotFoundError(
            f"No price CSVs under {raw_dir}. Run: python scripts/download_wanjawa.py"
        )
    frames = []
    for path in price_files:
        df = pd.read_csv(path, low_memory=False)
        df = _normalize_columns(df)
        df["source_file"] = str(path).replace("\\", "/")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "trade_date" not in df.columns:
        raise ValueError(f"Missing trade_date; columns={list(df.columns)}")
    # Mixed formats: 1/2/2007, 02-Jan-20, 2023-01-03 — try dayfirst then ISO
    parsed = pd.to_datetime(df["trade_date"], errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(df.loc[missing, "trade_date"], errors="coerce", dayfirst=False)
    df["trade_date"] = parsed
    df = df.dropna(subset=["trade_date", "stock_code"])
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    if "stock_name" in df.columns:
        df["stock_name"] = df["stock_name"].astype(str).str.strip()

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
            df[col] = np.nan

    # Day Price is the primary close; Adjust is often blank ("-")
    df["price"] = df["day_close"].fillna(df["adj_price"])
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Corporate-action heuristic: flag huge overnight jumps without volume context
    df = df.sort_values(["stock_code", "trade_date"])
    df["ret_1d"] = df.groupby("stock_code")["price"].pct_change()
    df["volume"] = df["volume"].fillna(0)
    df["zero_volume"] = df["volume"] <= 0
    df["days_since_trade"] = (
        df.groupby("stock_code")["zero_volume"]
        .transform(lambda s: s.groupby((~s).cumsum()).cumsum())
        .astype(int)
    )

    keep = [
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
        "price",
        "ret_1d",
        "zero_volume",
        "days_since_trade",
        "source_file",
    ]
    df = df[keep].drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
    return df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def load_sectors(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).rglob("*sector*.csv"))
    if not files:
        return pd.DataFrame(columns=["stock_code", "stock_name", "market_sector", "as_of_year"])
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        # Heuristic year from filename
        year = None
        for token in path.stem.replace("-", "_").split("_"):
            if token.isdigit() and len(token) == 4:
                year = int(token)
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if "sector" in cl:
                rename[c] = "market_sector"
            elif "code" in cl:
                rename[c] = "stock_code"
            elif "name" in cl:
                rename[c] = "stock_name"
        df = df.rename(columns=rename)
        df["as_of_year"] = year
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.upper()
    out = out.sort_values("as_of_year").drop_duplicates("stock_code", keep="last")
    return out[["stock_code", "stock_name", "market_sector", "as_of_year"]]
