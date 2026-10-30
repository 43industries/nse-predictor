"""Point-in-time fundamentals loader (as-of publish_date / report_date)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.config import settings
from nse_predictor.features.registry import FUNDAMENTALS as FUND_FEATURE_COLS
from nse_predictor.util.datetime import to_ns

FUND_COLS = [
    "stock_code",
    "report_date",
    "publish_date",
    "eps",
    "dividend",
    "book_value",
    "period",
]
MIN_COVERAGE = 0.30


def resolve_fundamentals_path(path: Path | None = None) -> Path | None:
    primary = Path(path or (settings.data_external_dir / "fundamentals.csv"))
    if primary.exists() and primary.stat().st_size > 0:
        return primary
    sample = settings.data_external_dir / "fundamentals_sample.csv"
    if sample.exists() and sample.stat().st_size > 0:
        return sample
    return None


def load_fundamentals(path: Path | None = None) -> pd.DataFrame:
    """Load fundamentals CSV; empty frame if missing."""
    resolved = resolve_fundamentals_path(path)
    if resolved is None:
        return pd.DataFrame(columns=FUND_COLS + ["asof_date"])

    df = pd.read_csv(resolved)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {
        "code": "stock_code",
        "ticker": "stock_code",
        "eps_basic": "eps",
        "div": "dividend",
        "bvps": "book_value",
        "book": "book_value",
        "published": "publish_date",
        "as_of": "report_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "stock_code" not in df.columns:
        return pd.DataFrame(columns=FUND_COLS + ["asof_date"])

    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    for col in ("report_date", "publish_date"):
        df[col] = to_ns(df[col]) if col in df.columns else pd.NaT
    df["asof_date"] = df["publish_date"].fillna(df["report_date"])
    for c in ("eps", "dividend", "book_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if "period" not in df.columns:
        df["period"] = ""
    df = df.dropna(subset=["stock_code", "asof_date"])
    return df.sort_values(["stock_code", "asof_date"]).reset_index(drop=True)


def join_fundamentals_asof(prices_panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Vectorized backward merge_asof on asof_date by stock_code (no leakage)."""
    empty_cols = (
        "eps",
        "dividend",
        "book_value",
        "eps_yoy",
        "div_yield_proxy",
        "book_to_price",
    )
    if fundamentals is None or fundamentals.empty:
        out = prices_panel.copy()
        for c in empty_cols:
            out[c] = np.nan
        return out

    left = prices_panel.copy()
    left["trade_date"] = to_ns(left["trade_date"])
    left["stock_code"] = left["stock_code"].astype(str).str.upper()

    fund = fundamentals.copy()
    fund["stock_code"] = fund["stock_code"].astype(str).str.upper()
    fund["asof_date"] = to_ns(fund["asof_date"])
    fund = fund.dropna(subset=["asof_date"]).sort_values(["stock_code", "asof_date"])
    fund["eps_yoy"] = fund.groupby("stock_code", sort=False)["eps"].pct_change()

    right_cols = ["asof_date", "eps", "dividend", "book_value", "eps_yoy"]
    frames: list[pd.DataFrame] = []
    # Per-ticker asof avoids pandas by= sort edge cases on mixed panels
    fund_by = {c: g for c, g in fund.groupby("stock_code", sort=False)}
    for code, part in left.groupby("stock_code", sort=False):
        p = part.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
        f = fund_by.get(code)
        if f is None or f.empty:
            frames.append(
                p.assign(
                    eps=np.nan,
                    dividend=np.nan,
                    book_value=np.nan,
                    eps_yoy=np.nan,
                )
            )
            continue
        r = (
            f[right_cols]
            .dropna(subset=["asof_date"])
            .sort_values("asof_date", kind="mergesort")
            .rename(columns={"asof_date": "trade_date"})
        )
        frames.append(pd.merge_asof(p, r, on="trade_date", direction="backward"))
    out = pd.concat(frames, ignore_index=True)
    price = out["price"] if "price" in out.columns else pd.Series(np.nan, index=out.index)
    out["div_yield_proxy"] = np.where(
        price.notna() & (price > 0) & out["dividend"].notna(),
        out["dividend"] / price,
        np.nan,
    )
    out["book_to_price"] = np.where(
        price.notna() & (price > 0) & out["book_value"].notna(),
        out["book_value"] / price,
        np.nan,
    )
    return out


def fundamentals_coverage(df: pd.DataFrame, cols: list[str] | None = None) -> float:
    cols = cols or ["eps", "eps_yoy", "div_yield_proxy", "book_to_price"]
    present = [c for c in cols if c in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].notna().any(axis=1).mean())
