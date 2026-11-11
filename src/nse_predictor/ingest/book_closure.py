"""Dividend book-closure calendar (leakage-safe via announce_date)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.config import settings

COLS = ["stock_code", "announce_date", "book_closure_date", "ex_date"]


def load_book_closures(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or (settings.data_external_dir / "book_closures.csv"))
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {
        "code": "stock_code",
        "announced": "announce_date",
        "book_closure": "book_closure_date",
        "closure_date": "book_closure_date",
        "ex": "ex_date",
        "ex_div": "ex_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "stock_code" not in df.columns or "book_closure_date" not in df.columns:
        return pd.DataFrame(columns=COLS)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    for c in ("announce_date", "book_closure_date", "ex_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce") if c in df.columns else pd.NaT
    df["announce_date"] = df["announce_date"].fillna(df["book_closure_date"])
    return df.dropna(subset=["stock_code", "book_closure_date"]).reset_index(drop=True)


def join_book_closure_features(panel: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["near_book_closure"] = 0.0
    out["days_since_book_closure"] = np.nan
    ev = events if events is not None else load_book_closures()
    if ev is None or ev.empty:
        return out

    frames: list[pd.DataFrame] = []
    by = {c: g for c, g in ev.groupby("stock_code", sort=False)}
    for code, part in out.groupby("stock_code", sort=False):
        p = part.sort_values("trade_date").copy()
        e = by.get(str(code).upper())
        if e is None or e.empty:
            frames.append(p)
            continue
        near = np.zeros(len(p), dtype=float)
        days_since = np.full(len(p), np.nan)
        dates = p["trade_date"].values
        for _, row in e.iterrows():
            ann = row["announce_date"]
            clo = row["book_closure_date"]
            if pd.isna(clo):
                continue
            if pd.isna(ann):
                ann = clo
            # Known only on/after announce
            mask_win = (dates >= np.datetime64(ann)) & (dates <= np.datetime64(clo))
            near = np.where(mask_win, 1.0, near)
            after = dates >= np.datetime64(clo)
            delta = (pd.to_datetime(dates) - pd.Timestamp(clo)).days.astype(float)
            days_since = np.where(after, delta, days_since)
        p["near_book_closure"] = near
        p["days_since_book_closure"] = days_since
        frames.append(p)
    return pd.concat(frames, ignore_index=True)
