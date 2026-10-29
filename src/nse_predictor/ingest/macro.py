"""CBK / macro series loaders. Synthetic series used when official CSV is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MACRO_COLS = [
    "observe_date",
    "cbk_rate",
    "tbill_91d",
    "usd_kes",
    "inflation_yoy",
    "lending_rate",
]


def load_macro_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "date": "observe_date",
        "cbk": "cbk_rate",
        "central_bank_rate": "cbk_rate",
        "t_bill_91": "tbill_91d",
        "tbill91": "tbill_91d",
        "usdkes": "usd_kes",
        "inflation": "inflation_yoy",
        "lending": "lending_rate",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["observe_date"] = pd.to_datetime(df["observe_date"])
    for c in MACRO_COLS[1:]:
        if c not in df.columns:
            df[c] = np.nan
    return df[MACRO_COLS].sort_values("observe_date")


def make_synthetic_macro(start: str = "2007-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Placeholder macro panel for pipeline development when CBK files are not yet collected.

    Rough historical anchors (not official): CBR moved ~8–13%, USD/KES ~60–160, inflation ~5–10%.
    Replace with official CBK downloads for production research.
    """
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, n)
    cbk = 8 + 4 * np.sin(2 * np.pi * t * 2) + rng.normal(0, 0.05, n)
    tbill = cbk + 1.5 + rng.normal(0, 0.1, n)
    usd_kes = 70 + 90 * t + 5 * np.sin(2 * np.pi * t * 3) + rng.normal(0, 0.2, n)
    inflation = 6 + 2 * np.sin(2 * np.pi * t * 1.5) + rng.normal(0, 0.1, n)
    lending = cbk + 5 + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "observe_date": dates,
            "cbk_rate": np.round(cbk, 3),
            "tbill_91d": np.round(tbill, 3),
            "usd_kes": np.round(usd_kes, 3),
            "inflation_yoy": np.round(inflation, 3),
            "lending_rate": np.round(lending, 3),
        }
    )
