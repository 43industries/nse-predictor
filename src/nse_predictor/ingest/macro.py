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


def official_marker_path(macro_file: Path) -> Path:
    """Sidecar marker: macro_cbk.csv → macro_cbk.csv.official (not .with_suffix)."""
    return Path(str(macro_file) + ".official")


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
    df["observe_date"] = pd.to_datetime(df["observe_date"], errors="coerce")
    for c in MACRO_COLS[1:]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return (
        df[MACRO_COLS]
        .dropna(subset=["observe_date"])
        .sort_values("observe_date")
        .reset_index(drop=True)
    )


def validate_official_macro(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Required columns, monotonic dates, not future-only stubs."""
    errors: list[str] = []
    if df is None or df.empty:
        return False, ["empty"]
    missing = [c for c in MACRO_COLS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return False, errors
    m = df.copy()
    m["observe_date"] = pd.to_datetime(m["observe_date"], errors="coerce")
    m = m.dropna(subset=["observe_date"]).sort_values("observe_date")
    if len(m) < 12:
        errors.append(f"too few rows ({len(m)})")
    if not m["observe_date"].is_monotonic_increasing:
        errors.append("observe_date not monotonic")
    future = int((m["observe_date"] > pd.Timestamp.today() + pd.Timedelta(days=30)).sum())
    if future:
        errors.append(f"{future} future stub rows")
    nums = MACRO_COLS[1:]
    if m[nums].notna().to_numpy().sum() == 0:
        errors.append("all macro values null")
    return len(errors) == 0, errors


def detect_macro_source(path: Path, df: pd.DataFrame) -> str:
    """Return official only when marker + valid panel; else synthetic (no silent mix)."""
    name = path.name.lower()
    if "synthetic" in name or "template" in name:
        return "synthetic"
    if not official_marker_path(path).exists():
        return "synthetic"
    ok, _ = validate_official_macro(df)
    return "official" if ok else "synthetic"


def make_synthetic_macro(start: str = "2007-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Placeholder macro panel when CBK files are not yet collected."""
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


def resolve_macro(
    macro_file: Path,
    price_min: str,
    price_max: str,
) -> tuple[pd.DataFrame, str]:
    """Load official if valid+marked; else keep file labeled synthetic, or generate placeholder."""
    if not (macro_file.exists() and macro_file.stat().st_size > 0):
        return make_synthetic_macro(price_min, price_max), "synthetic"

    loaded = load_macro_csv(macro_file)
    source = detect_macro_source(macro_file, loaded)
    ok, _errs = validate_official_macro(loaded)

    if source == "official" and ok:
        return loaded, "official"
    if source == "official" and not ok:
        return make_synthetic_macro(price_min, price_max), "synthetic_after_invalid_official"
    if not loaded.empty and ok:
        # Unmarked but structurally valid (approx anchors) — keep values, flag synthetic
        return loaded, "synthetic"
    return make_synthetic_macro(price_min, price_max), "synthetic"
