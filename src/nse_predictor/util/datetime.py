"""Datetime helpers shared across ingest / QA."""

from __future__ import annotations

import pandas as pd


def to_ns(s: pd.Series) -> pd.Series:
    """Parse to timezone-naive datetime64[ns] for merge_asof compatibility."""
    return pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
