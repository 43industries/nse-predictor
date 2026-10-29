#!/usr/bin/env python3
"""Build data/external/macro_cbk.csv from public approximate CBK anchors.

Replace with official CBK / KNBS downloads for production research.
Values are as-of published style (monthly), suitable for merge_asof joins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "macro_cbk.csv"

# Approximate public anchors (CBR %, USD/KES, inflation YoY). Interpolated monthly.
CBR_POINTS = [
    ("2007-01-01", 10.0),
    ("2009-01-01", 8.25),
    ("2011-12-01", 18.0),
    ("2013-06-01", 8.5),
    ("2015-07-01", 11.5),
    ("2018-01-01", 9.5),
    ("2020-04-01", 7.0),
    ("2022-06-01", 7.5),
    ("2023-06-01", 10.5),
    ("2024-08-01", 12.75),
    ("2025-06-01", 9.75),
    ("2025-12-01", 9.5),
]
USD_POINTS = [
    ("2007-01-01", 69.0),
    ("2011-01-01", 82.0),
    ("2015-01-01", 95.0),
    ("2020-01-01", 105.0),
    ("2023-01-01", 123.0),
    ("2023-10-01", 150.0),
    ("2024-06-01", 129.0),
    ("2025-12-01", 129.5),
]
INFL_POINTS = [
    ("2007-01-01", 6.0),
    ("2008-06-01", 15.0),
    ("2011-06-01", 14.0),
    ("2014-01-01", 6.5),
    ("2017-01-01", 8.0),
    ("2020-01-01", 5.5),
    ("2022-01-01", 5.5),
    ("2023-06-01", 7.9),
    ("2025-01-01", 3.5),
    ("2025-12-01", 3.8),
]


def _series(points: list[tuple[str, float]], index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series({pd.Timestamp(d): v for d, v in points}).sort_index()
    s = s.reindex(s.index.union(index)).sort_index().interpolate(method="time")
    return s.reindex(index).ffill().bfill()


def main() -> int:
    idx = pd.date_range("2007-01-01", "2025-12-01", freq="MS")
    cbk = _series(CBR_POINTS, idx)
    usd = _series(USD_POINTS, idx)
    infl = _series(INFL_POINTS, idx)
    tbill = cbk + 1.2
    lending = cbk + 5.5
    df = pd.DataFrame(
        {
            "observe_date": idx.strftime("%Y-%m-%d"),
            "cbk_rate": np.round(cbk.values, 3),
            "tbill_91d": np.round(tbill.values, 3),
            "usd_kes": np.round(usd.values, 3),
            "inflation_yoy": np.round(infl.values, 3),
            "lending_rate": np.round(lending.values, 3),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(df)}")
    print("NOTE: Approximate public anchors — replace with official CBK/KNBS series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
