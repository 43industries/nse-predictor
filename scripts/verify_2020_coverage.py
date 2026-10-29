#!/usr/bin/env python3
"""Inspect 2020 CSV date coverage and update data/manifest.json coverage_2020 section."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nse"
MANIFEST = ROOT / "data" / "manifest.json"


def find_2020_csv() -> Path | None:
    candidates = list(RAW.rglob("*2020*.csv"))
    candidates = [p for p in candidates if "sector" not in p.name.lower()]
    return candidates[0] if candidates else None


def main() -> int:
    path = find_2020_csv()
    if path is None:
        print("No 2020 price CSV found under data/raw/nse")
        print("Run download_wanjawa.py or generate_sample_panel.py first.")
        return 1

    df = pd.read_csv(path, nrows=5)
    date_col = df.columns[0]
    for c in df.columns:
        if str(c).strip().lower() in {"date", "trade_date"}:
            date_col = c
            break
    dates = pd.to_datetime(pd.read_csv(path, usecols=[date_col])[date_col], errors="coerce")
    min_d, max_d = dates.min(), dates.max()
    h2_missing = pd.isna(max_d) or max_d < pd.Timestamp("2020-07-01")
    report = {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "min_date": None if pd.isna(min_d) else str(min_d.date()),
        "max_date": None if pd.isna(max_d) else str(max_d.date()),
        "rows": int(dates.notna().sum()),
        "h2_missing": bool(h2_missing),
        "note": (
            "Jul–Dec 2020 missing — fill from MyStocks/NSE Data Services before continuous panel use."
            if h2_missing
            else "2020 appears to extend into H2; still spot-check trading-day continuity."
        ),
    }
    print(json.dumps(report, indent=2))

    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest["coverage_2020"] = report
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"updated {MANIFEST}")
    return 0 if not h2_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
