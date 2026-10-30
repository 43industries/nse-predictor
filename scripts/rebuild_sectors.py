#!/usr/bin/env python3
"""Rebuild data/processed/sectors.parquet from non-empty sector CSVs; report coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.ingest.prices import load_sectors
from nse_predictor.universe import investable_equities, load_universe


def main() -> int:
    sectors = load_sectors(settings.data_raw_dir)
    out = settings.data_processed_dir / "sectors.parquet"
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    sectors.to_parquet(out, index=False)

    prices_path = settings.data_processed_dir / "prices.parquet"
    investable: list[str] = []
    if prices_path.exists():
        import pandas as pd

        prices = pd.read_parquet(prices_path)
        uni = load_universe(prices)
        investable = investable_equities(uni)

    mapped = set(sectors["stock_code"].astype(str).str.upper()) if not sectors.empty else set()
    orphans = [c for c in investable if c not in mapped]
    cov = 1.0 - len(orphans) / max(len(investable), 1) if investable else 0.0

    report = {
        "n_mapped_total": int(len(sectors)),
        "investable": investable,
        "coverage": float(cov),
        "orphans": orphans,
        "path": str(out),
        "ok": cov >= 0.9 if investable else len(sectors) > 0,
    }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "sectors_rebuild.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
