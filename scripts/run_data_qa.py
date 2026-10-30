#!/usr/bin/env python3
"""Run price / macro / sector QA → reports/data_qa.json + data_qa.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.ingest.qa import run_price_qa, write_qa_reports


def main() -> int:
    processed = settings.data_processed_dir
    prices = processed / "prices.parquet"
    if not prices.exists():
        print("Missing prices.parquet — run scripts/run_ingest.py first", file=sys.stderr)
        return 1

    meta_path = processed / "ingest_meta.json"
    macro_source = "unknown"
    if meta_path.exists():
        try:
            macro_source = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "macro_source", "unknown"
            )
        except Exception:  # noqa: BLE001
            pass

    report = run_price_qa(macro_source=macro_source)
    json_path, md_path = write_qa_reports(report)
    summary = {
        "ok": report.get("ok"),
        "duplicates": report["duplicates"]["duplicate_rows"],
        "gaps": report["calendar_gaps"]["n_gap_events"],
        "jumps": report["jumps"]["n_jumps"],
        "macro_source": report["macro"].get("source"),
        "sector_coverage": report["sectors"].get("coverage"),
        "json": str(json_path),
        "md": str(md_path),
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
