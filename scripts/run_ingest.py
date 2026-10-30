#!/usr/bin/env python3
"""CLI: ingest raw NSE CSVs into parquet + database."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.ingest.pipeline import run_ingest
from nse_predictor.ingest.qa import run_price_qa, write_qa_reports


def main() -> int:
    summary = run_ingest()
    print(json.dumps(summary, indent=2))

    # Non-fatal QA summary after clean
    try:
        qa = run_price_qa(macro_source=summary.get("macro_source", "unknown"))
        write_qa_reports(qa)
        print("\n=== data QA (non-fatal) ===")
        print(
            json.dumps(
                {
                    "duplicates": qa["duplicates"]["duplicate_rows"],
                    "gaps": qa["calendar_gaps"]["n_gap_events"],
                    "jumps": qa["jumps"]["n_jumps"],
                    "zero_vol_flagged": len(qa["zero_volume"]["flagged"]),
                    "macro_source": qa["macro"].get("source"),
                    "sector_coverage": qa["sectors"].get("coverage"),
                    "ok": qa.get("ok"),
                },
                indent=2,
            )
        )
        if str(summary.get("macro_source", "")).startswith("synthetic"):
            print(
                "WARNING: macro_source is synthetic — replace data/external/macro_cbk.csv "
                "with official CBK/KNBS and add macro_cbk.csv.official marker.",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"QA warning (non-fatal): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
