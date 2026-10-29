#!/usr/bin/env python3
"""Run download (if needed) → ingest → EDA → train in one shot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.check_call([PY, str(ROOT / "scripts" / script)], cwd=ROOT)


def main() -> int:
    raw_csvs = list((ROOT / "data" / "raw" / "nse").rglob("NSE_data_all_stocks*.csv"))
    if not raw_csvs:
        run("download_wanjawa.py")
    else:
        print(f"Found {len(raw_csvs)} raw price CSVs — skipping download")
    run("verify_2020_coverage.py")
    run("run_ingest.py")
    run("run_eda_summary.py")
    run("train_evaluate.py")
    print("\nDone. Start API: uvicorn api.main:app --reload --app-dir .")
    print("Start dashboard: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
