#!/usr/bin/env python3
"""CLI: ingest raw NSE CSVs into parquet + database."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.ingest.pipeline import run_ingest


def main() -> int:
    summary = run_ingest()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
