#!/usr/bin/env python3
"""CLI wrapper: sync processed parquet panels into DATABASE_URL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.db.sync import sync_processed_to_db


if __name__ == "__main__":
    print(json.dumps(sync_processed_to_db(), indent=2, default=str))
