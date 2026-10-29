#!/usr/bin/env python3
"""Refresh latest_forecasts.parquet using saved models (no retrain)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.serving.score import score_latest_forecasts


def main() -> int:
    summary = score_latest_forecasts()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
