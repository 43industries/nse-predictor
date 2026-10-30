#!/usr/bin/env python3
"""Run file-based monitoring checks (pipeline, score hygiene, outcomes, gates).

Always exits 0 so Task Scheduler keeps succeeding; grep MONITOR_STATUS= on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.monitoring.runner import print_monitor_summary, run_all_checks
from nse_predictor.monitoring.status import history_path, status_path


def main() -> int:
    payload = run_all_checks()
    print_monitor_summary(payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"wrote {status_path()}")
    print(f"wrote {history_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
