#!/usr/bin/env python3
"""Daily pipeline: collect drop-folder CSVs → ingest → score (no full retrain).

Schedule on Windows Task Scheduler ~17:30 EAT on weekdays:
  .\\.venv\\Scripts\\python.exe scripts\\run_daily.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script: str) -> dict | None:
    print(f"\n=== {script} ===")
    proc = subprocess.run(
        [PY, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    # Best-effort parse last JSON object from stdout
    try:
        return json.loads(proc.stdout.strip().split("\n\n")[-1])
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    collect = run("collect_daily.py")
    run("run_ingest.py")
    score = run("score_latest.py")
    summary = {
        "collect": collect,
        "score": score,
        "ok": True,
    }
    print("\n=== daily summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
