#!/usr/bin/env python3
"""Daily pipeline: collect drop-folder CSVs → ingest → QA → score → monitor (no full retrain).

Schedule on Windows Task Scheduler ~17:30 EAT on weekdays:
  .\\.venv\\Scripts\\python.exe scripts\\run_daily.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable

from nse_predictor.config import settings
from nse_predictor.monitoring.status import save_daily_run


def _parse_json_stdout(stdout: str) -> dict | None:
    text = (stdout or "").strip()
    if not text:
        return None
    for chunk in reversed(text.split("\n\n")):
        chunk = chunk.strip()
        if chunk.startswith("{") or chunk.startswith("["):
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run(script: str, soft_fail: bool = False) -> dict | None:
    print(f"\n=== {script} ===")
    proc = subprocess.run(
        [PY, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        if soft_fail:
            return {"ok": False, "error": (proc.stderr or proc.stdout or "failed")[-500:]}
        raise SystemExit(proc.returncode)
    parsed = _parse_json_stdout(proc.stdout) or {}
    parsed["ok"] = True
    return parsed


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    collect = run("collect_daily.py", soft_fail=True) or {}
    empty_incoming = (not collect.get("files_processed")) and (
        collect.get("rows_added", 0) == 0
    )
    if empty_incoming:
        print(
            "WARN: no new incoming CSVs — continuing with existing daily/history data",
            file=sys.stderr,
        )

    cov = collect.get("coverage_pct")
    if cov is not None and cov < 80:
        print(
            f"HARD WARN: universe coverage {cov}% < 80% "
            f"(missing={collect.get('missing_vs_universe')})",
            file=sys.stderr,
        )

    run("run_ingest.py")
    qa = run("run_data_qa.py", soft_fail=True) or {}
    score = run("score_latest.py", soft_fail=True) or {}
    # score_latest may omit ok=True; treat n_stocks present as success
    score_ok = ("error" not in score) and (
        bool(score.get("ok")) or score.get("n_stocks") is not None
    )

    save_daily_run(
        {
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "collect_ok": bool(collect.get("ok", True)) and "error" not in collect,
            "empty_incoming": empty_incoming,
            "qa_ok": bool(qa.get("ok", True)) and "error" not in qa,
            "score_ok": score_ok,
            "coverage_pct": cov,
            "collect": {
                k: collect.get(k)
                for k in ("rows_added", "files_processed", "max_date", "coverage_pct")
            },
            "score": {
                k: score.get(k) for k in ("n_stocks", "as_of", "n_flat", "path") if k in score
            },
            "ok": True,
        },
        settings.data_processed_dir,
    )

    mon = run("run_monitor.py", soft_fail=True) or {}
    summary = {
        "collect": collect,
        "qa": qa,
        "score": score,
        "coverage_pct": cov,
        "coverage_ok": cov is None or cov >= 80,
        "monitor_status": mon.get("status") if isinstance(mon, dict) else None,
        "daily_run": str(settings.data_processed_dir / "daily_run.json"),
        "ok": True,
    }
    print("\n=== daily summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
