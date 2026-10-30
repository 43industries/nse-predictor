"""Persist monitor_status.json and append monitor_history.parquet."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings


def _processed(processed: Path | None = None) -> Path:
    return Path(processed or settings.data_processed_dir)


def status_path(processed: Path | None = None) -> Path:
    return _processed(processed) / "monitor_status.json"


def history_path(processed: Path | None = None) -> Path:
    return _processed(processed) / "monitor_history.parquet"


def daily_run_path(processed: Path | None = None) -> Path:
    return _processed(processed) / "daily_run.json"


def forecast_log_path(processed: Path | None = None) -> Path:
    return _processed(processed) / "forecast_log.parquet"


def collection_status_path(processed: Path | None = None) -> Path:
    return _processed(processed) / "collection_status.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def make_alert(code: str, severity: str, message: str, at: str | None = None) -> dict:
    sev = severity if severity in ("warn", "fail") else "warn"
    return {
        "code": str(code),
        "severity": sev,
        "message": str(message),
        "at": at or utc_now_iso(),
    }


def dedupe_alerts(alerts: list[dict]) -> list[dict]:
    """Keep highest severity per code (fail > warn)."""
    rank = {"warn": 1, "fail": 2}
    best: dict[str, dict] = {}
    for a in alerts:
        code = str(a.get("code") or "")
        if not code:
            continue
        prev = best.get(code)
        if prev is None or rank.get(a.get("severity"), 0) > rank.get(prev.get("severity"), 0):
            best[code] = a
    return list(best.values())


def overall_status(alerts: list[dict]) -> str:
    if any(a.get("severity") == "fail" for a in alerts):
        return "fail"
    if any(a.get("severity") == "warn" for a in alerts):
        return "warn"
    return "ok"


def load_monitor_status(processed: Path | None = None) -> dict:
    return read_json(status_path(processed))


def load_daily_run(processed: Path | None = None) -> dict:
    return read_json(daily_run_path(processed))


def load_collection_status(processed: Path | None = None) -> dict:
    return read_json(collection_status_path(processed))


def save_daily_run(payload: dict, processed: Path | None = None) -> Path:
    return write_json(daily_run_path(processed), {**payload, "written_at": utc_now_iso()})


def save_monitor_status(payload: dict, processed: Path | None = None) -> Path:
    return write_json(status_path(processed), payload)


def append_monitor_history(row: dict, processed: Path | None = None) -> Path:
    path = history_path(processed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame.to_parquet(path, index=False)
    return path


def load_monitor_history(processed: Path | None = None, tail: int = 14) -> pd.DataFrame:
    path = history_path(processed)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "checked_at" in df.columns:
        df = df.sort_values("checked_at")
    return df.tail(tail).reset_index(drop=True)
