"""Pipeline health checks from daily_run.json + collection_status."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from nse_predictor.config import settings
from nse_predictor.monitoring.status import load_collection_status, load_daily_run, make_alert

COVERAGE_WARN = 80.0
DAILY_STALE_WARN_HOURS = 36
DAILY_STALE_FAIL_HOURS = 96


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _age_hours(written: datetime | None) -> float | None:
    if written is None:
        return None
    return (datetime.now(timezone.utc) - written).total_seconds() / 3600.0


def check_pipeline(processed: Path | None = None) -> tuple[dict, list[dict]]:
    processed = Path(processed or settings.data_processed_dir)
    daily = load_daily_run(processed)
    collection = load_collection_status(processed)
    alerts: list[dict] = []

    age_hours = _age_hours(_parse_ts(daily.get("written_at") or daily.get("finished_at")))
    if age_hours is not None:
        if age_hours > DAILY_STALE_FAIL_HOURS:
            alerts.append(
                make_alert(
                    "daily_run_stale",
                    "fail",
                    f"daily_run.json age {age_hours:.0f}h > {DAILY_STALE_FAIL_HOURS}h",
                )
            )
        elif age_hours > DAILY_STALE_WARN_HOURS:
            alerts.append(
                make_alert(
                    "daily_run_stale",
                    "warn",
                    f"daily_run.json age {age_hours:.0f}h > {DAILY_STALE_WARN_HOURS}h",
                )
            )
    elif not daily:
        alerts.append(
            make_alert(
                "daily_run_missing",
                "warn",
                "No daily_run.json yet — run scripts/run_daily.py",
            )
        )

    cov = daily.get("coverage_pct", collection.get("coverage_pct"))
    if cov is not None and float(cov) < COVERAGE_WARN:
        alerts.append(
            make_alert(
                "coverage_low",
                "warn",
                f"universe coverage {cov}% < {COVERAGE_WARN:.0f}%",
            )
        )

    if daily.get("qa_ok") is False:
        alerts.append(make_alert("qa_soft_fail", "warn", "Last daily QA reported soft failure"))
    if daily.get("score_ok") is False:
        alerts.append(make_alert("score_step_fail", "fail", "Last daily score step failed"))
    if daily.get("collect_ok") is False and daily.get("empty_incoming"):
        alerts.append(make_alert("collect_empty", "warn", "Collector ran with no new incoming CSVs"))

    block = {
        "daily_run_present": bool(daily),
        "daily_age_hours": age_hours,
        "coverage_pct": cov,
        "qa_ok": daily.get("qa_ok"),
        "score_ok": daily.get("score_ok"),
        "collect_ok": daily.get("collect_ok"),
        "empty_incoming": daily.get("empty_incoming"),
        "collection_max_date": collection.get("max_date"),
        "collection_last_run": collection.get("last_run"),
    }
    return block, alerts
