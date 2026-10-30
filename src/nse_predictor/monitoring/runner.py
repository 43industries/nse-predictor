"""Orchestrate all monitor checks and persist status + history."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.monitoring.gates_watch import check_gates_and_drift
from nse_predictor.monitoring.outcomes import append_forecast_log, check_outcomes
from nse_predictor.monitoring.pipeline import check_pipeline
from nse_predictor.monitoring.score_checks import check_score_hygiene
from nse_predictor.monitoring.status import (
    append_monitor_history,
    dedupe_alerts,
    make_alert,
    overall_status,
    save_monitor_status,
    utc_now_iso,
)


def run_all_checks(
    processed: Path | None = None,
    models: Path | None = None,
    append_latest_to_log: bool = True,
) -> dict:
    processed = Path(processed or settings.data_processed_dir)
    models = Path(models or settings.models_dir)
    checked_at = utc_now_iso()
    alerts: list[dict] = []

    pipeline, a = check_pipeline(processed)
    alerts.extend(a)

    score, a = check_score_hygiene(processed, models)
    alerts.extend(a)

    if append_latest_to_log:
        fc_path = processed / "latest_forecasts.parquet"
        if fc_path.exists():
            try:
                append_forecast_log(pd.read_parquet(fc_path), processed, as_of_run=checked_at)
            except Exception as exc:  # noqa: BLE001
                alerts.append(
                    make_alert(
                        "forecast_log_append",
                        "warn",
                        f"could not append forecast_log: {exc}",
                        at=checked_at,
                    )
                )

    outcomes, a = check_outcomes(processed)
    alerts.extend(a)

    gates, a = check_gates_and_drift(score_block=score, processed=processed, models=models)
    alerts.extend(a)

    alerts = dedupe_alerts(alerts)
    status = overall_status(alerts)
    payload = {
        "checked_at": checked_at,
        "status": status,
        "pipeline": pipeline,
        "score": score,
        "outcomes": outcomes,
        "gates": gates,
        "alerts": alerts,
        "n_warn": sum(1 for a in alerts if a.get("severity") == "warn"),
        "n_fail": sum(1 for a in alerts if a.get("severity") == "fail"),
    }
    save_monitor_status(payload, processed)
    append_monitor_history(
        {
            "checked_at": checked_at,
            "status": status,
            "n_warn": payload["n_warn"],
            "n_fail": payload["n_fail"],
            "coverage_pct": pipeline.get("coverage_pct"),
            "as_of": score.get("as_of"),
            "abstain_rate": score.get("abstain_rate"),
            "hit_rate": (outcomes or {}).get("hit_rate_nonflat"),
            "rank_ic": ((outcomes or {}).get("rank_ic") or {}).get("mean_ic"),
        },
        processed,
    )
    return payload


def print_monitor_summary(payload: dict) -> None:
    status = payload.get("status", "ok")
    stream = sys.stderr if status in ("warn", "fail") else sys.stdout
    print(f"MONITOR_STATUS={status}", file=stream)
    for a in payload.get("alerts") or []:
        line = f"[{a.get('severity')}] {a.get('code')}: {a.get('message')}"
        print(line, file=sys.stderr if a.get("severity") in ("warn", "fail") else sys.stdout)
