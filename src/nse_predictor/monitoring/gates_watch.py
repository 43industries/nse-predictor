"""Echo validation gates + light live drift vs last evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from nse_predictor.config import ROOT, settings
from nse_predictor.monitoring.status import make_alert, read_json
from nse_predictor.training.meta import load_model_meta

GATE_STALE_DAYS = 90
ABSTAIN_DRIFT_PP = 0.25


def _mtime_age_days(path: Path) -> float | None:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 86400.0


def _iso_age_days(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    except ValueError:
        return None


def _eval_abstain_baseline(eval_path: Path) -> float | None:
    ev = read_json(eval_path)
    if not ev:
        return None
    base = (ev.get("diagnostics") or {}).get("abstain_rate")
    if base is None:
        base = (ev.get("models") or {}).get("lightgbm_direction", {}).get("abstain_rate")
    if base is None or base != base:
        return None
    return float(base)


def check_gates_and_drift(
    score_block: dict | None = None,
    processed: Path | None = None,
    models: Path | None = None,
    reports_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    processed = Path(processed or settings.data_processed_dir)
    models = Path(models or settings.models_dir)
    reports_dir = Path(reports_dir or (ROOT / "reports"))
    alerts: list[dict] = []

    val_path = reports_dir / "validation_report.json"
    eval_path = reports_dir / "evaluation_results.json"
    meta = load_model_meta(models)

    gates: dict = {
        "validation_report_present": val_path.exists(),
        "validation_status": None,
        "stress_fragile": None,
        "reasons": [],
        "trained_at": meta.get("trained_at"),
        "report_age_days": _mtime_age_days(val_path),
    }

    if not val_path.exists():
        alerts.append(
            make_alert(
                "validation_report_missing",
                "warn",
                "reports/validation_report.json missing — run train_evaluate.py",
            )
        )
    else:
        payload = read_json(val_path)
        vg = payload.get("validation_gates") or {}
        gates["validation_status"] = vg.get("status")
        gates["stress_fragile"] = payload.get("stress_fragile")
        gates["reasons"] = list(vg.get("reasons") or [])
        reasons_preview = "; ".join(gates["reasons"][:3])
        if vg.get("status") == "fail":
            alerts.append(
                make_alert("validation_status_fail", "warn", f"last validation_status=fail: {reasons_preview}")
            )
        elif vg.get("status") == "warn":
            alerts.append(
                make_alert("validation_status_warn", "warn", f"last validation_status=warn: {reasons_preview}")
            )
        if gates["stress_fragile"]:
            alerts.append(make_alert("stress_fragile_echo", "warn", "validation report stress_fragile=true"))
        age = gates["report_age_days"]
        if age is not None and age > GATE_STALE_DAYS:
            alerts.append(
                make_alert(
                    "validation_stale",
                    "warn",
                    f"validation_report age {age:.0f}d > {GATE_STALE_DAYS}d",
                )
            )

    trained_at = meta.get("trained_at")
    if trained_at:
        age_m = _iso_age_days(str(trained_at))
        if age_m is not None:
            gates["model_age_days"] = age_m
            if age_m > GATE_STALE_DAYS:
                alerts.append(
                    make_alert(
                        "model_stale",
                        "warn",
                        f"model trained_at age {age_m:.0f}d > {GATE_STALE_DAYS}d",
                    )
                )
    else:
        alerts.append(make_alert("model_meta_thin", "warn", "model_meta missing trained_at"))

    live_abstain = (score_block or {}).get("abstain_rate")
    if eval_path.exists() and live_abstain is not None and live_abstain == live_abstain:
        base = _eval_abstain_baseline(eval_path)
        if base is not None:
            drift = abs(float(live_abstain) - base)
            gates["abstain_baseline"] = base
            gates["abstain_live"] = float(live_abstain)
            gates["abstain_drift"] = drift
            if drift > ABSTAIN_DRIFT_PP:
                alerts.append(
                    make_alert(
                        "abstain_drift",
                        "warn",
                        f"abstain drift {drift:.0%} > {ABSTAIN_DRIFT_PP:.0%} "
                        f"(live={live_abstain:.0%} vs eval={base:.0%})",
                    )
                )

    return gates, alerts
