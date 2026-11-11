"""Optional MLflow logging for train_evaluate (no-op if mlflow missing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_training_run(
    results: dict,
    *,
    models_dir: Path | None = None,
    experiment: str = "nse_predictor",
) -> dict[str, Any]:
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "skipped": True, "error": str(exc)}

    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="train_evaluate"):
            mlflow.log_param("macro_source", results.get("macro_source"))
            mlflow.log_param("direction_head", (results.get("selection") or {}).get("direction_head"))
            mlflow.log_param("n_features", len(results.get("model_features") or []))
            mlflow.log_param("registry_version", (results.get("feature_cache") or {}).get("registry_version"))
            dir_m = (results.get("models") or {}).get("lightgbm_direction") or {}
            if "accuracy" in dir_m:
                mlflow.log_metric("test_accuracy", float(dir_m["accuracy"]))
            if "abstain_rate" in dir_m:
                mlflow.log_metric("test_abstain_rate", float(dir_m["abstain_rate"]))
            ric = ((results.get("diagnostics") or {}).get("rank_ic") or {}).get("mean_ic")
            if ric is not None:
                mlflow.log_metric("test_rank_ic", float(ric))
            gates = results.get("validation_gates") or {}
            if gates.get("status"):
                mlflow.log_param("validation_status", gates["status"])
            bt = results.get("costed_backtest") or {}
            if bt.get("ok"):
                mlflow.log_metric("bt_total_return", float(bt.get("total_return") or 0))
                mlflow.log_metric("bt_buy_hold_return", float(bt.get("buy_hold_return") or 0))
                mlflow.log_metric("bt_max_drawdown", float(bt.get("max_drawdown") or 0))
            if models_dir and Path(models_dir).exists():
                for name in (
                    "direction_model.joblib",
                    "return_model.joblib",
                    "volatility_model.joblib",
                    "model_meta.json",
                ):
                    p = Path(models_dir) / name
                    if p.exists():
                        mlflow.log_artifact(str(p))
            return {"ok": True, "run_id": mlflow.active_run().info.run_id}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
