"""Persist / load model_meta.json sidecar for train and score."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nse_predictor.config import settings
from nse_predictor.features.build import DEAD_ZONE_XS
from nse_predictor.features.registry import REGISTRY_VERSION


def meta_path(models_dir: Path | None = None) -> Path:
    return Path(models_dir or settings.models_dir) / "model_meta.json"


def write_model_meta(
    *,
    feature_cols: list[str],
    macro_source: str,
    direction_head: str,
    threshold: float,
    abstain_band: float,
    calibration_status: str,
    decision_mode: str,
    best_params: dict | None,
    target_return: str = "y_ret_5d_xs",
    dead_zone_xs: float = DEAD_ZONE_XS,
    extra: dict | None = None,
    models_dir: Path | None = None,
) -> Path:
    path = meta_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_cols": list(feature_cols),
        "macro_source": macro_source,
        "dead_zone_xs": dead_zone_xs,
        "direction_head": direction_head,
        "threshold": threshold,
        "abstain_band": abstain_band,
        "calibration_status": calibration_status,
        "decision_mode": decision_mode,
        "best_params": best_params or {},
        "target_return": target_return,
        "registry_version": REGISTRY_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_model_meta(models_dir: Path | None = None) -> dict:
    path = meta_path(models_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
